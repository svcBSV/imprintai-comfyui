"""
Imprint - ComfyUI Custom Nodes for AI Provenance
Blockchain-backed metadata logging for AI-generated content

Installation:
1. Copy this file to: ComfyUI/custom_nodes/imprint/imprint_nodes.py
2. Create __init__.py in the same folder (see below)
3. Restart ComfyUI

Get an API key and documentation at: https://imprintai.link/docs#comfyui

The logging node returns a txid and anchor_ready=True only when a real
blockchain transaction is available. A skipped, pending, failed, or simulated
anchor keeps anchor_ready=False; never label an image with its job ID.
"""

import requests
import json
import os
import hashlib
import math
import re
import io
import time
from typing import Any, Optional, Tuple


__version__ = "1.0.1"

TXID_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
HASH_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
REDUNDANCY_COPIES = {"low": 1, "medium": 3, "high": 5}
SUPPORTED_STEGO_METHODS = {"imprint-stego-v1", "imprint-stego-v3"}
STATUS_POLL_TIMEOUT_SECONDS = 105.0
STATUS_POLL_INITIAL_DELAY_SECONDS = 1.0
STATUS_POLL_MAX_DELAY_SECONDS = 8.0


def _is_sha256(value: str) -> bool:
    return bool(value and HASH_PATTERN.fullmatch(value))


def _status_endpoint(api_url: str, status_url: Any, job_id: Any) -> str:
    """Resolve a status URL without sending the API key to another origin."""
    base = api_url.rstrip("/")
    if isinstance(status_url, str):
        candidate = status_url.strip()
        if candidate.startswith("/"):
            return f"{base}{candidate}"
        if candidate == base or candidate.startswith(f"{base}/"):
            return candidate
    return f"{base}/api/status/{job_id}"


def _poll_for_provenance_reference(
    api_url: str,
    status_url: Any,
    job_id: Any,
    api_key: str,
    timeout_seconds: float = STATUS_POLL_TIMEOUT_SECONDS,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> Tuple[str, str]:
    """Wait for a real broadcast reference.

    Returns (txid, reason), where txid is non-empty only for a non-mock,
    broadcast/confirmed job with a valid 64-hex transaction reference.
    """
    endpoint = _status_endpoint(api_url, status_url, job_id)
    headers = {"x-api-key": api_key}
    deadline = monotonic_fn() + max(0.0, timeout_seconds)
    delay = STATUS_POLL_INITIAL_DELAY_SECONDS

    while True:
        if monotonic_fn() >= deadline:
            return "", "timeout"

        try:
            response = requests.get(endpoint, headers=headers, timeout=15)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            result = response.json()
        except (requests.exceptions.RequestException, ValueError, TypeError):
            return "", "unavailable"

        if not isinstance(result, dict):
            return "", "invalid"

        if result.get("mock"):
            return "", "mock"

        status = str(result.get("status", "")).strip().lower()
        txid = str(result.get("txid") or "").strip().lower()

        if status in {"broadcast", "confirmed"}:
            if TXID_PATTERN.fullmatch(txid):
                return txid, ""
            return "", "invalid"

        if status == "failed":
            return "", "failed"

        if status not in {"pending", "submitted"}:
            return "", "invalid"

        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            return "", "timeout"
        sleep_fn(min(delay, remaining))
        delay = min(delay * 2, STATUS_POLL_MAX_DELAY_SECONDS)


def _canonical_pixel_hash_from_rgba(width: int, height: int, rgba: bytes) -> str:
    """icph1: SHA-256 of LSB-zeroed, row-major RGB pixels."""
    rgb = bytearray(width * height * 3)
    destination = 0
    for source in range(0, len(rgba), 4):
        rgb[destination] = rgba[source] & 0xFE
        rgb[destination + 1] = rgba[source + 1] & 0xFE
        rgb[destination + 2] = rgba[source + 2] & 0xFE
        destination += 3
    prefix = f"icph1:{width}:{height}:".encode("utf-8")
    return hashlib.sha256(prefix + bytes(rgb)).hexdigest()


def _normalise_workflow_inputs(workflow_inputs_json: str) -> Optional[Any]:
    """Parse optional JSON before including it in the deterministic summary."""
    if not workflow_inputs_json or not workflow_inputs_json.strip():
        return None
    try:
        parsed = json.loads(workflow_inputs_json)
    except json.JSONDecodeError as error:
        raise ValueError(
            "workflow_inputs_json must be valid JSON when provided; it is used "
            "to derive input_summary_hash."
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(
            "workflow_inputs_json must be a JSON object so it can be recorded "
            "as workflow settings."
        )
    return parsed


def _input_summary_hash(
    model: str,
    model_version: str,
    prompt: str,
    negative_prompt: str,
    media_type: str,
    workflow_inputs: Optional[Any],
) -> str:
    """Create a deterministic, domain-separated summary of workflow inputs."""
    summary = {
        "version": "imprint-input-summary-v1",
        "model": model or "",
        "model_version": model_version or "",
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "type": media_type or "",
        "workflow_inputs": workflow_inputs,
    }
    canonical_json = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(
        b"imprint-input-summary-v1:" + canonical_json.encode("utf-8")
    ).hexdigest()


def _single_image_to_rgba(image) -> Tuple[int, int, bytearray, int]:
    """Convert one ComfyUI IMAGE tensor (1xHxWx3/4) to RGBA bytes.

    This intentionally uses ComfyUI's in-memory image tensor only for the
    canonical pixel hash and steganographic label. It does not claim those
    pixels are the bytes of a file saved later by a Save Image node.
    """
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Imprint image-aware nodes require ComfyUI's torch and numpy runtime."
        ) from error

    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a ComfyUI IMAGE tensor.")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            "Imprint records one image per anchor. Use a single-image batch "
            "or run each generated image through its own Imprint nodes."
        )

    height, width, channels = (int(image.shape[1]), int(image.shape[2]), int(image.shape[3]))
    if width <= 0 or height <= 0 or channels not in (3, 4):
        raise ValueError("image must have a non-empty HxWx3 or HxWx4 layout.")

    pixels = (
        image[0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        # ComfyUI's native Save Image encoder converts float pixels to uint8
        # by truncation. Use the same contract before calculating icph1 and
        # writing a label, otherwise the saved PNG can hash differently.
        .to(torch.uint8)
        .numpy()
    )
    if channels == 3:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        pixels = np.concatenate((pixels, alpha), axis=2)

    return width, height, bytearray(pixels.tobytes()), channels


def _rgba_to_single_image(rgba: bytearray, width: int, height: int, channels: int):
    """Return a ComfyUI IMAGE tensor from labelled RGBA bytes."""
    import numpy as np
    import torch

    pixels = np.frombuffer(bytes(rgba), dtype=np.uint8).reshape((height, width, 4))
    output = pixels[:, :, :channels].copy()
    return torch.from_numpy(output).unsqueeze(0).float().div(255.0)


def _payload_raw_span(payload_characters: int) -> int:
    # RGB channels hold one bit each while alpha bytes are skipped.
    return (payload_characters * 8 * 4 + 2) // 3


def _check_label_capacity(rgba: bytearray, redundancy: str, txid_length: int = 64) -> Optional[str]:
    copies = REDUNDANCY_COPIES[redundancy]
    capacity = (len(rgba) * 3) // 4
    spacing = capacity // 6
    span = _payload_raw_span(4 + 1 + txid_length)  # "PROV" + length + txid

    if copies > 1 and spacing < span:
        return (
            f"Image is too small for {redundancy} redundancy: copies would overlap. "
            "Use a larger image or redundancy=low."
        )
    if ((copies - 1) * spacing) + span > len(rgba):
        return (
            f"Image is too small for {redundancy} redundancy. "
            "Use a larger image or lower redundancy."
        )
    return None


def _encode_payload_into_rgba(rgba: bytearray, payload: bytes, offset: int) -> bool:
    bit_count = len(payload) * 8
    if bit_count > ((len(rgba) * 3) // 4) - offset:
        return False

    bit_index = 0
    for raw_index in range(offset, len(rgba)):
        if (raw_index + 1) % 4 == 0:  # alpha channel
            continue
        byte_index = bit_index // 8
        shift = 7 - (bit_index % 8)
        bit = (payload[byte_index] >> shift) & 1
        rgba[raw_index] = (rgba[raw_index] & 0xFE) | bit
        bit_index += 1
        if bit_index == bit_count:
            return True
    return False


# ---------------------------------------------------------------------------
# imprint-stego-v2 — invisible transform-domain watermark, Python port
#
# Pixel-identical to the TypeScript implementation in shared/robustSteganography.ts.
# All constants, CRC polynomial, LCG parameters, and tile-geometry calculations
# match exactly so labels produced here are decodable by the server and browser.
# ---------------------------------------------------------------------------

_ROBUST_GRID_SIZE    = 48
_ROBUST_REDUNDANCY   = 7
_ROBUST_MIN_DIM      = 384
_SUBTLE_MIN_DIM      = 768
_ROBUST_TARGET_DIFF  = 26
_ROBUST_MAX_ADJ      = 18
_ROBUST_MAGIC        = b"IMR2"
_ROBUST_TXID_BYTES   = 32
_ROBUST_PAYLOAD_BYTES = 4 + _ROBUST_TXID_BYTES + 4   # magic + txid + CRC-32
_ROBUST_PAYLOAD_BITS  = _ROBUST_PAYLOAD_BYTES * 8
_ROBUST_TILE_COUNT    = _ROBUST_GRID_SIZE * _ROBUST_GRID_SIZE
_ROBUST_REQUIRED_TILES = _ROBUST_PAYLOAD_BITS * _ROBUST_REDUNDANCY
_SUBTLE_MAGIC         = b"IMS2"
_SUBTLE_TARGET_DIFF   = 16
_SUBTLE_MAX_ADJ       = 12
_V2_MAGIC             = b"IMV2"
_V2_TARGET_COEFFICIENT = 8
_V2_MIN_COEFFICIENT    = 4
_V2_MAX_AMPLITUDE     = 4


def _robust_crc32(data: bytes) -> int:
    """IEEE 802.3 CRC-32, matching the TypeScript implementation."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF


def _robust_build_tile_order() -> list:
    """Fisher-Yates shuffle with an LCG identical to the TypeScript version."""
    order = list(range(_ROBUST_TILE_COUNT))
    state = 0x9E3779B9
    for i in range(len(order) - 1, 0, -1):
        state = ((state * 1664525) + 1013904223) & 0xFFFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


_ROBUST_TILE_ORDER = _robust_build_tile_order()


def _robust_tile_rects(width: int, height: int, tile_index: int):
    """Return (left_rect, right_rect) as (x0, x1, y0, y1) tuples."""
    row    = tile_index // _ROBUST_GRID_SIZE
    column = tile_index  % _ROBUST_GRID_SIZE
    cx0 = (column * width)       // _ROBUST_GRID_SIZE
    cx1 = ((column + 1) * width) // _ROBUST_GRID_SIZE
    cy0 = (row * height)          // _ROBUST_GRID_SIZE
    cy1 = ((row + 1) * height)    // _ROBUST_GRID_SIZE
    cw  = cx1 - cx0
    ch  = cy1 - cy0
    h_inset = max(1, int(cw * 0.16))
    v_inset = max(1, int(ch * 0.20))
    center  = (cx0 + cx1) // 2
    gap     = max(1, int(cw * 0.08))
    y0 = cy0 + v_inset
    y1 = max(y0 + 1, cy1 - v_inset)
    left  = (cx0 + h_inset,              max(cx0 + h_inset + 1, center - gap), y0, y1)
    right = (min(cx1 - h_inset - 1, center + gap), cx1 - h_inset,              y0, y1)
    return left, right


def _robust_mean_luma(rgba: bytearray, width: int, rect) -> float:
    x0, x1, y0, y1 = rect
    total = 0.0
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            off = (y * width + x) * 4
            total += rgba[off] * 0.299 + rgba[off + 1] * 0.587 + rgba[off + 2] * 0.114
            count += 1
    return total / count if count else 0.0


def _robust_shift_luma(rgba: bytearray, width: int, rect, amount: float) -> None:
    x0, x1, y0, y1 = rect
    a = int(round(amount))
    for y in range(y0, y1):
        for x in range(x0, x1):
            off = (y * width + x) * 4
            rgba[off]     = max(0, min(255, rgba[off]     + a))
            rgba[off + 1] = max(0, min(255, rgba[off + 1] + a))
            rgba[off + 2] = max(0, min(255, rgba[off + 2] + a))


def _check_robust_label_capacity(width: int, height: int) -> Optional[str]:
    if width < _ROBUST_MIN_DIM or height < _ROBUST_MIN_DIM:
        return (
            f"Social-sharing-resistant watermark requires an image at least "
            f"{_ROBUST_MIN_DIM} × {_ROBUST_MIN_DIM} pixels."
        )
    return None


# ---------------------------------------------------------------------------
# imprint-stego-v2 — invisible transform-domain watermark
#
# This is the current public v2 encoder.
# ---------------------------------------------------------------------------

def _v2_cell_rect(width: int, height: int, tile_index: int):
    row = tile_index // _ROBUST_GRID_SIZE
    column = tile_index % _ROBUST_GRID_SIZE
    return (
        (column * width) // _ROBUST_GRID_SIZE,
        ((column + 1) * width) // _ROBUST_GRID_SIZE,
        (row * height) // _ROBUST_GRID_SIZE,
        ((row + 1) * height) // _ROBUST_GRID_SIZE,
    )


def _v2_dct_basis(u: int, v: int, x: float, y: float) -> float:
    alpha_u = math.sqrt(0.5) if u == 0 else 1.0
    alpha_v = math.sqrt(0.5) if v == 0 else 1.0
    return (
        0.25 * alpha_u * alpha_v *
        math.cos((math.pi * (2 * x + 1) * u) / 16) *
        math.cos((math.pi * (2 * y + 1) * v) / 16)
    )


def _v2_carrier_basis(x: float, y: float) -> float:
    return _v2_dct_basis(1, 2, x, y) - _v2_dct_basis(2, 1, x, y)


_V2_CARRIER_NORM = sum(
    _v2_carrier_basis(x, y) ** 2 for y in range(8) for x in range(8)
)


def _v2_sample_luma(rgba: bytearray, width: int, rect, sample_x: int, sample_y: int) -> float:
    x0, x1, y0, y1 = rect
    x = max(x0, min(x1 - 1, int(math.floor(x0 + ((sample_x + 0.5) * (x1 - x0)) / 8 - 0.5 + 0.5))))
    y = max(y0, min(y1 - 1, int(math.floor(y0 + ((sample_y + 0.5) * (y1 - y0)) / 8 - 0.5 + 0.5))))
    off = (y * width + x) * 4
    return rgba[off] * 0.299 + rgba[off + 1] * 0.587 + rgba[off + 2] * 0.114


def _v2_coefficient(rgba: bytearray, width: int, rect) -> float:
    return sum(
        _v2_sample_luma(rgba, width, rect, x, y) * _v2_carrier_basis(x, y)
        for y in range(8) for x in range(8)
    )


def _v2_supports_target(rgba: bytearray, width: int, rect, target: float) -> bool:
    coefficient = _v2_coefficient(rgba, width, rect)
    amplitude = max(
        -_V2_MAX_AMPLITUDE,
        min(_V2_MAX_AMPLITUDE, (target - coefficient) / _V2_CARRIER_NORM),
    )
    x0, x1, y0, y1 = rect
    projected = 0.0
    for y in range(8):
        for x in range(8):
            px = max(x0, min(x1 - 1, int(math.floor(x0 + ((x + 0.5) * (x1 - x0)) / 8))))
            py = max(y0, min(y1 - 1, int(math.floor(y0 + ((y + 0.5) * (y1 - y0)) / 8))))
            basis = _v2_carrier_basis(
                ((px - x0 + 0.5) / (x1 - x0)) * 8 - 0.5,
                ((py - y0 + 0.5) / (y1 - y0)) * 8 - 0.5,
            )
            adjustment = amplitude * basis
            off = (py * width + px) * 4
            luma = (
                _v2_round_byte(rgba[off] + adjustment) * 0.299 +
                _v2_round_byte(rgba[off + 1] + adjustment) * 0.587 +
                _v2_round_byte(rgba[off + 2] + adjustment) * 0.114
            )
            projected += luma * _v2_carrier_basis(x, y)
    return projected >= _V2_MIN_COEFFICIENT if target >= 0 else projected <= -_V2_MIN_COEFFICIENT


def _check_v2_carrier(rgba: bytearray, width: int, height: int) -> Optional[str]:
    if width < _ROBUST_MIN_DIM or height < _ROBUST_MIN_DIM:
        return f"V2 watermarking requires an image at least {_ROBUST_MIN_DIM} × {_ROBUST_MIN_DIM} pixels."
    quorum = (_ROBUST_REDUNDANCY // 2) + 1
    for bit_index in range(_ROBUST_PAYLOAD_BITS):
        negative = 0
        positive = 0
        for replica in range(_ROBUST_REDUNDANCY):
            tile = _ROBUST_TILE_ORDER[bit_index * _ROBUST_REDUNDANCY + replica]
            rect = _v2_cell_rect(width, height, tile)
            negative += _v2_supports_target(rgba, width, rect, -_V2_TARGET_COEFFICIENT)
            positive += _v2_supports_target(rgba, width, rect, _V2_TARGET_COEFFICIENT)
        if negative < quorum or positive < quorum:
            return (
                f"This image cannot safely carry the invisible v2 watermark: bit {bit_index + 1} has "
                f"{negative}/{_ROBUST_REDUNDANCY} dark and {positive}/{_ROBUST_REDUNDANCY} light "
                f"carrier directions; needs {quorum} of each."
            )
    return None


def _v2_round_byte(value: float) -> int:
    return max(0, min(255, int(math.floor(value + 0.5))))


def _embed_v2_txid(rgba: bytearray, width: int, height: int, txid: str) -> None:
    """Embed an IMV2 txid payload, matching shared/robustSteganography.ts."""
    err = _check_v2_carrier(rgba, width, height)
    if err:
        raise ValueError(err)
    if not TXID_PATTERN.fullmatch(txid):
        raise ValueError("txid must be a 64-character hexadecimal string.")

    payload = bytearray(_ROBUST_PAYLOAD_BYTES)
    payload[:4] = _V2_MAGIC
    for i in range(_ROBUST_TXID_BYTES):
        payload[4 + i] = int(txid[i * 2: i * 2 + 2], 16)
    crc = _robust_crc32(bytes(payload[:36]))
    payload[36:40] = crc.to_bytes(4, "big")

    for bit_index in range(_ROBUST_PAYLOAD_BITS):
        bit = (payload[bit_index // 8] >> (7 - (bit_index % 8))) & 1
        target = _V2_TARGET_COEFFICIENT if bit else -_V2_TARGET_COEFFICIENT
        for replica in range(_ROBUST_REDUNDANCY):
            tile = _ROBUST_TILE_ORDER[bit_index * _ROBUST_REDUNDANCY + replica]
            rect = _v2_cell_rect(width, height, tile)
            coefficient = _v2_coefficient(rgba, width, rect)
            amplitude = max(
                -_V2_MAX_AMPLITUDE,
                min(_V2_MAX_AMPLITUDE, (target - coefficient) / _V2_CARRIER_NORM),
            )
            x0, x1, y0, y1 = rect
            for y in range(y0, y1):
                for x in range(x0, x1):
                    basis = _v2_carrier_basis(
                        ((x - x0 + 0.5) / (x1 - x0)) * 8 - 0.5,
                        ((y - y0 + 0.5) / (y1 - y0)) * 8 - 0.5,
                    )
                    adjustment = amplitude * basis
                    off = (y * width + x) * 4
                    rgba[off] = _v2_round_byte(rgba[off] + adjustment)
                    rgba[off + 1] = _v2_round_byte(rgba[off + 1] + adjustment)
                    rgba[off + 2] = _v2_round_byte(rgba[off + 2] + adjustment)


# ---------------------------------------------------------------------------
# imprint-stego-v3 — public QIM watermark, matching robustSteganography.ts
# ---------------------------------------------------------------------------

_V3_GRID_SIZE = 64
_V3_REDUNDANCY = 7
_V3_EXTRA_REDUNDANCY = 4
_V3_STRONG_REDUNDANCY = _V3_REDUNDANCY + _V3_EXTRA_REDUNDANCY
_V3_MIN_DIM = 512
_V3_PILOT_BITS = 32
_V3_QUANTUM = 28
_V3_MIN_ACTIVITY = 0.25
_V3_MAGIC = b"IMV3"
_V3_LEGACY_SLOT_COUNT = (_V3_PILOT_BITS + _ROBUST_PAYLOAD_BITS) * _V3_REDUNDANCY
_V3_SLOT_COUNT = (_V3_PILOT_BITS + _ROBUST_PAYLOAD_BITS) * _V3_STRONG_REDUNDANCY


def _v3_tile_order() -> list:
    order = list(range(_V3_GRID_SIZE * _V3_GRID_SIZE))
    state = 0x517CC1B7
    for index in range(len(order) - 1, 0, -1):
        state = ((state * 1664525) + 1013904223) & 0xFFFFFFFF
        other = state % (index + 1)
        order[index], order[other] = order[other], order[index]
    return order


_V3_TILE_ORDER = _v3_tile_order()


def _v3_cell_rect(width: int, height: int, tile_index: int):
    row = tile_index // _V3_GRID_SIZE
    column = tile_index % _V3_GRID_SIZE
    return (
        (column * width) // _V3_GRID_SIZE,
        ((column + 1) * width) // _V3_GRID_SIZE,
        (row * height) // _V3_GRID_SIZE,
        ((row + 1) * height) // _V3_GRID_SIZE,
    )


def _v3_sample_point(rect, sample_x: int, sample_y: int):
    x0, x1, y0, y1 = rect
    x = max(x0, min(x1 - 1, int(math.floor(x0 + ((sample_x + 0.5) * (x1 - x0)) / 8))))
    y = max(y0, min(y1 - 1, int(math.floor(y0 + ((sample_y + 0.5) * (y1 - y0)) / 8))))
    return x, y


def _v3_basis(x: float, y: float) -> float:
    return _v2_dct_basis(1, 1, x, y)


def _v3_basis_at(x: float, y: float) -> float:
    return 0.25 * math.cos(math.pi * (2 * x + 1) / 16) * math.cos(math.pi * (2 * y + 1) / 16)


def _v3_coefficient(rgba: bytearray, width: int, rect) -> float:
    total = 0.0
    for y in range(8):
        for x in range(8):
            px, py = _v3_sample_point(rect, x, y)
            off = (py * width + px) * 4
            total += (rgba[off] * 0.299 + rgba[off + 1] * 0.587 + rgba[off + 2] * 0.114) * _v3_basis(x, y)
    return total


def _v3_activity(rgba: bytearray, width: int, rect) -> float:
    samples = []
    for y in range(8):
        for x in range(8):
            px, py = _v3_sample_point(rect, x, y)
            off = (py * width + px) * 4
            samples.append(rgba[off] * 0.299 + rgba[off + 1] * 0.587 + rgba[off + 2] * 0.114)
    total = 0.0
    count = 0
    for y in range(8):
        for x in range(8):
            index = y * 8 + x
            if x < 7:
                total += abs(samples[index] - samples[index + 1])
                count += 1
            if y < 7:
                total += abs(samples[index] - samples[index + 8])
                count += 1
    return total / count if count else 0.0


def _v3_candidates(rgba: bytearray, width: int, height: int):
    slots = _V3_TILE_ORDER[:_V3_SLOT_COUNT]
    return slots if all(
        _v3_activity(rgba, width, _v3_cell_rect(width, height, tile)) >= _V3_MIN_ACTIVITY
        for tile in slots
    ) else []


def _v3_tile_for_replica(bit_index: int, replica: int, redundancy: int) -> int:
    if redundancy == _V3_REDUNDANCY or replica < _V3_REDUNDANCY:
        return _V3_TILE_ORDER[bit_index * _V3_REDUNDANCY + replica]
    return _V3_TILE_ORDER[
        _V3_LEGACY_SLOT_COUNT + bit_index * _V3_EXTRA_REDUNDANCY + (replica - _V3_REDUNDANCY)
    ]


def _v3_nearest_target(coefficient: float, bit: int) -> float:
    center = int(math.floor(coefficient / (_V3_QUANTUM * 2) + 0.5))
    candidates = [
        (center * 2 + bit) * _V3_QUANTUM,
        ((center - 1) * 2 + bit) * _V3_QUANTUM,
        ((center + 1) * 2 + bit) * _V3_QUANTUM,
    ]
    return min(candidates, key=lambda candidate: abs(candidate - coefficient))


def _v3_bit(coefficient: float) -> int:
    return abs(int(math.floor(coefficient / _V3_QUANTUM + 0.5))) % 2


def _v3_projected_coefficient(rgba: bytearray, width: int, rect, target: float):
    coefficient = _v3_coefficient(rgba, width, rect)
    delta = target - coefficient
    x0, x1, y0, y1 = rect
    projected = 0.0
    maximum = 0.0
    for y in range(8):
        for x in range(8):
            px, py = _v3_sample_point(rect, x, y)
            basis = _v3_basis_at(
                ((px - x0 + 0.5) / (x1 - x0)) * 8 - 0.5,
                ((py - y0 + 0.5) / (y1 - y0)) * 8 - 0.5,
            )
            adjustment = delta * basis
            maximum = max(maximum, abs(adjustment))
            off = (py * width + px) * 4
            luma = (
                _v2_round_byte(rgba[off] + adjustment) * 0.299 +
                _v2_round_byte(rgba[off + 1] + adjustment) * 0.587 +
                _v2_round_byte(rgba[off + 2] + adjustment) * 0.114
            )
            projected += luma * _v3_basis(x, y)
    return projected, maximum


def _v3_supports_bit(rgba: bytearray, width: int, rect, bit: int) -> bool:
    target = _v3_nearest_target(_v3_coefficient(rgba, width, rect), bit)
    projected, maximum = _v3_projected_coefficient(rgba, width, rect, target)
    return _v3_bit(projected) == bit and abs(projected - target) <= 18 and maximum <= 7


def _check_v3_carrier(rgba: bytearray, width: int, height: int) -> Optional[str]:
    if width < _V3_MIN_DIM or height < _V3_MIN_DIM:
        return f"V3 watermarking requires an image at least {_V3_MIN_DIM} × {_V3_MIN_DIM} pixels."
    candidates = _v3_candidates(rgba, width, height)
    if len(candidates) != _V3_SLOT_COUNT:
        return (
            f"This image does not have enough stable textured carrier cells for the robust watermark. "
            f"Found fewer than {_V3_SLOT_COUNT}; use a larger or more detailed image."
        )
    for tile in candidates:
        rect = _v3_cell_rect(width, height, tile)
        if not _v3_supports_bit(rgba, width, rect, 0) or not _v3_supports_bit(rgba, width, rect, 1):
            return "This image has an unstable v3 carrier and would exceed its signed visual-change limit."
    return None


def _embed_v3_txid(rgba: bytearray, width: int, height: int, txid: str) -> None:
    err = _check_v3_carrier(rgba, width, height)
    if err:
        raise ValueError(err)
    if not TXID_PATTERN.fullmatch(txid):
        raise ValueError("txid must be a 64-character hexadecimal string.")
    payload = bytearray(_ROBUST_PAYLOAD_BYTES)
    payload[:4] = _V3_MAGIC
    for i in range(_ROBUST_TXID_BYTES):
        payload[4 + i] = int(txid[i * 2:i * 2 + 2], 16)
    payload[36:40] = _robust_crc32(bytes(payload[:36])).to_bytes(4, "big")
    candidates = _v3_candidates(rgba, width, height)
    for bit_index in range(_V3_PILOT_BITS + _ROBUST_PAYLOAD_BITS):
        bit = (bit_index % 2) if bit_index < _V3_PILOT_BITS else (
            payload[(bit_index - _V3_PILOT_BITS) // 8] >> (7 - ((bit_index - _V3_PILOT_BITS) % 8))
        ) & 1
        for replica in range(_V3_STRONG_REDUNDANCY):
            rect = _v3_cell_rect(width, height, _v3_tile_for_replica(bit_index, replica, _V3_STRONG_REDUNDANCY))
            target = _v3_nearest_target(_v3_coefficient(rgba, width, rect), bit)
            delta = target - _v3_coefficient(rgba, width, rect)
            x0, x1, y0, y1 = rect
            for y in range(y0, y1):
                for x in range(x0, x1):
                    adjustment = delta * _v3_basis_at(
                        ((x - x0 + 0.5) / (x1 - x0)) * 8 - 0.5,
                        ((y - y0 + 0.5) / (y1 - y0)) * 8 - 0.5,
                    )
                    off = (y * width + x) * 4
                    rgba[off] = _v2_round_byte(rgba[off] + adjustment)
                    rgba[off + 1] = _v2_round_byte(rgba[off + 1] + adjustment)
                    rgba[off + 2] = _v2_round_byte(rgba[off + 2] + adjustment)


def _subtle_mask_weight(rect, x: int, y: int) -> float:
    """Feathered mask matching shared/robustSteganography.ts exactly."""
    x0, x1, y0, y1 = rect
    horizontal = min((x - x0 + 0.5) / (x1 - x0), (x1 - x - 0.5) / (x1 - x0))
    vertical = min((y - y0 + 0.5) / (y1 - y0), (y1 - y - 0.5) / (y1 - y0))
    edge_distance = max(0.0, min(0.5, horizontal, vertical)) * 2
    return math.sin((math.pi / 2) * edge_distance) ** 0.5


def _subtle_mean_mask_weight(rect) -> float:
    x0, x1, y0, y1 = rect
    total = 0.0
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            total += _subtle_mask_weight(rect, x, y)
            count += 1
    return total / count if count else 0.0


def _subtle_shift_luma(rgba: bytearray, width: int, rect, amount: float) -> None:
    x0, x1, y0, y1 = rect
    for y in range(y0, y1):
        for x in range(x0, x1):
            off = (y * width + x) * 4
            adjustment = int(round(amount * _subtle_mask_weight(rect, x, y)))
            rgba[off] = max(0, min(255, rgba[off] + adjustment))
            rgba[off + 1] = max(0, min(255, rgba[off + 1] + adjustment))
            rgba[off + 2] = max(0, min(255, rgba[off + 2] + adjustment))


def _subtle_shifted_mean_luma(rgba: bytearray, width: int, rect, amount: float) -> float:
    x0, x1, y0, y1 = rect
    total = 0.0
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            off = (y * width + x) * 4
            adjustment = int(round(amount * _subtle_mask_weight(rect, x, y)))
            red = max(0, min(255, rgba[off] + adjustment))
            green = max(0, min(255, rgba[off + 1] + adjustment))
            blue = max(0, min(255, rgba[off + 2] + adjustment))
            total += red * 0.299 + green * 0.587 + blue * 0.114
            count += 1
    return total / count if count else 0.0


def _check_subtle_label_carrier(rgba: bytearray, width: int, height: int) -> Optional[str]:
    if width < _SUBTLE_MIN_DIM or height < _SUBTLE_MIN_DIM:
        return (
            f"Subtle watermarking requires an image at least "
            f"{_SUBTLE_MIN_DIM} × {_SUBTLE_MIN_DIM} pixels."
        )
    quorum = _ROBUST_REDUNDANCY // 2 + 1
    for bit_index in range(_ROBUST_PAYLOAD_BITS):
        negative_support = 0
        positive_support = 0
        for replica in range(_ROBUST_REDUNDANCY):
            tile = _ROBUST_TILE_ORDER[bit_index * _ROBUST_REDUNDANCY + replica]
            left, right = _robust_tile_rects(width, height, tile)
            difference = _robust_mean_luma(rgba, width, right) - _robust_mean_luma(rgba, width, left)
            mask_span = _subtle_mean_mask_weight(left) + _subtle_mean_mask_weight(right)
            for target in (-_SUBTLE_TARGET_DIFF, _SUBTLE_TARGET_DIFF):
                adjustment = max(
                    -_SUBTLE_MAX_ADJ,
                    min(_SUBTLE_MAX_ADJ, (target - difference) / mask_span),
                )
                projected_difference = (
                    _subtle_shifted_mean_luma(rgba, width, right, adjustment)
                    - _subtle_shifted_mean_luma(rgba, width, left, -adjustment)
                )
                supports_target = (
                    projected_difference < 0 if target < 0 else projected_difference >= 0
                )
                if supports_target:
                    if target < 0:
                        negative_support += 1
                    else:
                        positive_support += 1
        if negative_support < quorum or positive_support < quorum:
            return (
                f"This image does not leave enough tonal headroom for Subtle's "
                f"{_ROBUST_REDUNDANCY}-copy watermark (this bit has "
                f"{negative_support}/{_ROBUST_REDUNDANCY} dark and "
                f"{positive_support}/{_ROBUST_REDUNDANCY} light vote directions; "
                f"needs {quorum} of each). Choose v2 Robust or use a different "
                f"source image."
            )
    return None


def _embed_subtle_txid(rgba: bytearray, width: int, height: int, txid: str) -> None:
    """Embed a lower-visibility v2 label with an IMS2 payload."""
    err = _check_subtle_label_carrier(rgba, width, height)
    if err:
        raise ValueError(err)
    if not TXID_PATTERN.fullmatch(txid):
        raise ValueError("txid must be a 64-character hexadecimal string.")

    payload = bytearray(_ROBUST_PAYLOAD_BYTES)
    payload[:4] = _SUBTLE_MAGIC
    for i in range(_ROBUST_TXID_BYTES):
        payload[4 + i] = int(txid[i * 2: i * 2 + 2], 16)
    crc = _robust_crc32(bytes(payload[:36]))
    payload[36] = (crc >> 24) & 0xFF
    payload[37] = (crc >> 16) & 0xFF
    payload[38] = (crc >> 8) & 0xFF
    payload[39] = crc & 0xFF

    for bit_index in range(_ROBUST_PAYLOAD_BITS):
        bit = (payload[bit_index // 8] >> (7 - (bit_index % 8))) & 1
        target = _SUBTLE_TARGET_DIFF if bit else -_SUBTLE_TARGET_DIFF
        for replica in range(_ROBUST_REDUNDANCY):
            tile = _ROBUST_TILE_ORDER[bit_index * _ROBUST_REDUNDANCY + replica]
            left, right = _robust_tile_rects(width, height, tile)
            diff = _robust_mean_luma(rgba, width, right) - _robust_mean_luma(rgba, width, left)
            mask_span = _subtle_mean_mask_weight(left) + _subtle_mean_mask_weight(right)
            adj = max(-_SUBTLE_MAX_ADJ, min(_SUBTLE_MAX_ADJ, (target - diff) / mask_span))
            _subtle_shift_luma(rgba, width, left, -adj)
            _subtle_shift_luma(rgba, width, right, adj)


class ImprintLog:
    """
    Log AI workflow metadata to the BSV blockchain.
    Creates an immutable provenance record for AI-generated content.

    The record is written into an unspent 1-satoshi output using the
    imprint-chain-v1 envelope: one frozen UTXO per step, logically chained
    by previous-txid references (never spent). Anyone with a BSV explorer
    and the published format can verify it independently.

    Strongly recommended: provide `output_hash` — the SHA-256 of the exact
    saved image bytes (before any label/watermark is embedded) — and
    `canonical_pixel_hash`, which survives LSB label embedding so the
    distributed labelled file can be verified against the on-chain record.
    Both are recorded on-chain so the asset can be verified even if the
    service disappears. Use `input_summary_hash` to record a hash of the
    prompt and parameters instead of raw detail.

    Example (hash the saved file):
        sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()

    Canonical pixel hash (icph1) — sha256 over
    "icph1:<width>:<height>:" + RGB bytes (row-major) with LSBs zeroed:
        from PIL import Image
        import hashlib
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        data = img.tobytes()  # RGBA
        rgb = bytearray()
        for i in range(0, len(data), 4):
            rgb.append(data[i] & 0xFE)
            rgb.append(data[i+1] & 0xFE)
            rgb.append(data[i+2] & 0xFE)
        cph = hashlib.sha256(f"icph1:{w}:{h}:".encode() + bytes(rgb)).hexdigest()
    """

    @staticmethod
    def canonical_pixel_hash_from_image(pil_image) -> str:
        """Compute icph1 from a PIL image (kept for standalone callers)."""
        image = pil_image.convert("RGBA")
        return _canonical_pixel_hash_from_rgba(
            image.width, image.height, image.tobytes()
        )
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "SDXL", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "enable_logging": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Anchor metadata",
                        "label_off": "Skip anchoring",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "model_version": ("STRING", {"default": "", "multiline": False}),
                "workflow_id": ("STRING", {"default": "", "multiline": False}),
                "previous_txid": ("STRING", {"default": "", "multiline": False}),
                # output_file_path is the only automatic source of output_hash:
                # it must point to the exact bytes emitted by an encoder.
                "output_file_path": ("STRING", {"default": "", "multiline": False}),
                "workflow_inputs_json": ("STRING", {"default": "", "multiline": True}),
                "type": (["image_generation", "video_generation", "image_edit", "upscale", "custom"],),
                "api_url": ("STRING", {"default": "https://imprintai.link", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "txid",
        "anchor_ready",
        "output_hash",
        "canonical_pixel_hash",
        "input_summary_hash",
    )
    FUNCTION = "log_metadata"
    CATEGORY = "Imprint/Provenance"
    
    def log_metadata(
        self,
        model: str,
        api_key: str,
        enable_logging: bool = True,
        image=None,
        prompt: str = "",
        negative_prompt: str = "",
        model_version: str = "",
        workflow_id: str = "",
        previous_txid: str = "",
        output_file_path: str = "",
        output_hash: str = "",
        canonical_pixel_hash: str = "",
        input_summary_hash: str = "",
        workflow_inputs_json: str = "",
        type: str = "image_generation",
        api_url: str = "https://imprintai.link"
    ) -> Tuple[str, bool, str, str, str]:
        """Derive hashes locally, then optionally anchor a provenance assertion.

        `output_hash` is only computed automatically from `output_file_path`,
        which must name the exact already-encoded file bytes. The IMAGE tensor
        is never substituted as an exact-file hash.
        """
        try:
            workflow_inputs = _normalise_workflow_inputs(workflow_inputs_json)
            derived_input_hash = _input_summary_hash(
                model,
                model_version,
                prompt,
                negative_prompt,
                type,
                workflow_inputs,
            )
        except ValueError as error:
            print(f"[Imprint] Error: {error}")
            return ("", False, "", "", "")

        resolved_input_hash = input_summary_hash.strip().lower() or derived_input_hash
        if not _is_sha256(resolved_input_hash):
            print("[Imprint] Error: input_summary_hash must be a 64-character SHA-256 hex value")
            return ("", False, "", "", "")

        resolved_canonical_hash = canonical_pixel_hash.strip().lower()
        if image is not None:
            try:
                width, height, rgba, _ = _single_image_to_rgba(image)
                derived_canonical_hash = _canonical_pixel_hash_from_rgba(width, height, rgba)
            except (RuntimeError, ValueError) as error:
                print(f"[Imprint] Error: cannot derive canonical pixel hash: {error}")
                return ("", False, "", "", resolved_input_hash)

            if resolved_canonical_hash and resolved_canonical_hash != derived_canonical_hash:
                print(
                    "[Imprint] Error: supplied canonical_pixel_hash does not match "
                    "the connected image tensor."
                )
                return ("", False, "", "", resolved_input_hash)
            resolved_canonical_hash = derived_canonical_hash
            print(f"[Imprint] Derived icph1 canonical pixel hash for {width}x{height} image.")

        if resolved_canonical_hash and not _is_sha256(resolved_canonical_hash):
            print("[Imprint] Error: canonical_pixel_hash must be a 64-character SHA-256 hex value")
            return ("", False, "", "", resolved_input_hash)

        resolved_output_hash = output_hash.strip().lower()
        if output_file_path.strip():
            try:
                with open(output_file_path.strip(), "rb") as output_file:
                    file_hash = hashlib.sha256(output_file.read()).hexdigest()
            except OSError as error:
                print(f"[Imprint] Error: cannot read output_file_path: {error}")
                return ("", False, "", resolved_canonical_hash, resolved_input_hash)
            if resolved_output_hash and resolved_output_hash != file_hash:
                print(
                    "[Imprint] Error: supplied output_hash does not match the exact "
                    "bytes at output_file_path."
                )
                return ("", False, "", resolved_canonical_hash, resolved_input_hash)
            resolved_output_hash = file_hash
            print("[Imprint] Derived output_hash from exact encoded file bytes.")

        if resolved_output_hash and not _is_sha256(resolved_output_hash):
            print("[Imprint] Error: output_hash must be a 64-character SHA-256 hex value")
            return ("", False, "", resolved_canonical_hash, resolved_input_hash)

        if not enable_logging:
            print("[Imprint] Anchoring skipped by enable_logging=False. No API request was made.")
            return (
                "",
                False,
                resolved_output_hash,
                resolved_canonical_hash,
                resolved_input_hash,
            )

        if not api_key:
            api_key = os.getenv("IMPRINT_API_KEY", "")
        
        if not api_key:
            print("[Imprint] Error: No API key provided")
            return (
                "",
                False,
                resolved_output_hash,
                resolved_canonical_hash,
                resolved_input_hash,
            )
        
        payload = {
            "model": model,
            "type": type,
        }
        
        if prompt:
            payload["prompt"] = prompt
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if model_version:
            payload["model_version"] = model_version
        if workflow_id:
            payload["workflow_id"] = workflow_id
        if previous_txid:
            payload["previous_txid"] = previous_txid
        if resolved_output_hash:
            payload["output_hash"] = resolved_output_hash
        if resolved_canonical_hash:
            payload["canonical_pixel_hash"] = resolved_canonical_hash
        if not resolved_output_hash and not resolved_canonical_hash:
            print("[Imprint] Warning: no output_hash provided - the on-chain "
                  "record will not carry a media fingerprint. Recommended: "
                  "sha256 of the exact saved image bytes (pre-embed).")
        payload["input_summary_hash"] = resolved_input_hash
        if workflow_inputs is not None:
            payload["settings"] = workflow_inputs
        
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/api/imprint",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            result = response.json()
            
            if result.get("success"):
                txid = str(result.get("txid") or "").strip().lower()
                if txid:
                    is_mock = result.get("mock", False)
                    if is_mock:
                        print("[Imprint] Anchoring temporarily unavailable.")
                        return (
                            "",
                            False,
                            resolved_output_hash,
                            resolved_canonical_hash,
                            resolved_input_hash,
                        )
                    if not TXID_PATTERN.fullmatch(txid):
                        print("[Imprint] Anchoring failed: invalid provenance reference.")
                        return (
                            "",
                            False,
                            resolved_output_hash,
                            resolved_canonical_hash,
                            resolved_input_hash,
                        )
                    print(f"[Imprint] Provenance reference ready: {txid}")
                    return (
                        txid,
                        True,
                        resolved_output_hash,
                        resolved_canonical_hash,
                        resolved_input_hash,
                    )

                # /api/imprint can accept the request while the anchor is
                # still broadcasting. Do not turn the missing txid into a
                # successful ComfyUI result: the job ID is only a handle
                # that must be polled via status_url.
                job_id = result.get("job_id")
                if job_id:
                    status_url = result.get("status_url", "")
                    print("[Imprint] Waiting for provenance reference...")
                    txid, poll_error = _poll_for_provenance_reference(
                        api_url,
                        status_url,
                        job_id,
                        api_key,
                    )
                    if txid:
                        print(f"[Imprint] Provenance reference ready: {txid}")
                        return (
                            txid,
                            True,
                            resolved_output_hash,
                            resolved_canonical_hash,
                            resolved_input_hash,
                        )
                    if poll_error in {"timeout", "unavailable"}:
                        print("[Imprint] Anchoring temporarily unavailable.")
                    elif poll_error == "mock":
                        print("[Imprint] Anchoring temporarily unavailable.")
                    else:
                        print("[Imprint] Anchoring failed.")
                    return (
                        "",
                        False,
                        resolved_output_hash,
                        resolved_canonical_hash,
                        resolved_input_hash,
                    )

                print("[Imprint] Anchoring failed: no provenance reference was returned.")
                return (
                    "",
                    False,
                    resolved_output_hash,
                    resolved_canonical_hash,
                    resolved_input_hash,
                )
            else:
                print("[Imprint] Anchoring failed.")
                return (
                    "",
                    False,
                    resolved_output_hash,
                    resolved_canonical_hash,
                    resolved_input_hash,
                )
                
        except requests.exceptions.RequestException:
            print("[Imprint] Anchoring temporarily unavailable.")
            return (
                "",
                False,
                resolved_output_hash,
                resolved_canonical_hash,
                resolved_input_hash,
            )
        except json.JSONDecodeError:
            print("[Imprint] Invalid response from server")
            return (
                "",
                False,
                resolved_output_hash,
                resolved_canonical_hash,
                resolved_input_hash,
            )


class ImprintLabel:
    """Embed a confirmed Imprint txid in a ComfyUI image.

    Two methods are available:

    • Exact-file authenticity label (imprint-stego-v1, default) — RGB
      least-significant-bit label. Invisible
      and lossless for PNG files saved without re-encoding. Robust to nothing
      beyond bit-exact copies; a social-media download will strip it.

    • Robust watermark (imprint-stego-v3) — an `IMV3` public transform-domain watermark spread
      across a 64 × 64 grid with pilots, 7+4 redundant voting, QIM, and CRC-32 validation.
      It is designed for common JPEG/WebP recompression and moderate resizing.
      It requires at least 512 × 512 pixels and does not
      make an icph1 canonical-pixel-hash claim for transformed derivatives.

    Connect `txid` and `anchor_ready` from Imprint - Log Provenance. A pending
    job, skipped anchor, failure, or simulated transaction leaves the image
    unchanged and returns `labelled=False`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "txid": ("STRING", {"default": "", "multiline": False}),
                "anchor_ready": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "stego_method": (
                    ["imprint-stego-v1", "imprint-stego-v3"],
                    {"default": "imprint-stego-v1"},
                ),
                # redundancy only applies to imprint-stego-v1
                "redundancy": (["low", "medium", "high"], {"default": "medium"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("labelled_image", "labelled")
    FUNCTION = "embed_txid"
    CATEGORY = "Imprint/Provenance"

    def embed_txid(
        self,
        image,
        txid: str,
        anchor_ready: bool,
        stego_method: str = "imprint-stego-v1",
        redundancy: str = "medium",
    ):
        if not anchor_ready:
            print(
                "[Imprint] Image not labelled: anchor_ready is false. "
                "Only a completed real blockchain anchor may be embedded."
            )
            return (image, False)
        if stego_method not in SUPPORTED_STEGO_METHODS:
            print(
                f"[Imprint] Image not labelled: unsupported stego method '{stego_method}'. "
                "Choose imprint-stego-v1 or imprint-stego-v3."
            )
            return (image, False)
        normalised_txid = (txid or "").strip().lower()
        if not TXID_PATTERN.fullmatch(normalised_txid):
            print(
                "[Imprint] Image not labelled: txid must be a confirmed "
                "64-character hexadecimal transaction ID."
            )
            return (image, False)

        try:
            width, height, rgba, channels = _single_image_to_rgba(image)

            if stego_method == "imprint-stego-v3":
                # The redundancy dropdown is ignored for the fixed v3 carrier.
                capacity_error = _check_v3_carrier(rgba, width, height)
                if capacity_error:
                    print(f"[Imprint] Image not labelled: {capacity_error}")
                    return (image, False)
                try:
                    _embed_v3_txid(rgba, width, height, normalised_txid)
                except ValueError as embed_error:
                    print(f"[Imprint] Image not labelled: {embed_error}")
                    return (image, False)
                labelled = _rgba_to_single_image(rgba, width, height, channels)
                print(
                    f"[Imprint] Embedded confirmed txid using {stego_method}. "
                    "Connect labelled_image to Save Image to write the labelled PNG."
                )
                return (labelled, True)

            # Default: imprint-stego-v1 RGB-LSB label.
            if redundancy not in REDUNDANCY_COPIES:
                print("[Imprint] Image not labelled: redundancy must be low, medium, or high.")
                return (image, False)
            capacity_error = _check_label_capacity(rgba, redundancy, len(normalised_txid))
            if capacity_error:
                print(f"[Imprint] Image not labelled: {capacity_error}")
                return (image, False)

            payload = (
                b"PROV"
                + bytes([len(normalised_txid)])
                + normalised_txid.encode("ascii")
            )
            capacity = (len(rgba) * 3) // 4
            spacing = capacity // 6
            for copy_index in range(REDUNDANCY_COPIES[redundancy]):
                if not _encode_payload_into_rgba(rgba, payload, copy_index * spacing):
                    print(
                        "[Imprint] Image not labelled: insufficient capacity for "
                        f"{redundancy} redundancy."
                    )
                    return (image, False)

            labelled = _rgba_to_single_image(rgba, width, height, channels)
            print(
                f"[Imprint] Embedded confirmed txid with {redundancy} redundancy "
                "(imprint-stego-v1). Connect labelled_image to Save Image to write the labelled PNG."
            )
            return (labelled, True)
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"[Imprint] Image not labelled: {error}")
            return (image, False)


class ImprintExportC2paPng:
    """Write a final labelled PNG, adding a C2PA manifest when available.

    A PNG/JUMBF C2PA manifest can only survive in exact encoded PNG bytes. This
    output node asks ImprintAI to apply the label and, when configured, attach
    the already-anchored assertion without re-anchoring. It writes those exact
    response bytes into ComfyUI's output directory and deliberately exposes no
    IMAGE output that could be re-encoded by a native Save Image node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "txid": ("STRING", {"default": "", "multiline": False}),
                "anchor_ready": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "imprint", "multiline": False}),
            },
            "optional": {
                "stego_method": (
                    ["imprint-stego-v1", "imprint-stego-v3"],
                    {"default": "imprint-stego-v1"},
                ),
                # redundancy only applies to imprint-stego-v1
                "redundancy": (["low", "medium", "high"], {"default": "medium"}),
                "api_url": ("STRING", {"default": "https://imprintai.link", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("labelled_png_path", "exported", "c2pa_status")
    FUNCTION = "export_png"
    CATEGORY = "Imprint/Provenance"
    OUTPUT_NODE = True

    def export_png(
        self,
        image,
        api_key: str,
        txid: str,
        anchor_ready: bool,
        stego_method: str = "imprint-stego-v1",
        redundancy: str = "medium",
        filename_prefix: str = "imprint",
        api_url: str = "https://imprintai.link",
    ):
        if not anchor_ready:
            print(
                "[Imprint] Labelled PNG not exported: anchor_ready is false. "
                "Only a completed real provenance anchor may be labelled."
            )
            return ("", False, "")
        if stego_method not in SUPPORTED_STEGO_METHODS:
            print(
                f"[Imprint] Labelled PNG not exported: unsupported stego method '{stego_method}'. "
                "Choose imprint-stego-v1 or imprint-stego-v3."
            )
            return ("", False, "")
        normalised_txid = (txid or "").strip().lower()
        if not TXID_PATTERN.fullmatch(normalised_txid):
            print("[Imprint] Labelled PNG not exported: provenance reference must be 64 hexadecimal characters.")
            return ("", False, "")
        api_key = (api_key or os.getenv("IMPRINT_API_KEY", "")).strip()
        if not api_key:
            print("[Imprint] Labelled PNG not exported: no API key provided.")
            return ("", False, "")

        use_v3 = stego_method == "imprint-stego-v3"

        try:
            width, height, rgba, _channels = _single_image_to_rgba(image)

            if use_v3:
                # The server applies the luminance-domain watermark after its
                # canonical-pixel-hash eligibility check. Send original pixels.
                capacity_error = _check_v3_carrier(rgba, width, height)
                if capacity_error:
                    print(f"[Imprint] Labelled PNG not exported: {capacity_error}")
                    return ("", False, "")
            else:
                if redundancy not in REDUNDANCY_COPIES:
                    print("[Imprint] Labelled PNG not exported: redundancy must be low, medium, or high.")
                    return ("", False, "")
                capacity_error = _check_label_capacity(rgba, redundancy, len(normalised_txid))
                if capacity_error:
                    print(f"[Imprint] Labelled PNG not exported: {capacity_error}")
                    return ("", False, "")

            # The server receives PNG bytes and returns the same pixels with the
            # steganographic label (v1 or v3) embedded and a C2PA/JUMBF manifest
            # attached. It looks up the existing assertion by txid and API-key
            # owner, so no second blockchain anchor is created.
            try:
                from PIL import Image
            except ImportError as error:
                raise RuntimeError(
                    "Imprint C2PA export requires Pillow, which is included with ComfyUI."
                ) from error

            encoded = io.BytesIO()
            Image.frombytes("RGBA", (width, height), bytes(rgba)).save(encoded, format="PNG")
            post_data = {
                "txid": normalised_txid,
                "stegoMethod": stego_method,
            }
            if not use_v3:
                post_data["redundancy"] = redundancy
            response = requests.post(
                f"{api_url.rstrip('/')}/api/c2pa/label",
                data=post_data,
                files={"image": ("imprint-source.png", encoded.getvalue(), "image/png")},
                headers={"x-api-key": api_key},
                timeout=60,
            )
            if not getattr(response, "ok", False):
                try:
                    detail = response.json().get("error", "Unknown error")
                except (ValueError, AttributeError):
                    detail = "Unknown error"
                print(f"[Imprint] Labelled PNG not exported: {detail}")
                return ("", False, "")

            status = getattr(response, "headers", {}).get(
                "X-Imprint-C2PA-Status", ""
            ).strip().lower()
            if status not in {"signed", "not-configured", "failed"}:
                print("[Imprint] Labelled PNG not exported: invalid signing status.")
                return ("", False, "")

            try:
                import folder_paths
            except ImportError as error:
                raise RuntimeError(
                    "Imprint C2PA export must run inside a ComfyUI installation."
                ) from error

            output_dir = os.path.abspath(folder_paths.get_output_directory())
            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename_prefix.strip() or "imprint")
            output_path = os.path.join(
                output_dir, f"{safe_prefix}_{normalised_txid[:12]}.png"
            )
            if os.path.commonpath([output_dir, os.path.abspath(output_path)]) != output_dir:
                raise ValueError("filename_prefix must resolve inside ComfyUI's output directory.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as output_file:
                output_file.write(response.content)

            method_label = stego_method if use_v3 else f"imprint-stego-v1 ({redundancy} redundancy)"
            if status == "signed":
                print(
                    f"[Imprint] Exported labelled PNG with a cryptographically signed "
                    f"C2PA manifest ({method_label}): {output_path}. "
                    "Public trust depends on the verifier."
                )
            else:
                print(
                    f"[Imprint] Exported labelled PNG ({method_label}): {output_path}. "
                    "C2PA signature unavailable."
                )
            return (output_path, True, status)
        except (ImportError, RuntimeError, ValueError, OSError, requests.exceptions.RequestException) as error:
            print(f"[Imprint] Labelled PNG not exported: {error}")
            return ("", False, "")


class ImprintVerify:
    """
    Verify a blockchain transaction ID and retrieve metadata.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "txid": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "api_url": ("STRING", {"default": "https://imprintai.link", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("valid", "metadata_json")
    FUNCTION = "verify_txid"
    CATEGORY = "Imprint/Provenance"
    
    def verify_txid(
        self,
        txid: str,
        api_url: str = "https://imprintai.link"
    ) -> Tuple[bool, str]:
        """Verify a transaction and return metadata."""
        
        if not txid or len(txid) != 64:
            print("[Imprint] Invalid txid format")
            return (False, "{}")
        
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/api/verify/{txid}",
                timeout=30
            )
            
            result = response.json()
            
            if result.get("valid"):
                # The public endpoint calls this C2PA object an assertion.
                # Keep the node output aligned with that current contract.
                assertion = result.get("assertion", {})
                print(f"[Imprint] Verified txid: {txid}")
                return (True, json.dumps(assertion, indent=2))
            else:
                error = result.get("error", "Not found")
                print(f"[Imprint] Verification failed: {error}")
                return (False, "{}")
                
        except requests.exceptions.RequestException as e:
            print(f"[Imprint] Request failed: {str(e)}")
            return (False, "{}")


class ImprintAPIKey:
    """
    Load Imprint API key from environment variable.
    Set IMPRINT_API_KEY in your environment or ComfyUI startup script.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "env_var_name": ("STRING", {"default": "IMPRINT_API_KEY", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"
    CATEGORY = "Imprint/Provenance"
    
    def get_key(self, env_var_name: str = "IMPRINT_API_KEY") -> Tuple[str]:
        """Get API key from environment variable."""
        key = os.getenv(env_var_name, "")
        if key:
            print(f"[Imprint] Loaded API key from {env_var_name}")
        else:
            print(f"[Imprint] Warning: {env_var_name} not set")
        return (key,)


NODE_CLASS_MAPPINGS = {
    "ImprintLog": ImprintLog,
    "ImprintLabel": ImprintLabel,
    "ImprintExportC2paPng": ImprintExportC2paPng,
    "ImprintVerify": ImprintVerify,
    "ImprintAPIKey": ImprintAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImprintLog": "Imprint - Log Provenance",
    "ImprintLabel": "Imprint - Label Image",
    "ImprintExportC2paPng": "Imprint - Export Labelled PNG (+ C2PA)",
    "ImprintVerify": "Imprint - Verify Transaction",
    "ImprintAPIKey": "Imprint - API Key",
}
