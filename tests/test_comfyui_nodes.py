"""Focused contract checks for the bundled ComfyUI nodes.

Run with: python tests/comfyui_nodes.test.py
"""

import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


NODE_PATH = pathlib.Path(__file__).parents[1] / "imprint_nodes.py"
requests_stub = types.ModuleType("requests")
requests_stub.exceptions = types.SimpleNamespace(
    RequestException=RuntimeError,
)
requests_stub.post = None
requests_stub.get = None
sys.modules.setdefault("requests", requests_stub)
spec = importlib.util.spec_from_file_location("imprint_nodes", NODE_PATH)
nodes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nodes
spec.loader.exec_module(nodes)


class Response:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class ComfyNodeContractTests(unittest.TestCase):
    def test_log_node_derives_hashes_without_manual_hash_inputs(self):
        inputs = nodes.ImprintLog.INPUT_TYPES()
        optional_inputs = inputs["optional"]

        self.assertNotIn("output_hash", optional_inputs)
        self.assertNotIn("canonical_pixel_hash", optional_inputs)
        self.assertNotIn("input_summary_hash", optional_inputs)
        self.assertEqual(
            nodes.ImprintLog.RETURN_NAMES,
            (
                "txid",
                "anchor_ready",
                "output_hash",
                "canonical_pixel_hash",
                "input_summary_hash",
            ),
        )

    def test_pending_anchor_polls_until_reference_is_ready(self):
        txid = "c" * 64
        response = Response({
            "success": True,
            "job_id": "job-123",
            "status": "pending",
            "status_url": "/api/status/job-123",
        })
        pending = Response({"status": "pending", "txid": None})
        ready = Response({"status": "broadcast", "txid": txid})
        with patch.object(nodes.requests, "post", return_value=response), patch.object(
            nodes.requests, "get", side_effect=[pending, ready]
        ), patch.object(nodes.time, "sleep"):
            actual, success, output_hash, canonical_hash, input_hash = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(actual, txid)
        self.assertTrue(success)
        self.assertEqual(output_hash, "")
        self.assertEqual(canonical_hash, "")
        self.assertRegex(input_hash, r"^[a-f0-9]{64}$")

    def test_pending_anchor_failure_never_returns_job_id(self):
        response = Response({
            "success": True,
            "job_id": "job-123",
            "status": "pending",
        })
        failed = Response({"status": "failed", "txid": None, "error": "provider failed"})
        with patch.object(nodes.requests, "post", return_value=response), patch.object(
            nodes.requests, "get", return_value=failed
        ):
            actual, success, _, _, _ = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(actual, "")
        self.assertFalse(success)

    def test_pending_mock_anchor_stays_unready(self):
        response = Response({
            "success": True,
            "job_id": "job-123",
            "status_url": "/api/status/job-123",
        })
        mock_status = Response({
            "status": "broadcast",
            "txid": "d" * 64,
            "mock": True,
        })
        with patch.object(nodes.requests, "post", return_value=response), patch.object(
            nodes.requests, "get", return_value=mock_status
        ):
            actual, success, _, _, _ = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(actual, "")
        self.assertFalse(success)

    def test_status_poll_timeout_returns_no_reference(self):
        clock = Mock(side_effect=[0.0, 0.0, 1.0, 1.0])
        with patch.object(
            nodes.requests,
            "get",
            return_value=Response({"status": "pending", "txid": None}),
        ):
            txid, reason = nodes._poll_for_provenance_reference(
                "https://imprintai.link",
                "/api/status/job-123",
                "job-123",
                "key",
                timeout_seconds=1.0,
                sleep_fn=lambda _delay: None,
                monotonic_fn=clock,
            )
        self.assertEqual(txid, "")
        self.assertEqual(reason, "timeout")

    def test_external_status_url_falls_back_to_api_origin(self):
        self.assertEqual(
            nodes._status_endpoint(
                "https://imprintai.link/",
                "https://attacker.invalid/collect",
                "job-123",
            ),
            "https://imprintai.link/api/status/job-123",
        )

    def test_immediate_anchor_returns_transaction_id(self):
        txid = "a" * 64
        response = Response({"success": True, "txid": txid})
        with patch.object(nodes.requests, "post", return_value=response):
            actual, success, output_hash, canonical_hash, input_hash = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(actual, txid)
        self.assertTrue(success)
        self.assertEqual(output_hash, "")
        self.assertEqual(canonical_hash, "")
        self.assertRegex(input_hash, r"^[a-f0-9]{64}$")

    def test_immediate_invalid_reference_is_not_ready(self):
        response = Response({"success": True, "txid": "job-123"})
        with patch.object(nodes.requests, "post", return_value=response):
            actual, success, _, _, _ = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(actual, "")
        self.assertFalse(success)

    def test_disabled_logging_makes_no_request_but_returns_derived_hashes(self):
        with patch.object(nodes.requests, "post") as post:
            txid, ready, output_hash, canonical_hash, input_hash = nodes.ImprintLog().log_metadata(
                model="SDXL",
                api_key="",
                enable_logging=False,
                prompt="a test prompt",
                workflow_inputs_json='{"seed": 42, "steps": 30}',
            )
        post.assert_not_called()
        self.assertEqual(txid, "")
        self.assertFalse(ready)
        self.assertEqual(output_hash, "")
        self.assertEqual(canonical_hash, "")
        self.assertRegex(input_hash, r"^[a-f0-9]{64}$")

    def test_mock_response_is_not_eligible_for_labelling(self):
        response = Response({"success": True, "txid": "b" * 64, "mock": True})
        with patch.object(nodes.requests, "post", return_value=response):
            txid, ready, _, _, _ = nodes.ImprintLog().log_metadata(
                model="SDXL", api_key="key"
            )
        self.assertEqual(txid, "")
        self.assertFalse(ready)

    def test_exact_output_hash_comes_only_from_file_bytes(self):
        with tempfile.NamedTemporaryFile() as output:
            output.write(b"exact encoded image bytes")
            output.flush()
            with patch.object(nodes.requests, "post") as post:
                _, ready, output_hash, _, _ = nodes.ImprintLog().log_metadata(
                    model="SDXL",
                    api_key="",
                    enable_logging=False,
                    output_file_path=output.name,
                )
        post.assert_not_called()
        self.assertFalse(ready)
        self.assertEqual(
            output_hash,
            hashlib.sha256(b"exact encoded image bytes").hexdigest(),
        )

    def test_lsb_embedding_preserves_icph1_and_uses_prov_payload(self):
        width, height = 64, 64
        rgba = bytearray([120, 121, 122, 255] * (width * height))
        before = nodes._canonical_pixel_hash_from_rgba(width, height, rgba)
        txid = "c" * 64
        payload = b"PROV" + bytes([len(txid)]) + txid.encode("ascii")
        self.assertIsNone(nodes._check_label_capacity(rgba, "medium"))

        capacity = (len(rgba) * 3) // 4
        spacing = capacity // 6
        for copy_index in range(nodes.REDUNDANCY_COPIES["medium"]):
            self.assertTrue(
                nodes._encode_payload_into_rgba(rgba, payload, copy_index * spacing)
            )

        self.assertEqual(
            before, nodes._canonical_pixel_hash_from_rgba(width, height, rgba)
        )

        decoded_bits = []
        for index in range(0, len(rgba)):
            if (index + 1) % 4 != 0:
                decoded_bits.append(str(rgba[index] & 1))
            if len(decoded_bits) == len(payload) * 8:
                break
        decoded = bytes(
            int("".join(decoded_bits[offset:offset + 8]), 2)
            for offset in range(0, len(decoded_bits), 8)
        )
        self.assertEqual(decoded, payload)

    def test_label_node_refuses_unready_or_invalid_inputs(self):
        original_image = object()
        unchanged, labelled = nodes.ImprintLabel().embed_txid(
            original_image, "job-123", False
        )
        self.assertIs(unchanged, original_image)
        self.assertFalse(labelled)
        unchanged, labelled = nodes.ImprintLabel().embed_txid(
            original_image, "not-a-txid", True
        )
        self.assertIs(unchanged, original_image)
        self.assertFalse(labelled)

    def test_label_node_normalises_txid_before_embedding(self):
        width, height = 64, 64
        rgba = bytearray([120, 121, 122, 255] * (width * height))
        txid = "D" * 64
        with patch.object(
            nodes,
            "_single_image_to_rgba",
            return_value=(width, height, rgba, 3),
        ), patch.object(
            nodes,
            "_rgba_to_single_image",
            return_value="labelled-image",
        ):
            labelled_image, labelled = nodes.ImprintLabel().embed_txid(
                object(), f"  {txid}  ", True,
                stego_method="imprint-stego-v1", redundancy="low",
            )

        self.assertTrue(labelled)
        self.assertEqual(labelled_image, "labelled-image")
        bits = []
        for index in range(len(rgba)):
            if (index + 1) % 4 != 0:
                bits.append(str(rgba[index] & 1))
            if len(bits) == (4 + 1 + 64) * 8:
                break
        payload = bytes(
            int("".join(bits[offset:offset + 8]), 2)
            for offset in range(0, len(bits), 8)
        )
        self.assertEqual(payload, b"PROV" + bytes([64]) + txid.lower().encode("ascii"))

    def test_robust_capacity_check_rejects_small_images(self):
        self.assertIsNotNone(nodes._check_robust_label_capacity(383, 384))
        self.assertIsNotNone(nodes._check_robust_label_capacity(384, 383))
        self.assertIsNone(nodes._check_robust_label_capacity(384, 384))
        self.assertIsNone(nodes._check_robust_label_capacity(1024, 1024))

    def test_v2_is_not_offered_or_accepted(self):
        methods = nodes.ImprintLabel.INPUT_TYPES()["optional"]["stego_method"][0]
        export_methods = nodes.ImprintExportC2paPng.INPUT_TYPES()["optional"]["stego_method"][0]
        self.assertEqual(methods, ["imprint-stego-v1", "imprint-stego-v3"])
        self.assertEqual(export_methods, ["imprint-stego-v1", "imprint-stego-v3"])
        self.assertNotIn("imprint-stego-v2", methods)
        self.assertNotIn("imprint-stego-v2", export_methods)
        original_image = object()
        unchanged, labelled = nodes.ImprintLabel().embed_txid(
            original_image, "e" * 64, True, stego_method="imprint-stego-v2"
        )
        self.assertIs(unchanged, original_image)
        self.assertFalse(labelled)

    def test_v3_label_node_refuses_small_image(self):
        # A 300×300 image is below the v3 minimum.
        width, height = 300, 300
        rgba = bytearray([100, 120, 140, 255] * (width * height))
        original_image = object()
        with patch.object(
            nodes,
            "_single_image_to_rgba",
            return_value=(width, height, rgba, 3),
        ):
            result_image, labelled = nodes.ImprintLabel().embed_txid(
                original_image, "e" * 64, True,
                stego_method="imprint-stego-v3",
            )
        self.assertFalse(labelled)
        self.assertIs(result_image, original_image)

    def test_invisible_v3_label_node_embeds_with_bounded_modulation(self):
        width, height = 768, 768
        rgba = bytearray()
        for y in range(height):
            for x in range(width):
                v = 100 + (x % 31) + (y % 23)
                rgba += bytes([v, v + 10, v + 20, 255])
        original = bytes(rgba)

        with patch.object(
            nodes, "_single_image_to_rgba",
            return_value=(width, height, rgba, 3),
        ), patch.object(
            nodes,
            "_rgba_to_single_image",
            return_value="labelled-image",
        ):
            labelled_image, labelled = nodes.ImprintLabel().embed_txid(
                object(), "a1b2c3d4" * 8, True,
                stego_method="imprint-stego-v3",
            )

        self.assertTrue(labelled)
        self.assertEqual(labelled_image, "labelled-image")
        self.assertIn(
            "imprint-stego-v3",
            nodes.ImprintLabel.INPUT_TYPES()["optional"]["stego_method"][0],
        )
        largest_change = max(
            abs(rgba[index] - original[index]) for index in range(len(rgba))
        )
        self.assertGreater(largest_change, 0)
        self.assertLessEqual(largest_change, 7)

    def test_invisible_v3_preflight_rejects_small_images(self):
        width, height = 511, 511
        rgba = bytearray([128, 128, 128, 255] * (width * height))
        error = nodes._check_v3_carrier(rgba, width, height)
        self.assertIsNotNone(error)
        self.assertIn("512", error)

    def test_invisible_v3_preflight_accepts_textured_carrier(self):
        width, height = 768, 768
        rgba = bytearray()
        for y in range(height):
            for x in range(width):
                v = 100 + (x % 31) + (y % 23)
                rgba += bytes([v, v + 10, v + 20, 255])
        self.assertIsNone(nodes._check_v3_carrier(rgba, width, height))
        nodes._embed_v3_txid(rgba, width, height, "a1b2c3d4" * 8)

    def test_robust_crc32_matches_reference_vector(self):
        # CRC-32 of b"\x00" * 4 is a known IEEE 802.3 value.
        self.assertEqual(nodes._robust_crc32(b""), 0x00000000)
        self.assertEqual(nodes._robust_crc32(b"\x00" * 4), 0x2144DF1C)

    def test_c2pa_export_refuses_unready_anchor_without_request(self):
        original_image = object()
        with patch.object(nodes.requests, "post") as post:
            path, exported, status = nodes.ImprintExportC2paPng().export_png(
                original_image, "key", "a" * 64, False
            )
        post.assert_not_called()
        self.assertEqual(path, "")
        self.assertFalse(exported)
        self.assertEqual(status, "")
        self.assertEqual(
            nodes.ImprintExportC2paPng.RETURN_NAMES,
            ("labelled_png_path", "exported", "c2pa_status"),
        )
        self.assertNotIn("IMAGE", nodes.ImprintExportC2paPng.RETURN_TYPES)

    def test_c2pa_export_writes_labelled_png_when_signature_unavailable(self):
        import types as _types

        width, height = 64, 64
        rgba = bytearray([120, 121, 122, 255] * (width * height))
        txid = "b" * 64
        returned_png = b"UNSIGNED_LABELLED_PNG"

        class FakeResponse:
            ok = True
            content = returned_png
            headers = {"X-Imprint-C2PA-Status": "not-configured"}

        class _FakePILImg:
            def save(self, buf, format=None):
                buf.write(b"FAKE_PNG_BYTES")

        fake_pil_image_mod = _types.ModuleType("PIL.Image")
        fake_pil_image_mod.frombytes = lambda mode, size, data: _FakePILImg()
        fake_pil_mod = _types.ModuleType("PIL")
        fake_pil_mod.Image = fake_pil_image_mod

        with tempfile.TemporaryDirectory() as output_dir:
            fake_folder_paths = _types.ModuleType("folder_paths")
            fake_folder_paths.get_output_directory = lambda: output_dir
            with patch.object(nodes.requests, "post", return_value=FakeResponse()), \
                 patch.object(
                     nodes, "_single_image_to_rgba",
                     return_value=(width, height, rgba, 3),
                 ), \
                 patch.dict("sys.modules", {
                     "folder_paths": fake_folder_paths,
                     "PIL": fake_pil_mod,
                     "PIL.Image": fake_pil_image_mod,
                 }):
                path, exported, status = nodes.ImprintExportC2paPng().export_png(
                    object(),
                    api_key="testkey",
                    txid=txid,
                    anchor_ready=True,
                    filename_prefix="unsigned",
                )

            self.assertTrue(exported)
            self.assertEqual(status, "not-configured")
            self.assertEqual(pathlib.Path(path).read_bytes(), returned_png)

    def test_c2pa_export_v3_forwards_stego_method_and_skips_v1_capacity(self):
        """ImprintExportC2paPng with v3 forwards the method."""
        import types as _types
        import io as _io

        width, height = 512, 512
        rgba = bytearray()
        for y in range(height):
            for x in range(width):
                v = 100 + (x % 31) + (y % 23)
                rgba += bytes([v, v + 10, v + 20, 255])
        txid = "a1b2c3d4" * 8

        captured = {}

        class FakeResponse:
            ok = True
            content = b"PNG_DATA"
            headers = {"X-Imprint-C2PA-Status": "signed"}

            def json(self):
                return {}

        fake_folder_paths = _types.ModuleType("folder_paths")
        fake_folder_paths.get_output_directory = lambda: "/tmp/imprint_test_output"

        # Minimal PIL.Image stub — only the frombytes path is exercised here.
        class _FakePILImg:
            def save(self, buf, format=None):
                buf.write(b"FAKE_PNG_BYTES")

        fake_pil_image_mod = _types.ModuleType("PIL.Image")
        fake_pil_image_mod.frombytes = lambda mode, size, data: _FakePILImg()
        fake_pil_mod = _types.ModuleType("PIL")
        fake_pil_mod.Image = fake_pil_image_mod

        def fake_post(url, data=None, files=None, headers=None, timeout=None):
            captured["data"] = data
            captured["files"] = files
            return FakeResponse()

        with patch.object(nodes.requests, "post", side_effect=fake_post), \
             patch.object(
                 nodes, "_single_image_to_rgba",
                 return_value=(width, height, rgba, 3),
             ), \
             patch.dict("sys.modules", {
                 "folder_paths": fake_folder_paths,
                 "PIL": fake_pil_mod,
                 "PIL.Image": fake_pil_image_mod,
             }):
            path, exported, status = nodes.ImprintExportC2paPng().export_png(
                object(),
                api_key="testkey",
                txid=txid,
                anchor_ready=True,
                stego_method="imprint-stego-v3",
                filename_prefix="test",
            )

        self.assertTrue(exported, "export_png should succeed with a signed response")
        self.assertEqual(status, "signed")
        # stegoMethod must be forwarded to the server
        self.assertEqual(captured["data"].get("stegoMethod"), "imprint-stego-v3")
        # redundancy should not be forwarded for v3
        self.assertNotIn("redundancy", captured["data"])
        # The image file must have been sent
        self.assertIn("image", captured["files"])

    def test_c2pa_export_v3_rejects_small_carrier_before_request(self):
        width, height = 511, 511
        rgba = bytearray([128, 128, 128, 255] * (width * height))

        with patch.object(nodes.requests, "post") as post, patch.object(
            nodes,
            "_single_image_to_rgba",
            return_value=(width, height, rgba, 3),
        ):
            path, exported, status = nodes.ImprintExportC2paPng().export_png(
                object(),
                api_key="testkey",
                txid="a1b2c3d4" * 8,
                anchor_ready=True,
                stego_method="imprint-stego-v3",
            )

        post.assert_not_called()
        self.assertEqual(path, "")
        self.assertFalse(exported)
        self.assertEqual(status, "")

    def test_verify_returns_current_assertion(self):
        assertion = {"actions": [{"action": "c2pa.created"}], "custom": {"x": 1}}
        response = Response({"valid": True, "assertion": assertion})
        with patch.object(nodes.requests, "get", return_value=response):
            valid, output = nodes.ImprintVerify().verify_txid("a" * 64)
        self.assertTrue(valid)
        self.assertEqual(json.loads(output), assertion)


if __name__ == "__main__":
    unittest.main()
