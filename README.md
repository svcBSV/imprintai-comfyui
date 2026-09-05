# ImprintAI Provenance for ComfyUI

Opt-in ComfyUI nodes for recording AI-generation provenance with ImprintAI,
embedding a confirmed Bitcoin SV transaction reference into an image, and
exporting a labelled PNG with an optional C2PA manifest.

**Version:** 1.0.1
**Default API:** <https://imprintai.link>  
**Licence:** Apache-2.0

This repository contains client-side ComfyUI integration code only. The
ImprintAI API, account system, transaction broadcaster, signing service, and
private operational configuration are not included.

## Installation

### ComfyUI Manager

Registry publication is pending creation of the immutable ImprintAI publisher
identity. Until the Registry listing is live, use one of the manual methods
below. Do not trust an unofficial package using a similar name.

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/svcBSV/imprintai-comfyui.git
cd imprintai-comfyui
python -m pip install -r requirements.txt
```

Restart ComfyUI completely after installation or update.

### Tagged release

Download a tagged source archive from
<https://github.com/svcBSV/imprintai-comfyui/releases> and extract it to:

```text
ComfyUI/custom_nodes/imprintai-comfyui/
```

The Python files must be directly inside that directory, not another nested
folder. A versioned compatibility mirror remains at
<https://imprintai.link/comfyui/> while existing direct links migrate.

After extraction, verify the distributed source files from the package root:

```bash
sha256sum -c CHECKSUMS.sha256
```

`CHECKSUMS.sha256` covers the listed source and documentation files inside the
release; it is not a checksum of GitHub's generated ZIP or tar archive.

## API key

Create an approved ImprintAI account and API key through
<https://imprintai.link>. The recommended approach is to set
`IMPRINT_API_KEY` in the environment used to start ComfyUI, then connect
**Imprint - API Key** to the nodes that require it. Do not put an API key in a
workflow intended for sharing.

No node sends media or metadata when `enable_logging` is false. When logging or
final export is enabled, the relevant metadata and image are sent to the
configured API. ImprintAI processes uploaded media transiently; prompts and
other submitted metadata may be retained and published unencrypted on a public
blockchain. Omit private prompt text or submit an appropriate summary/hash.

## Safe workflow

```text
KSampler -> VAE Decode (IMAGE) ------------+-> Imprint - Log Provenance
                                            |            |
                                            |            +-> txid ---------+
                                            |            +-> anchor_ready -+
                                            |                              |
                                            +------------------------------+
                                                                           v
                                      Imprint - Export Labelled PNG (+ C2PA)
                                                                           |
                                                                           v
                                                              final PNG in output/
```

Only a real, non-mock `broadcast` or `confirmed` transaction produces a
64-hex `txid` with `anchor_ready=true`. A job ID is never a provenance
reference and must never be embedded. Pending jobs are polled for a bounded
period; timeout or failure leaves the reference empty and the anchor unready.

Do not route the dedicated export back through ComfyUI's native **Save Image**
node. Re-encoding can remove C2PA/JUMBF metadata. The exporter intentionally
returns a file path rather than an IMAGE preview.

## Nodes

### Imprint - API Key

Loads the API key from `IMPRINT_API_KEY`, or another explicitly selected
environment-variable name. It does not store or transmit the key by itself.

### Imprint - Log Provenance

Calculates the public `icph1` canonical pixel hash from one connected IMAGE,
derives a deterministic input-summary hash, and optionally asks ImprintAI to
anchor metadata. It returns:

- `txid`: a real 64-hex transaction reference, or an empty string
- `anchor_ready`: true only when that reference is eligible for labelling
- `output_hash`: SHA-256 only when an exact already-encoded file path is given
- `canonical_pixel_hash`: LSB-invariant `icph1` hash of connected pixels
- `input_summary_hash`: deterministic hash of supplied generation inputs

The optional prompts and workflow JSON may become public. The exact-file hash
is deliberately separate from the in-memory image hash.

### Imprint - Label Image

Embeds a confirmed reference into an IMAGE tensor for stego-only workflows.
`imprint-stego-v1` authenticates the original canonical pixels.
`imprint-stego-v3` is transform-resistant and may recover the reference after
common resizing/recompression, but does not prove transformed pixel equality.
The legacy v2 method is decode-only and is not offered for new labels.

Native Save Image may be used for this stego-only output, but it cannot
preserve a C2PA manifest.

### Imprint - Export Labelled PNG (+ C2PA)

Sends the source image and confirmed reference to the dedicated finalization
endpoint and writes the exact returned PNG bytes. A valid labelled PNG remains
exportable when C2PA signing is unavailable or fails.

The `c2pa_status` output is one of:

- `signed`: a cryptographic C2PA signature was added
- `not-configured`: no signer was available
- `failed`: signing failed, but the returned PNG still has its provenance label

A cryptographically signed manifest does not imply C2PA certification,
membership, public trust-list inclusion, or acceptance by every verifier.
Trust is verifier-dependent.

### Imprint - Verify Transaction

Queries the public verification endpoint for a 64-hex transaction reference
and returns the current assertion metadata. This verifies the service's chain
record; exact-file v1 and transformed-copy v3 have different media claims as
described above.

## Versioning and compatibility

Releases follow semantic versioning. Patch releases preserve node interfaces,
minor releases may add backwards-compatible inputs or nodes, and major
releases may change workflows or API contracts. The package version appears in
`pyproject.toml`, `release.json`, and `imprint_nodes.__version__`; CI requires
them to match.

The default API URL is stable, but every networked node exposes `api_url` for
self-hosted or development use. Cross-origin status URLs returned during
polling are ignored so an API key is not sent to another origin.

## Development

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/validate_distribution.py
```

The repository contains ready-to-install workflow definitions for contract
tests, Registry metadata validation, version matching, basic secret scanning,
release archives, checksums, and official Registry publication. They remain
templates until a repository administrator grants workflow-file permission and
places them under `.github/workflows/`. See
[workflow-templates](workflow-templates/) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Support and security

Use GitHub Issues for reproducible node defects and documentation problems.
Account approval, billing, transaction, and hosted API incidents belong to the
ImprintAI service support channel rather than this client repository.

Do not open a public issue for a suspected credential leak or exploitable
security problem. Follow [SECURITY.md](SECURITY.md).