# Changelog

All notable changes use semantic versioning.

## 1.2.0

- Added explicit sparse-safe V4 to Label Image and Export Labelled PNG (+ C2PA)
- Preserved V1 defaults and frozen V3 compatibility; V4 never silently replaces V3
- Enforced V4's 512 × 512 minimum, maximum 32 unsafe skipped cells and ten safe
  copies of each pilot/payload bit; unsafe cells remain untouched
- Documented API selection, reference-only verification and transformation limits
- Updated Pillow minimum to 12.3.0 and clarified Registry versus mirror versions

## 1.1.0

- Added opt-in `imprint-prompt-enc-v1` prompt encryption using AES-256-GCM,
  PBKDF2-SHA256 passphrases, or a portable raw 32-byte local key file
- Added local salted input-summary hashes and prompt-size limits so encryption
  never falls back to plaintext

## 1.0.1

- Added the complete Apache-2.0 licence text
- Corrected extracted-source checksum instructions and release packaging
- Published GitHub Actions definitions as inactive templates until repository
  workflow permission is granted

## 1.0.0

- Initial public package of the ImprintAI ComfyUI provenance nodes
- Real-reference polling with cross-origin API-key protection
- Exact-pixel v1 and transform-resistant v3 labelling
- Dedicated labelled PNG export with explicit C2PA signing status
- Environment-based API-key node and public transaction verification