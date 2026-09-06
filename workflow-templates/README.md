# Workflow templates

These definitions are ready to copy to `.github/workflows/` once a repository
administrator authorizes workflow-file changes:

- `validate.yml` runs contract and distribution checks
- `release.yml` builds tagged ZIP archives and SHA-256 files
- `publish-registry.yml` publishes manually through the official Comfy action

Registry publication requires adding `REGISTRY_ACCESS_TOKEN` as a GitHub
Actions secret for the configured immutable `imprintai` publisher. Never
commit that token.