# Contributing

Contributions that improve compatibility, accessibility, tests, documentation,
or failure handling are welcome.

1. Open an issue before a breaking node-interface or provenance-format change.
2. Never add telemetry, secrets, private API implementation, or sample API keys.
3. Preserve the rule that only a real 64-hex broadcast/confirmed transaction
   reference can set `anchor_ready=true`.
4. Preserve honest C2PA language: cryptographic signing is not certification or
   universal trust.
5. Add or update contract tests and run the validation commands in README.
6. Update the changelog for user-visible changes.

Pull requests should explain compatibility impact and how the change was
tested. Provenance format changes must remain aligned with the public ImprintAI
specifications and browser/server decoders.