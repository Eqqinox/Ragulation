# Security Policy

## Supported Versions

This project is pre-release (`0.x`). Only the `main` branch receives
security fixes; there are no maintained release branches yet.

| Version | Supported |
| ------- | --------- |
| 0.x     | yes       |

## Reporting a Vulnerability

Report suspected vulnerabilities privately to meknaci81@gmail.com. Include:

- A description of the issue and its potential impact.
- Steps to reproduce, or a proof of concept.
- The affected commit or version.

Expect an acknowledgment within 5 business days. Do not open a public issue
for unpatched vulnerabilities.

## Scope

This project ingests and indexes publicly available EU regulatory texts
(EUR-Lex, EDPB) and serves them through a retrieval-augmented generation
pipeline. Retrieved and generated content is treated as untrusted data
throughout the pipeline: no retrieved or generated text is ever executed
or evaluated as code, and the only network calls are to fetch published
regulatory documents and to the local embedding/vector-store services.
