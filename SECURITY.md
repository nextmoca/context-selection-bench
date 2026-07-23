# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability
reporting for this repository (Security tab, "Report a vulnerability"). Do not
open a public issue for security problems.

We will acknowledge reports within 5 business days.

## Scope

This repository contains a benchmark harness that runs locally against model
APIs. There is no hosted service in this repository. Reports we care about
include:

- Anything that could cause the harness to leak API keys or credentials
- Code execution risks in fetch scripts or the deposit tooling
- Integrity issues in the manifest verification path
  (`tools/verify_manifests.py`)

## Supported versions

Only the latest release is supported with security fixes.
