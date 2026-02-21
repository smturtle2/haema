# Release Guide

## Overview

PyPI publishing is automated by:

- `.github/workflows/publish-pypi.yml`

Trigger conditions:

- push tag `v*` (example: `v0.1.0`)
- manual `workflow_dispatch`

## GitHub Actions Jobs

`build` job:

- install build tooling
- build `sdist` and wheel (`python -m build`)
- run `twine check dist/*`
- upload `dist/` artifact

`publish` job:

- download `dist/` artifact
- publish via `pypa/gh-action-pypi-publish@release/v1`
- uses OIDC Trusted Publishing

## One-Time PyPI Setup

1. Create the project on PyPI.
2. Configure Trusted Publisher on PyPI:
   - GitHub owner/repository
   - workflow file: `publish-pypi.yml`
   - environment: `pypi`
3. Ensure GitHub Actions permission for `id-token: write` (already in workflow).

## Release Steps

1. Update version in `pyproject.toml`.
2. Commit changes.
3. Create and push tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

4. Verify workflow run in GitHub Actions.
5. Verify package on PyPI.
