# Publishing `bvchen` to PyPI

This project is already packaged according to the official Python
Packaging Authority (PyPA) recommendations (PEP 621 `pyproject.toml`,
src-layout, wheel + sdist). Follow these steps to publish it on
[PyPI](https://pypi.org) — the official Python package index.

## 0. One-time preparation

1. Create an account on <https://pypi.org> (and optionally on
   <https://test.pypi.org> for dry runs).
2. Metadata is already filled in (`pyproject.toml`): authors Mukul
   Bijalwan and Puneet Kumar Gupta. Before each release just bump
   `version`, and point `[project.urls]` at your final GitHub repository
   if its name differs.
3. Make sure the package builds cleanly:

   ```bash
   pip install build twine
   python -m build
   twine check dist/*
   ```

## Option A — Trusted publishing (recommended, no tokens)

1. Push this repository to GitHub.
2. On PyPI go to *Account settings → Publishing* and add a **pending
   trusted publisher**:
   - owner: `<your-github-username>`
   - repository: `bvchen`
   - workflow: `publish.yml`
   - environment: `pypi`
3. Create a release (or push a tag `v0.1.0`). The included workflow
   `.github/workflows/publish.yml` then builds the artifacts and uploads
   them automatically via OIDC.

## Option B — Upload with an API token

```bash
pip install build twine
python -m build

# upload to TestPyPI first (recommended)
twine upload --repository testpypi dist/*

# check it installs:
pip install --index-url https://test.pypi.org/simple/ bvchen

# finally, upload to PyPI
twine upload dist/*
```

When asked for credentials use `__token__` as username and your API
token (from *Account settings → API tokens*) as password.

## After publishing

- The package page appears at `https://pypi.org/project/bvchen/`.
- Users can install it with `pip install bvchen`.
- For a new version: update `version` in `pyproject.toml` and
  `__init__.py`, rebuild, re-upload.

## Notes

- The name `bvchen` was chosen to match the reference R package
  (`BvChen`); check its availability on PyPI before your first upload.
- Keep `dist/` out of git (already in `.gitignore`); CI rebuilds it.
