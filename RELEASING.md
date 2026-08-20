# Releasing

This repository is `cuda-planning`; what it publishes to PyPI is **`cuplan`**,
the same name you import. Releases run from a tag, through
[`.github/workflows/release.yml`](.github/workflows/release.yml). The workflow
authenticates with [Trusted Publishing][tp], so no API token is stored in the
repository.

[tp]: https://docs.pypi.org/trusted-publishers/

## One-time PyPI setup

The project does not exist on PyPI yet, so the first release needs a *pending*
publisher — a trusted publisher declared before the project's first upload.

1. Sign in to PyPI → **Your projects** → **Publishing** →
   <https://pypi.org/manage/account/publishing/>.
2. Under "Add a new pending publisher", fill in:

   | Field | Value |
   | :--- | :--- |
   | PyPI project name | `cuplan` |
   | Owner | `openplan-labs` |
   | Repository name | `cuda-planning` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first two rows differ on purpose: the repository is `cuda-planning`,
   the distribution is `cuplan`. PyPI matches the publisher against the name
   in the package metadata, not the repository, and a mismatch fails the
   upload with "Non-user identities cannot create new projects".

3. Optionally repeat on [TestPyPI](https://test.pypi.org/manage/account/publishing/)
   with environment `testpypi`, which enables the rehearsal below.

The environment names matter: the workflow's publish jobs run in GitHub
environments called `pypi` and `testpypi`, and PyPI checks that claim in the
OIDC token. Adding those environments in **Settings → Environments** with
required reviewers is worth doing — it turns "push a tag" into "push a tag,
then approve", which is a useful pause before an irreversible upload.

## Rehearse (optional)

**Actions → Release → Run workflow**, target `testpypi`. This builds, checks and
smoke-tests exactly as a real release does, then uploads to TestPyPI, where a
version number can be spent freely.

## Release

1. Update `__version__` in `cuplan/__init__.py`.
2. Add the version's section to [`CHANGELOG.md`](CHANGELOG.md) — the workflow
   copies it verbatim into the GitHub release notes.
3. **First release only** — the README currently tells the truth about a
   package that is not on PyPI yet. Make it tell the truth about one that is:

   - Replace the placeholder badge

     ```
     [![PyPI](https://img.shields.io/badge/PyPI-not%20yet%20published-6d8298)](https://openplan-labs.github.io/cuda-planning/install/)
     ```

     with the live one

     ```
     [![PyPI](https://img.shields.io/pypi/v/cuplan?color=c2472c)](https://pypi.org/project/cuplan/)
     ```

   - Delete the "Not yet on PyPI — until then: `pip install git+…`" sentence
     from the install section, and the same caveat in `docs/install.md`.

4. Commit, then tag and push:

   ```sh
   git commit -am "Release 0.1.0"
   git tag v0.1.0
   git push origin main v0.1.0
   ```

The workflow then builds the sdist and wheel, fails if the tag disagrees with
`cuplan.__version__`, runs `twine check --strict`, verifies the wheel carries
the four `.cu` kernel sources, installs the built wheel on Python 3.10–3.13 and
solves an instance with it, publishes to PyPI, and finally creates the GitHub
release with the distributions attached.

## What the checks are protecting against

- **Tag/version drift** — a tag that disagrees with the package version ships an
  artifact whose name lies about its contents; PyPI uploads cannot be replaced.
- **Missing kernels** — the CUDA sources are compiled at runtime by NVRTC. A
  wheel without them installs fine and fails on the user's first GPU call.
- **A broken CPU path** — `pip install cuplan` with no CuPy gives the
  NumPy backend. The smoke test runs that exact path, on a runner with no GPU,
  because it is what most first-time users get.

Version numbers follow [SemVer](https://semver.org/). While the API is at
`0.x`, minor versions may break it; the changelog says so when they do.
