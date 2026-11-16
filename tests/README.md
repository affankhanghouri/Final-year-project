# Automated tests

Run from the repository root after editable installation:

```sh
venv/bin/python -B -m unittest discover -s tests -v
```

The seven existing test modules cover sessions, camera/buffer behavior, detector
normalization, inference workers, stability, guidance, and lifecycle cleanup.
`test_lifecycle_cleanup.py` imports the live preview from `scripts/` to test its
shutdown behavior with mocks. Production imports use the installed `homesense`
package; there are no source-path modifications in the tests.

Future cross-component tests can live in `tests/integration/`. Add an `__init__.py`
in that directory when introducing those tests so the same unittest discovery
command recurses into it. No placeholder test suite is needed now.

Physical camera testing remains in `scripts/` and is separate from automated tests.
