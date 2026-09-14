# Development

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Frontend:

```bash
cd frontend
npm install
npm test
npm run build
```

## TDD workflow

For behavioral changes:

1. add or modify a test that expresses the external behavior
2. run it and confirm the failure is meaningful
3. implement the smallest correct behavior
4. run the focused test, then the full suite
5. refactor only while coverage remains green

The repository history contains dedicated failing-test commits for the core forecasting contracts, ingestion/API contracts, and persisted training artefacts.

## Reproducing the real run

```bash
energy-twin demo --data-dir data --start-date 2009-01-01 --end-date 2009-03-31
```

Do not commit downloaded source data, processed data, trained joblib files, or locally generated metrics. They are runtime/research artefacts tied to the data-source retrieval time.

## CI

The main CI workflow verifies static Python quality, all Python tests, frontend tests/build, deterministic end-to-end API readiness and Docker image construction. Network-heavy full evaluation is a separate scheduled/manual workflow.
