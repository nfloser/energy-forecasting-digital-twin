# Architecture

## Goal

The system creates a traceable digital representation of historical building energy demand and its short-term forecast state. The architecture is intentionally compact: data engineering, feature generation and model evaluation live in one Python package; FastAPI exposes persisted research artefacts; React renders them.

## Components

### Ingestion boundary

`energy_twin.ingestion` owns source-specific schemas. UCI's semicolon-delimited minute data and Open-Meteo JSON are converted immediately into canonical columns. Raw provider formats do not propagate into modelling code.

### Canonical time series

The canonical table is hourly UTC data with `timestamp`, `consumption_kwh`, `temperature_c`, and `humidity_pct`. Duplicate handling, missingness, temporal regularity and timezone awareness are reported explicitly. No blanket forward fill is performed.

### Feature engineering

`energy_twin.features` creates calendar features, weather variables, degree-style variables, demand lags and shifted rolling statistics. Demand-derived features use only values preceding the target timestamp.

### Forecasting and evaluation

Baselines and trained models share the same chronological evaluation horizon. An expanding training window is used. Evaluation output is persisted as JSON rather than being trapped in notebook state.

### Persistence

High-frequency numeric observations are persisted as a canonical CSV in v1 because the demo dataset is small enough for a single-process research runtime. Model artefacts use joblib and metadata/metrics/provenance use versionable JSON. This is deliberately not an enterprise database design; a production-scale deployment would replace the CSV boundary with a relational/time-series store without changing the domain/evaluation contracts.

### API and UI

FastAPI is the only data interface used by the React client. The frontend has no credentials or provider-specific code. Nginx proxies `/api/*` to the backend in the container runtime.

## Deployment

Docker Compose runs two services: `backend` and `frontend`. The backend mounts `./data` so trained artefacts survive container replacement. No microservices are introduced for ingestion/training because the current workload does not justify them.

## Semantic layer

RDF is deferred in v1. Numeric measurements are a poor fit for triple storage in this project, and the current metadata graph is small enough that JSON provenance is clearer. ADR-003 defines the trigger for adding RDF/PROV-O later: cross-asset metadata relationships or SPARQL provenance questions that cannot be served cleanly by the existing model registry.
