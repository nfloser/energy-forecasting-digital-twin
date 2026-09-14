# ADR-002: Keep numeric time series tabular in v1

**Status:** Accepted

## Context

The primary workload is vectorized hourly feature engineering and model training over numerical arrays.

## Decision

Persist the canonical demo series as a tabular CSV behind a repository boundary. Persist metrics/provenance as JSON and estimators as joblib.

## Consequences

The design stays inspectable and dependency-light. A production multi-building deployment should replace CSV with a relational/time-series store without changing the domain contracts.
