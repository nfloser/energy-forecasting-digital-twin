# ADR-004: Establish naive baselines before advanced ML

**Status:** Accepted

## Context

A complex model can produce plausible-looking forecasts without adding value over persistence or daily seasonality.

## Decision

Persistence and 24-hour seasonal naive are mandatory evaluation references. Ridge precedes nonlinear tree modelling; neural networks are excluded from v1.

## Consequences

The project optimizes for measured incremental value and interpretability rather than model novelty.
