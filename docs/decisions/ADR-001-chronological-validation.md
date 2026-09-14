# ADR-001: Use chronological validation instead of random splitting

**Status:** Accepted

## Context

Forecasting has a strict information order. Random train/test splitting can leak later regimes into training and overstate performance.

## Decision

All reported forecasting evaluation uses forward chronological splits. v1 implements an expanding-window protocol.

## Consequences

Metrics are more defensible but may be worse than random-split metrics. That is an intended consequence, not a defect.
