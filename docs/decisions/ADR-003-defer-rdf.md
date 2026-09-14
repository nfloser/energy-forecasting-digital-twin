# ADR-003: Defer RDF until semantic queries justify it

**Status:** Accepted

## Context

RDF is useful for asset relationships and provenance traversal, but bulk numerical time-series analytics are not a good reason to introduce a triple store.

## Decision

v1 keeps explicit model/provenance metadata in JSON and does not introduce RDF solely for portfolio appearance.

## Consequences

The runtime is smaller and easier to reproduce. RDF/PROV-O should be introduced when the twin contains multiple buildings/meters/districts and users require semantic provenance or SPARQL relationship queries.
