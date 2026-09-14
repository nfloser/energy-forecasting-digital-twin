# Limitations

1. **Representativeness** — the UCI energy source describes one household in Sceaux. Results cannot be generalized to offices, industrial sites, districts or national grids without additional evidence.
2. **Historical period** — the source energy measurements end in 2010. The project demonstrates methodology and software architecture, not present-day load behavior.
3. **Weather resolution** — Open-Meteo historical weather is gridded/reanalysis data and may differ from conditions at the exact building.
4. **Forecast horizon** — v1 evaluates only one hour ahead. Multi-step forecasting can accumulate error and requires separate evaluation.
5. **Distribution shift** — tariff changes, occupancy changes, retrofits, appliance replacement and extreme weather can invalidate learned relationships.
6. **No calibrated intervals** — v1 does not estimate prediction intervals. The dashboard intentionally avoids uncertainty graphics that would imply calibration that was never measured.
7. **Storage scale** — canonical CSV storage is appropriate for the portfolio/research workload but not for high-volume multi-building production ingestion.
8. **Operational safety** — the system is not validated for grid dispatch, safety-critical control, billing, or financial decision automation.
9. **Model scope** — Ridge and histogram gradient boosting are deliberately modest. Neural networks are not included because simpler families must establish the value of the feature/data pipeline first.
10. **External services** — reproduction of the real-data run depends on the continued availability and terms of the UCI and Open-Meteo services.
