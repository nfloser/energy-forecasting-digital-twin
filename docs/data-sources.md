# Data sources

## Energy: UCI Individual Household Electric Power Consumption

- Source: UCI Machine Learning Repository
- DOI: `10.24432/C58K54`
- Location: Sceaux, France
- Coverage: December 2006 to November 2010
- Native resolution: one minute
- Licence: CC BY 4.0
- Relevant field: `Global_active_power` in kilowatts

The loader converts each observed minute from kW to kWh by dividing by 60 and sums within the UTC hour. It records minute coverage and discards hours below the configured 80% coverage threshold rather than pretending the missing minutes were observed.

The source includes missing measurements. Missingness is therefore a real characteristic of the dataset, not an ingestion bug.

## Weather: Open-Meteo Historical Weather API

- Location: coordinates representing Sceaux, France (`48.778`, `2.290`)
- Variables: 2 m temperature and 2 m relative humidity
- Resolution requested: hourly
- Data licence: CC BY 4.0 with attribution
- API access: subject to Open-Meteo terms and limits

Open-Meteo historical weather is gridded/reanalysis-oriented data. It must not be described as a sensor mounted on the household. Spatial resolution and reanalysis methodology can smooth local microclimate effects.

## Time alignment

Provider-local timestamps are interpreted in `Europe/Paris` and normalized to UTC at the ingestion boundary. DST transitions are handled explicitly. Energy and weather are joined one-to-one on the normalized hourly timestamp.

## Missing data policy

The project does not silently forward-fill all missing values. Energy hours with insufficient source coverage are excluded. Weather rows missing required variables are removed after the merge and the final canonical series is assessed with the data-quality reporter.

## Reproducibility

The final canonical frame is hashed using SHA-256 after deterministic column ordering. The hash is persisted in model metadata and provenance so a model version can be tied to the exact processed dataset used for training.
