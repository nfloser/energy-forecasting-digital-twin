# Modelling

## Forecast target

v1 predicts consumption one hour ahead. A longer horizon is not created by relabeling repeated one-step predictions; six-hour and 24-hour forecasting require explicit multi-step evaluation and are future work.

## Features

The current feature set contains:

- hour of day
- day of week
- weekend flag
- month
- temperature
- relative humidity
- heating-degree style value relative to 18 °C
- cooling-degree style value relative to 22 °C
- 1-hour demand lag
- 24-hour demand lag
- 168-hour demand lag
- shifted 6-hour rolling mean
- shifted 6-hour rolling standard deviation

The temperature thresholds are modelling features, not building-physics claims. Their value must be judged empirically.

## Leakage prevention

All demand-derived predictors are shifted by at least one observation before rolling statistics are calculated. A dedicated test changes a future target to an extreme value and asserts that earlier feature rows remain byte-for-byte equivalent.

## Model hierarchy

### Persistence

The previous observed consumption is the reference baseline. On smooth, high-frequency demand series this can be a strong forecast, making it a useful minimum comparison.

### Seasonal naive

The second baseline uses the value 24 hourly steps earlier. It represents daily periodicity without learned parameters.

### Ridge regression

Ridge regression provides an interpretable regularized linear benchmark. It tests whether the engineered features add predictive information beyond the naive baselines.

### HistGradientBoostingRegressor

The tree-based model captures nonlinear interactions while remaining materially simpler than adding external boosting libraries or neural networks. A fixed random seed is persisted.

## Model selection

Both trained model families are evaluated using the same walk-forward folds. The final demo forecast selects the ML family with the lower evaluated MAE. Baselines remain visible regardless of which ML model is selected.
