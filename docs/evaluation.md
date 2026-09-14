# Evaluation

## Why random splits are invalid here

Random splitting allows later temporal regimes to influence training while earlier points appear in the test set. That violates the causal information order of forecasting and can produce deceptively optimistic metrics.

## Expanding-window protocol

After leakage-safe feature construction, evaluation begins with a minimum chronological training prefix. The next contiguous block is evaluated, the training prefix expands, and the process repeats. The default real-data run uses seven-day test blocks after at least 30 days of complete modelling rows.

Within each test block the one-step persistence and seasonal baselines use only observations that precede each target. Trained estimators are fitted on the fold's training prefix and predict the test feature rows.

## Metrics

- **MAE**: primary scale-dependent error in kWh
- **RMSE**: penalizes larger misses more strongly
- **MAPE**: reported only for non-zero target values
- **R²**: supplementary context, never the primary forecasting claim

Relative improvement is computed against persistence:

`(persistence_MAE - model_MAE) / persistence_MAE * 100`

A negative result is shown as worse performance; it is not clipped or reframed.

## Deterministic CI versus full evaluation

CI uses a deterministic synthetic fixture solely to verify contracts, leakage behavior, evaluation plumbing and API integration. It is not evidence of real predictive skill.

The `Full real-data evaluation` workflow downloads UCI and Open-Meteo data and writes the real metrics as workflow artefacts. Full evaluation is kept separate from per-commit CI because external downloads and model fitting are slower and less reliable than deterministic fixtures.

## Interpreting results

No model is considered successful merely because training completed. The relevant question is whether its out-of-sample MAE/RMSE improves on the naive references across forward folds. If both ML models lose to persistence, the correct project conclusion is that the tested features/models did not add sufficient predictive value for that evaluation period.
