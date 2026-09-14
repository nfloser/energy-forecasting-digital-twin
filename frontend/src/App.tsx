import { useEffect, useMemo, useState } from "react";
import { formatNumber, improvementLabel } from "./metrics";

type HistoryRow = {
  timestamp: string;
  consumption_kwh: number;
  temperature_c: number;
  humidity_pct: number;
};

type ModelMetric = {
  model: string;
  mae: number;
  rmse: number;
  mape: number | null;
  r2: number | null;
  relative_improvement_vs_persistence_pct: number;
};

type Evaluation = {
  validation: string;
  folds: number;
  observations_evaluated: number;
  models: ModelMetric[];
};

type Forecast = {
  mode: string;
  methodological_note: string;
  target_timestamp: string;
  selected_model: string;
  model_id: string;
  predicted_consumption_kwh: number;
  actual_consumption_kwh: number;
};

type Provenance = {
  dataset_hash_sha256: string;
  training_period: { start: string; end: string };
  feature_set: string[];
  selected_model: string;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

function LineChart({ history, forecast }: { history: HistoryRow[]; forecast: Forecast | null }) {
  const values = history.map((row) => row.consumption_kwh);
  const all = forecast ? [...values, forecast.predicted_consumption_kwh] : values;
  if (all.length < 2) return <div className="empty">No historical series available.</div>;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const scaleY = (value: number) => 148 - ((value - min) / Math.max(max - min, 0.001)) * 120;
  const points = values
    .map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 620},${scaleY(value)}`)
    .join(" ");
  return (
    <svg className="chart" viewBox="0 0 620 170" role="img" aria-label="Historical consumption chart">
      <line x1="0" y1="148" x2="620" y2="148" className="axis" />
      <polyline points={points} className="actualLine" />
      {forecast && (
        <circle cx="620" cy={scaleY(forecast.predicted_consumption_kwh)} r="5" className="forecastDot" />
      )}
    </svg>
  );
}

export default function App() {
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [provenance, setProvenance] = useState<Provenance | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getJson<HistoryRow[]>("/history?limit=168"),
      getJson<Evaluation>("/metrics"),
      getJson<Forecast>("/forecast"),
      getJson<Provenance>("/provenance"),
    ])
      .then(([historyData, metricsData, forecastData, provenanceData]) => {
        setHistory(historyData);
        setEvaluation(metricsData);
        setForecast(forecastData);
        setProvenance(provenanceData);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const latest = history.at(-1);
  const selectedMetric = useMemo(
    () => evaluation?.models.find((model) => model.model === forecast?.selected_model),
    [evaluation, forecast],
  );

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">RESEARCH SOFTWARE · DIGITAL TWIN</p>
          <h1>Energy demand, measured against reality.</h1>
          <p className="lede">
            Short-term forecasting with real energy and weather data, leakage-safe features, chronological
            validation and explicit naive baselines.
          </p>
        </div>
        <div className="status"><span /> {error ? "Artefacts unavailable" : "Research run loaded"}</div>
      </header>

      {error && <section className="notice">Run the documented demo command first. API detail: {error}</section>}

      <section className="metricGrid">
        <article className="metric"><span>Current demand</span><strong>{formatNumber(latest?.consumption_kwh)} kWh</strong><small>{latest?.timestamp ?? "—"}</small></article>
        <article className="metric"><span>1h forecast</span><strong>{formatNumber(forecast?.predicted_consumption_kwh)} kWh</strong><small>{forecast?.selected_model ?? "—"}</small></article>
        <article className="metric"><span>MAE</span><strong>{formatNumber(selectedMetric?.mae)} kWh</strong><small>walk-forward evaluation</small></article>
        <article className="metric"><span>RMSE</span><strong>{formatNumber(selectedMetric?.rmse)} kWh</strong><small>{evaluation?.folds ?? 0} folds</small></article>
      </section>

      <section className="panel wide">
        <div className="panelHeader"><div><p className="eyebrow">ACTUAL + FORECAST</p><h2>Last 168 hours</h2></div><div className="legend"><i className="actualLegend" /> actual <i className="forecastLegend" /> forecast</div></div>
        <LineChart history={history} forecast={forecast} />
        {forecast && <p className="methodNote">{forecast.methodological_note}</p>}
      </section>

      <section className="twoCol">
        <article className="panel">
          <p className="eyebrow">MODEL EVALUATION</p><h2>Baseline comparison</h2>
          <div className="modelTable">
            <div className="tableHead"><span>Model</span><span>MAE</span><span>RMSE</span><span>vs persistence</span></div>
            {evaluation?.models.map((model) => (
              <div className="tableRow" key={model.model}>
                <span>{model.model}</span><span>{formatNumber(model.mae)}</span><span>{formatNumber(model.rmse)}</span>
                <span className={model.relative_improvement_vs_persistence_pct < 0 ? "negative" : "positive"}>
                  {model.model === "persistence" ? "reference" : improvementLabel(model.relative_improvement_vs_persistence_pct)}
                </span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <p className="eyebrow">TWIN STATE</p><h2>Context & provenance</h2>
          <dl className="facts">
            <div><dt>Temperature</dt><dd>{formatNumber(latest?.temperature_c, 1)} °C</dd></div>
            <div><dt>Humidity</dt><dd>{formatNumber(latest?.humidity_pct, 0)}%</dd></div>
            <div><dt>Model version</dt><dd>{forecast?.model_id ?? "—"}</dd></div>
            <div><dt>Training period</dt><dd>{provenance ? `${provenance.training_period.start.slice(0, 10)} → ${provenance.training_period.end.slice(0, 10)}` : "—"}</dd></div>
            <div><dt>Dataset hash</dt><dd className="mono">{provenance?.dataset_hash_sha256.slice(0, 16) ?? "—"}…</dd></div>
            <div><dt>Features</dt><dd>{provenance?.feature_set.length ?? 0}</dd></div>
          </dl>
        </article>
      </section>
    </main>
  );
}
