from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import uvicorn

from energy_twin.ingestion import (
    UCI_POWER_URL,
    fetch_historical_weather,
    merge_energy_weather,
    parse_uci_power,
)
from energy_twin.training import publish_training_run

SCEAUX_LATITUDE = 48.778
SCEAUX_LONGITUDE = 2.290

SOURCE_METADATA = {
    "energy": {
        "name": "UCI Individual Household Electric Power Consumption",
        "doi": "10.24432/C58K54",
        "license": "CC BY 4.0",
        "location": "Sceaux, France",
        "native_resolution": "1 minute",
    },
    "weather": {
        "name": "Open-Meteo Historical Weather API",
        "variables": ["temperature_2m", "relative_humidity_2m"],
        "license": "CC BY 4.0 data; API usage subject to Open-Meteo terms",
        "location": "Sceaux, France coordinates",
        "native_resolution": "hourly",
    },
}


def prepare_real_data(
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "household_power.zip"
        with urlopen(UCI_POWER_URL, timeout=120) as response, archive.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as bundle:
            member = next(
                name
                for name in bundle.namelist()
                if name.endswith("household_power_consumption.txt")
            )
            with bundle.open(member) as raw:
                import io

                text = io.TextIOWrapper(raw, encoding="utf-8")
                energy = parse_uci_power(text)

    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    energy = energy[(energy["timestamp"] >= start) & (energy["timestamp"] < end_exclusive)]
    if energy.empty:
        raise ValueError("requested date range produced no UCI energy observations")

    weather = fetch_historical_weather(
        latitude=SCEAUX_LATITUDE,
        longitude=SCEAUX_LONGITUDE,
        start_date=start_date,
        end_date=end_date,
        timezone="Europe/Paris",
    )
    merged = merge_energy_weather(energy, weather)
    merged = merged.dropna(subset=["temperature_c", "humidity_pct"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("energy/weather merge produced no complete rows")

    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "processed" / "hourly.csv", index=False)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Energy forecasting digital twin")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="download real data, evaluate models and publish artefacts")
    demo.add_argument("--data-dir", type=Path, default=Path("data"))
    demo.add_argument("--start-date", default="2009-01-01")
    demo.add_argument("--end-date", default="2009-03-31")

    prepare = sub.add_parser("prepare-data", help="download and merge the real demo data")
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--start-date", default="2009-01-01")
    prepare.add_argument("--end-date", default="2009-03-31")

    train = sub.add_parser("train", help="evaluate and train from an existing processed CSV")
    train.add_argument("--data-dir", type=Path, default=Path("data"))

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-data":
        prepare_real_data(args.data_dir, start_date=args.start_date, end_date=args.end_date)
        return
    if args.command == "train":
        frame = pd.read_csv(args.data_dir / "processed" / "hourly.csv", parse_dates=["timestamp"])
        publish_training_run(frame, args.data_dir, source_metadata=SOURCE_METADATA)
        return
    if args.command == "demo":
        frame = prepare_real_data(args.data_dir, start_date=args.start_date, end_date=args.end_date)
        publish_training_run(frame, args.data_dir, source_metadata=SOURCE_METADATA)
        return
    if args.command == "serve":
        uvicorn.run("energy_twin.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
