"""Download UCI source once, train chronologically, and save reproducible artifacts."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from traffic_model import CATEGORICAL, FEATURES, NUMERIC, chronological_split, prepare_data, predict_volume

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"


def download_source():
    directory = ROOT / "data" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "metro-interstate-traffic-volume.zip"
    if not archive.exists():
        print("Downloading original UCI archive...", flush=True)
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Flowcast-Student-Project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        # Validate before preserving it. Never overwrite the original on retraining.
        with zipfile.ZipFile(io.BytesIO(content)) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("The dataset archive is damaged; retry the download.")
        archive.write_bytes(content)
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if name.endswith(".csv.gz"))
        csv_file = directory / "Metro_Interstate_Traffic_Volume.csv.gz"
        if not csv_file.exists():
            csv_file.write_bytes(zipped.read(member))
    return csv_file, hashlib.sha256(archive.read_bytes()).hexdigest()


def scores(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def hourly_profile(frame):
    return [None if pd.isna(value) else round(float(value), 2)
            for value in frame.groupby("hour").traffic_volume.mean().reindex(range(24))]


def fit_dataset(frame, artifacts, source, raw_records, feature_columns=None):
    """Shared fitting path for the built-in source and validated uploaded CSVs."""
    feature_columns = list(FEATURES if feature_columns is None else feature_columns)
    training, evaluation = chronological_split(frame)
    feature_columns = [key for key in feature_columns if key not in NUMERIC or training[key].notna().any()]
    numeric_columns = [key for key in NUMERIC if key in feature_columns]
    categorical_columns = [key for key in CATEGORICAL if key in feature_columns]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = [("categorical", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_columns)]
    if numeric_columns:
        transformers.insert(0, ("numeric", numeric, numeric_columns))
    transform = ColumnTransformer(transformers)
    model = Pipeline([("preprocess", transform), ("regression", LinearRegression())])
    model.fit(training[feature_columns], training.traffic_volume)
    predicted = predict_volume(model, evaluation)
    baseline = np.full(len(evaluation), training.traffic_volume.mean())
    metrics, baseline_metrics = scores(evaluation.traffic_volume, predicted), scores(evaluation.traffic_volume, baseline)
    weather = sorted(training.weather.unique().tolist()) if "weather" in feature_columns else []
    examples = evaluation[evaluation.weather.isin(weather)] if weather else evaluation
    if examples.empty:
        examples = training
    valid = examples[numeric_columns].notna().all(axis=1)
    if valid.any():
        examples = examples[valid]
    commute_examples = examples[(examples.weekday < 5) & (examples.hour == 8)]
    if weather:
        commute_examples = commute_examples[commute_examples.weather == "Clear"]
    # Choose an everyday scenario chronologically, without selecting by model error.
    candidate = commute_examples.iloc[0] if len(commute_examples) else examples.iloc[len(examples) // 2]
    sample = {"date": candidate.date_time.strftime("%Y-%m-%d"), "hour": int(candidate.hour)}
    for key in numeric_columns:
        sample[key] = float(candidate[key] if pd.notna(candidate[key]) else training[key].median())
    if weather:
        sample["weather"] = candidate.weather
    if "holiday" in feature_columns:
        sample["holiday"] = bool(candidate.holiday)
    scatter_indices = np.linspace(0, len(evaluation) - 1, min(450, len(evaluation)), dtype=int)
    timeline = evaluation.iloc[-72:]
    summary = {
        "source": source, "feature_columns": feature_columns,
        "raw_records": raw_records, "records": len(frame), "duplicates_removed": raw_records - len(frame),
        "start": frame.date_time.iloc[0].isoformat(), "end": frame.date_time.iloc[-1].isoformat(),
        "average_volume": float(frame.traffic_volume.mean()), "peak_hour": int(frame.groupby("hour").traffic_volume.mean().idxmax()),
        "temperature_anomalies": int(frame.temperature.isna().sum()) if "temperature" in feature_columns else 0,
        "split": {"train_count": len(training), "test_count": len(evaluation),
                  "train_start": training.date_time.iloc[0].isoformat(), "train_end": training.date_time.iloc[-1].isoformat(),
                  "test_start": evaluation.date_time.iloc[0].isoformat(), "test_end": evaluation.date_time.iloc[-1].isoformat()},
        "metrics": metrics, "baseline_metrics": baseline_metrics,
        "mae_improvement": 100 * (1 - metrics["mae"] / baseline_metrics["mae"]) if baseline_metrics["mae"] else None,
        "weather_categories": weather, "sample": sample, "sample_actual": float(candidate.traffic_volume),
        "sample_is_test": bool(candidate.date_time >= evaluation.date_time.iloc[0]),
        "sample_imputed": bool(candidate[numeric_columns].isna().any()) if numeric_columns else False,
        "charts": {"hourly": {"all": hourly_profile(frame),
                              "weekday": hourly_profile(frame[frame.weekday < 5]),
                              "weekend": hourly_profile(frame[frame.weekday >= 5])},
                   "weekday": [None if pd.isna(value) else round(float(value), 2) for value in frame.groupby("weekday").traffic_volume.mean().reindex(range(7))],
                   "scatter": [{"x": float(evaluation.traffic_volume.iloc[i]), "y": float(predicted[i])} for i in scatter_indices],
                   "timeline": {"labels": timeline.date_time.dt.strftime("%b %d %H:%M").tolist(),
                                "actual": timeline.traffic_volume.tolist(), "predicted": predicted[-len(timeline):].tolist()}},
        "trained_at": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__,
    }
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "model.joblib")
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    output = evaluation[["date_time", "traffic_volume"]].copy()
    output["predicted_volume"] = predicted
    output["baseline_volume"] = baseline
    output.to_csv(artifacts / "evaluation.csv", index=False)
    return summary


def train():
    csv_file, checksum = download_source()
    raw = pd.read_csv(csv_file, keep_default_na=False)
    summary = fit_dataset(prepare_data(raw), ROOT / "artifacts", {
        "name": "UCI Metro Interstate Traffic Volume", "url": SOURCE_PAGE,
        "author": "John Hogue", "license": "CC BY 4.0", "archive_sha256": checksum,
    }, len(raw))
    print(json.dumps({"training_hours": summary["split"]["train_count"],
                      "evaluation_hours": summary["split"]["test_count"], "metrics": summary["metrics"],
                      "baseline": summary["baseline_metrics"], "sklearn_version": sklearn.__version__}, indent=2))
    return summary


if __name__ == "__main__":
    train()
