"""Acceptance checks against the real, saved model and untouched UCI data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app import create_app
from traffic_model import FEATURES, NUMERIC, chronological_split, prepare_data, predict_volume, validate_input

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    application = create_app(dataset_root=tmp_path_factory.mktemp('dataset-store'))
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def example(app):
    return dict(app.extensions["dataset_store"].active.summary["sample"])


@pytest.fixture(scope="module")
def prepared():
    source = ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv.gz"
    return prepare_data(pd.read_csv(source, keep_default_na=False))


@pytest.mark.parametrize("path,expected", [
    ("/", "Traffic, in perspective."), ("/predict", "Prediction conditions"),
    ("/performance", "Behind the predictions."), ("/about", "From observations to understanding."),
    ("/static/style.css", "--blue"), ("/static/app.js", "api/predict"),
    ("/static/vendor/chart.umd.min.js", "Chart.js"),
])
def test_pages_and_local_assets(client, path, expected):
    response = client.get(path)
    assert response.status_code == 200
    assert expected in response.get_data(as_text=True)


def test_ready_and_summary(client):
    assert client.get("/api/health").json["status"] == "ready"
    summary = client.get("/api/summary").json
    assert summary["raw_records"] == 48204
    assert summary["records"] == 40575
    assert len(summary["charts"]["hourly"]["all"]) == 24
    assert len(summary["charts"]["weekday"]) == 7


def test_prediction_matches_saved_pipeline_and_repeated_calls(app, client, example):
    _conditions, features = validate_input(example, app.extensions["dataset_store"].active.summary["weather_categories"])
    expected = predict_volume(app.extensions["dataset_store"].active.model, features)[0]
    first = client.post("/api/predict", json=example)
    second = client.post("/api/predict", json=example)
    assert first.status_code == 200
    assert first.json["predicted_volume"] == pytest.approx(expected, abs=1e-10)
    assert second.json == first.json
    assert first.json["predicted_volume"] >= 0
    assert first.json["conditions"]["date"] == example["date"]
    assert first.json["unit"] == "vehicles/hour"


@pytest.mark.parametrize("field", ["date", "hour", "temperature", "rainfall", "snowfall", "cloud_cover", "weather", "holiday"])
def test_missing_fields_rejected(client, example, field):
    del example[field]
    response = client.post("/api/predict", json=example)
    assert response.status_code == 400
    assert field in response.json["errors"]


@pytest.mark.parametrize("field,value", [
    ("date", "2026-02-30"), ("date", "16/10/2026"), ("date", 123),
    ("hour", -1), ("hour", 24), ("hour", 8.5), ("hour", True),
    ("rainfall", -0.1), ("snowfall", -1), ("rainfall", "NaN"),
    ("temperature", "Infinity"), ("temperature", -61), ("temperature", 61),
    ("temperature", ""), ("rainfall", None), ("cloud_cover", -1),
    ("cloud_cover", 101), ("weather", "Unknown"), ("holiday", "false"),
])
def test_invalid_inputs_rejected(client, example, field, value):
    example[field] = value
    response = client.post("/api/predict", json=example)
    assert response.status_code == 400
    assert field in response.json["errors"]


@pytest.mark.parametrize("body", [None, [], "hello", 42])
def test_non_object_json_rejected(client, body):
    response = client.post("/api/predict", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400


def test_malformed_and_oversized_requests(client):
    assert client.post("/api/predict", data="{", content_type="application/json").status_code == 400
    assert client.post("/api/predict", data="hello").status_code == 400
    response = client.post("/api/predict", data="x" * 17000, content_type="application/json")
    assert response.status_code == 413


def test_boundary_inputs_and_future_date(client, example):
    example.update(date="2030-02-28", hour=0, temperature=-60, rainfall=0, snowfall=0, cloud_cover=100, holiday=True)
    response = client.post("/api/predict", json=example)
    assert response.status_code == 200
    assert np.isfinite(response.json["predicted_volume"])
    example.update(hour=23, temperature=60, cloud_cover=0)
    assert client.post("/api/predict", json=example).status_code == 200


def test_split_has_no_time_leakage(app, prepared):
    training, evaluation = chronological_split(prepared)
    split = app.extensions["dataset_store"].active.summary["split"]
    assert prepared.date_time.is_unique and prepared.date_time.is_monotonic_increasing
    assert training.date_time.max() < evaluation.date_time.min()
    assert not set(training.date_time) & set(evaluation.date_time)
    assert len(training) == split["train_count"] == 32460
    assert len(evaluation) == split["test_count"] == 8115
    assert training.date_time.iloc[-1].isoformat() == split["train_end"]
    assert evaluation.date_time.iloc[0].isoformat() == split["test_start"]


def test_preprocessing_is_fitted_on_training_only(app, prepared):
    training, _evaluation = chronological_split(prepared)
    preprocessing = app.extensions["dataset_store"].active.model.named_steps["preprocess"]
    imputer = preprocessing.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, training[NUMERIC].median().values)
    scaler = preprocessing.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, training[NUMERIC].fillna(training[NUMERIC].median()).mean().values)
    assert int(training.holiday.sum()) > 0  # Preserve the literal source value 'None'.


def test_form_features_match_the_original_held_out_observation(app, example, prepared):
    _conditions, features = validate_input(example, app.extensions["dataset_store"].active.summary["weather_categories"])
    timestamp = pd.Timestamp(example["date"]) + pd.Timedelta(hours=example["hour"])
    source = prepared.loc[prepared.date_time == timestamp].iloc[0]
    for key in FEATURES:
        assert features.iloc[0][key] == source[key]


def test_saved_metrics_and_csv_reproduce_exactly(app, client, prepared):
    training, evaluation = chronological_split(prepared)
    predictions = predict_volume(app.extensions["dataset_store"].active.model, evaluation)
    summary = app.extensions["dataset_store"].active.summary
    saved = pd.read_csv(ROOT / "artifacts/evaluation.csv")
    np.testing.assert_allclose(saved.predicted_volume, predictions, rtol=1e-12)
    np.testing.assert_allclose(saved.baseline_volume, training.traffic_volume.mean())
    actual = evaluation.traffic_volume
    assert summary["metrics"]["mae"] == pytest.approx(mean_absolute_error(actual, predictions))
    assert summary["metrics"]["rmse"] == pytest.approx(np.sqrt(mean_squared_error(actual, predictions)))
    assert summary["metrics"]["r2"] == pytest.approx(r2_score(actual, predictions))
    assert summary["metrics"]["mae"] < summary["baseline_metrics"]["mae"]
    response = client.get("/download/evaluation.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert len(saved) == len(evaluation)


def test_lower_bound_keeps_fractional_precision():
    class NegativeAndFractionalModel:
        def predict(self, _features):
            return np.array([-120.5, 123.456])
    features = pd.DataFrame([{feature: 0 for feature in FEATURES}] * 2)
    np.testing.assert_array_equal(predict_volume(NegativeAndFractionalModel(), features), [0.0, 123.456])


def test_missing_artifacts_gives_setup_guidance(tmp_path):
    with pytest.raises(RuntimeError, match="python train.py"):
        create_app(tmp_path)


def test_shared_training_preserves_original_model_metrics(app, prepared, tmp_path):
    from train import fit_dataset
    original = app.extensions["dataset_store"].active.summary
    retrained = fit_dataset(prepared, tmp_path, original["source"], original["raw_records"])
    for metric in ["mae", "rmse", "r2"]:
        assert retrained["metrics"][metric] == pytest.approx(original["metrics"][metric], abs=1e-9)
