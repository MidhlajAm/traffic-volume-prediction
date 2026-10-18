"""End-to-end CSV recognition, model selection, and failure-preservation checks."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import create_app
from dataset_import import CSVError, inspect_csv, prepare_uploaded

ROOT = Path(__file__).resolve().parents[1]


def traffic_csv(rows=240, weather=False, locations=False):
    times = pd.date_range("2024-01-01", periods=rows, freq="h")
    frame = pd.DataFrame({"DateTime": times, "Vehicles": 150 + times.hour * 12 + times.dayofweek * 8})
    if weather:
        frame["temperature_c"] = 12 + np.sin(np.arange(rows) / 24)
        frame["rainfall"] = 0
        frame["weather"] = "Clear"
    if locations:
        first, second = frame.copy(), frame.copy()
        first["Junction"], second["Junction"] = "1", "2"
        second["Vehicles"] *= 3
        frame = pd.concat([first, second], ignore_index=True)
    return frame.to_csv(index=False, lineterminator="\n").encode()


@pytest.fixture
def application(tmp_path):
    app = create_app(dataset_root=tmp_path / "datasets")
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(application):
    return application.test_client()


def upload(client, content=None, filename="my-traffic.csv", **labels):
    return client.post("/api/datasets/inspect", data={"file": (io.BytesIO(traffic_csv() if content is None else content), filename), **labels})


def train_upload(client, content=None, **options):
    inspected = upload(client, content)
    assert inspected.status_code == 201, inspected.json
    result = client.post("/api/datasets/train", json={"id": inspected.json["id"], **options})
    assert result.status_code == 200, result.json
    return inspected.json["id"]


def test_minimal_csv_autodetects_and_changes_every_model_surface(application, client):
    original_hash = (ROOT / "artifacts/model.joblib").read_bytes()
    dataset_id = train_upload(client)
    summary = client.get("/api/summary").json
    assert summary["dataset"]["id"] == dataset_id
    assert summary["feature_columns"] == ["hour", "weekday", "month"]
    assert summary["records"] == 240
    assert summary["dataset"]["location"] == "Location not supplied"
    prediction = client.post("/api/predict", json={"date": "2024-02-01", "hour": 9, "dataset_id": dataset_id})
    assert prediction.status_code == 200
    assert prediction.json["dataset_id"] == dataset_id
    assert set(prediction.json["conditions"]) == {"date", "hour"}
    for path in ["/", "/predict", "/performance", "/about", "/datasets"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "my-traffic" in response.get_data(as_text=True)
    assert "I-94" not in client.get("/predict").get_data(as_text=True)
    evaluation = pd.read_csv(io.BytesIO(client.get("/download/evaluation.csv").data))
    assert len(evaluation) == 48
    assert pd.to_datetime(evaluation.date_time).min() > pd.Timestamp(summary["split"]["train_end"])
    assert (ROOT / "artifacts/model.joblib").read_bytes() == original_hash
    assert client.get(f"/download/dataset.csv?dataset_id={dataset_id}").data == traffic_csv()


def test_switch_back_and_selection_survives_restart(application, client):
    original = client.get("/api/summary").json["sample"]
    baseline_prediction = client.post("/api/predict", json=original).json["predicted_volume"]
    dataset_id = train_upload(client)
    restored = create_app(dataset_root=application.extensions["dataset_store"].root)
    assert restored.test_client().get("/api/health").json["dataset_id"] == dataset_id
    assert client.post("/api/datasets/activate", json={"id": "builtin"}).status_code == 200
    assert client.post("/api/predict", json=original).json["predicted_volume"] == baseline_prediction
    restored = create_app(dataset_root=application.extensions["dataset_store"].root)
    assert restored.extensions["dataset_store"].active.id == "builtin"
    assert client.post("/api/datasets/activate", json={"id": dataset_id}).status_code == 200
    assert len(client.get("/api/datasets").json["datasets"]) == 2


def test_optional_weather_and_celsius_are_used(client):
    train_upload(client, traffic_csv(weather=True))
    summary = client.get("/api/summary").json
    assert set(summary["feature_columns"]) == {"temperature", "rainfall", "weather", "hour", "weekday", "month"}
    assert client.post("/api/predict", json=summary["sample"]).status_code == 200
    missing = {**summary["sample"]}
    del missing["temperature"]
    assert "temperature" in client.post("/api/predict", json=missing).json["errors"]
    assert -60 <= summary["sample"]["temperature"] <= 60


def test_unknown_temperature_unit_requires_confirmation_and_preserves_active(client):
    content = traffic_csv(weather=True).replace(b"temperature_c", b"temperature")
    inspected = upload(client, content).json
    assert inspected["temperature_unit"] is None
    result = client.post("/api/datasets/train", json={"id": inspected["id"]})
    assert result.status_code == 400 and "Celsius or Kelvin" in result.json["message"]
    assert client.get("/api/health").json["dataset_id"] == "builtin"
    assert client.post("/api/datasets/train", json={"id": inspected["id"], "temperature_unit": "C"}).status_code == 200


def test_uci_columns_and_kelvin_conversion_are_detected(client):
    raw = pd.read_csv(ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv.gz", keep_default_na=False).head(500)
    content = raw.to_csv(index=False).encode()
    inspected = upload(client, content).json
    assert inspected["temperature_unit"] == "K"
    assert inspected["mapping"]["traffic_volume"] == "traffic_volume"
    assert client.post("/api/datasets/train", json={"id": inspected["id"]}).status_code == 200
    assert client.get("/api/summary").json["weather_categories"]


def test_locations_are_selected_before_deduplication(client):
    inspected = upload(client, traffic_csv(locations=True)).json
    assert inspected["unique_hours"] == 480 and inspected["duplicates"] == 0
    assert len(inspected["locations"]) == 2
    assert client.post("/api/datasets/train", json={"id": inspected["id"]}).status_code == 400
    assert client.post("/api/datasets/train", json={"id": inspected["id"], "location": "2"}).status_code == 200
    summary = client.get("/api/summary").json
    assert summary["raw_records"] == summary["records"] == 240
    assert summary["dataset"]["location"] == "Junction: 2"
    assert summary["average_volume"] == pytest.approx(pd.read_csv(io.BytesIO(traffic_csv())).Vehicles.mean() * 3)


def test_separate_date_and_hour_semicolon_csv_and_duplicates():
    frame = pd.read_csv(io.BytesIO(traffic_csv()))
    stamp = pd.to_datetime(frame.pop("DateTime"))
    frame["Date"], frame["Hour"] = stamp.dt.strftime("%Y-%m-%d"), stamp.dt.hour
    frame = pd.concat([frame, frame.iloc[:2]], ignore_index=True)
    info, raw = inspect_csv(frame.to_csv(sep=";", index=False).encode())
    assert info["duplicates"] == 2
    prepared, original_rows, _location = prepare_uploaded(raw, info)
    assert len(prepared) == 240 and original_rows == 242
    assert prepared.date_time.is_unique and prepared.date_time.is_monotonic_increasing


@pytest.mark.parametrize("content,phrase", [
    (b"", "non-empty"),
    (b"name,value\na,1\nb,2\n", "traffic-volume"),
    (b"DateTime,Vehicles\ninvalid,12\n", "time"),
    (traffic_csv(rows=50), "100 distinct"),
    (traffic_csv().replace(b",150\n", b",-1\n", 1), "nonnegative"),
    (traffic_csv().replace(b",150\n", b",inf\n", 1), "nonnegative"),
    (traffic_csv().replace(b"00:00:00", b"00:15:00", 1), "whole-hour"),
    (traffic_csv().replace(b"2024-01-01", b"01/01/2024", 1), "year-first"),
    (b"DateTime,Date Time,Vehicles\na,b,1\n", "distinct"),
    (b"DateTime,Vehicles\n2024-01-01 00:00:00Z,1\n", "timezone"),
], ids=["empty", "not-traffic", "bad-time", "too-small", "negative-target", "infinite-target", "sub-hour", "ambiguous-date", "duplicate-header", "timezone"])
def test_bad_csv_rejected_without_replacing_original(client, content, phrase):
    result = upload(client, content)
    assert result.status_code == 400
    assert phrase.lower() in result.json["message"].lower()
    assert client.get("/api/health").json["dataset_id"] == "builtin"
    assert len(client.get("/api/datasets").json["datasets"]) == 1


def test_missing_file_wrong_extension_and_size_limit(client):
    assert client.post("/api/datasets/inspect").status_code == 400
    assert upload(client, filename="traffic.xlsx").status_code == 400
    assert upload(client, b"x" * (10 * 1024 * 1024 + 1)).status_code == 400
    assert client.post("/api/datasets/inspect", data=b"x" * (11 * 1024 * 1024)).status_code == 413


def test_ambiguous_columns_and_nonvarying_target(client):
    frame = pd.read_csv(io.BytesIO(traffic_csv()))
    frame["traffic_volume"] = frame.Vehicles
    assert "More than one" in upload(client, frame.to_csv(index=False).encode()).json["message"]
    del frame["traffic_volume"]
    frame["Vehicles"] = 10
    inspected = upload(client, frame.to_csv(index=False).encode()).json
    failed = client.post("/api/datasets/train", json={"id": inspected["id"]})
    assert failed.status_code == 400 and "must vary" in failed.json["message"]
    assert client.get("/api/health").json["dataset_id"] == "builtin"


def test_failed_fit_and_busy_training_preserve_predictions(application, client, monkeypatch):
    sample = application.extensions["dataset_store"].active.summary["sample"]
    before = client.post("/api/predict", json=sample).json
    inspected = upload(client).json
    def fail_fit(*_args, **_kwargs):
        raise ValueError("Synthetic training failure")
    monkeypatch.setattr("dataset_store.fit_dataset", fail_fit)
    assert client.post("/api/datasets/train", json={"id": inspected["id"]}).status_code == 400
    assert client.post("/api/predict", json=sample).json == before
    store = application.extensions["dataset_store"]
    with store.operation_lock:
        assert client.post("/api/datasets/activate", json={"id": "builtin"}).status_code == 409
        assert client.post("/api/predict", json=sample).status_code == 200


def test_untrained_missing_and_path_like_ids_are_rejected(client):
    inspected = upload(client).json
    for dataset_id in [inspected["id"], "../artifacts", "f" * 32, None, {"id": "builtin"}]:
        assert client.post("/api/datasets/activate", json={"id": dataset_id}).status_code == 400
    assert client.get("/api/health").json["dataset_id"] == "builtin"


def test_old_prediction_page_cannot_silently_use_another_model(client):
    sample = client.get("/api/summary").json["sample"]
    train_upload(client)
    response = client.post("/api/predict", json={**sample, "dataset_id": "builtin"})
    assert response.status_code == 409 and "Reload" in response.json["message"]


def test_sample_download_can_be_uploaded_and_trained(client):
    response = client.get("/download/sample-traffic.csv")
    assert response.status_code == 200
    inspected = upload(client, response.data).json
    assert inspected["rows"] == 168 and inspected["temperature_unit"] == "C"
    assert client.post("/api/datasets/train", json={"id": inspected["id"]}).status_code == 200


def test_sparse_hours_and_zero_baseline_mae_are_json_safe(client):
    # Test-period counts equal the training mean, making baseline MAE exactly zero.
    times = pd.date_range("2024-01-01", periods=150, freq="D")
    counts = [0, 20] * 60 + [10] * 30
    frame = pd.DataFrame({"DateTime": times.strftime("%Y-%m-%d 08:00:00"), "Vehicles": counts})
    train_upload(client, frame.to_csv(index=False).encode())
    summary = client.get("/api/summary").json
    assert summary["mae_improvement"] is None
    assert summary["charts"]["hourly"]["all"][0] is None
    assert client.get("/performance").status_code == 200
    assert b"Baseline MAE is zero" in client.get("/performance").data
    assert b"NaN" not in client.get("/api/summary").data


def test_all_missing_optional_numeric_column_is_excluded(client):
    frame = pd.read_csv(io.BytesIO(traffic_csv()))
    frame["temperature_c"] = ""
    train_upload(client, frame.to_csv(index=False).encode())
    assert "temperature" not in client.get("/api/summary").json["feature_columns"]


def test_labels_are_escaped_and_filename_cannot_escape_store(application, client):
    inspected = upload(client, filename="../../test.csv", name="<script>alert(1)</script>", location="Test road").json
    assert inspected["dataset"]["filename"] == "test.csv"
    assert client.post("/api/datasets/train", json={"id": inspected["id"]}).status_code == 200
    page = client.get("/datasets").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert (application.extensions["dataset_store"].root / inspected["id"] / "source.csv").exists()
