"""Local Flask application. Run train.py before starting this server."""
import io
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from traffic_model import InputError, predict_volume, validate_input
from dataset_import import CSVError, MAX_BYTES
from dataset_store import DatasetBusy, DatasetStore

ROOT = Path(__file__).resolve().parent


def create_app(artifact_dir=None, dataset_root=None):
    artifacts = Path(artifact_dir) if artifact_dir else ROOT / "artifacts"
    if not all((artifacts / name).exists() for name in ["model.joblib", "summary.json"]):
        raise RuntimeError("Trained model missing. Run: python train.py, then python app.py")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 64 * 1024
    store = DatasetStore(artifacts, Path(dataset_root) if dataset_root else ROOT / "data" / "datasets")
    app.extensions["dataset_store"] = store

    @app.before_request
    def request_size_limit():
        if request.path != "/api/datasets/inspect":
            request.max_content_length = 16 * 1024

    @app.get("/")
    @app.get("/predict")
    @app.get("/performance")
    @app.get("/about")
    @app.get("/datasets")
    def page():
        selected = request.path.strip("/") or "dashboard"
        return render_template("index.html", page=selected, data=store.active.summary,
                               catalog=store.catalog() if selected == "datasets" else None)

    @app.get("/api/summary")
    def get_summary():
        return jsonify(store.active.summary)

    @app.get("/api/health")
    def health():
        return jsonify(status="ready", model="Linear Regression", dataset_id=store.active.id)

    @app.post("/api/predict")
    def predict():
        bundle = store.active
        summary = bundle.summary
        try:
            payload = request.get_json()
            if isinstance(payload, dict) and payload.get("dataset_id", bundle.id) != bundle.id:
                return jsonify(message="The active dataset changed. Reload this page before predicting.", errors={"form": "Reload to use the active dataset's inputs."}), 409
            conditions, features = validate_input(payload, summary["weather_categories"], summary["feature_columns"])
        except (BadRequest, UnsupportedMediaType):
            return jsonify(message="Send valid JSON with Content-Type: application/json.", errors={"form": "Invalid JSON request."}), 400
        except InputError as error:
            return jsonify(message=str(error), errors=error.errors), 400
        try:
            prediction = float(predict_volume(bundle.model, features)[0])
            if not math.isfinite(prediction):
                raise ValueError("Non-finite prediction")
        except (ValueError, OverflowError, FloatingPointError):
            return jsonify(message="Weather values are too large to produce an estimate. Use realistic hourly measurements.",
                           errors={"form": "These weather values exceed the model's numerical range."}), 400
        return jsonify(predicted_volume=prediction, unit="vehicles/hour", conditions=conditions, dataset_id=bundle.id)

    @app.get("/download/evaluation.csv")
    def download_evaluation():
        return send_file(store.active.directory / "evaluation.csv", as_attachment=True)

    @app.get("/api/datasets")
    def datasets():
        return jsonify(store.catalog())

    @app.post("/api/datasets/inspect")
    def inspect_dataset():
        file = request.files.get("file")
        if file is None or not file.filename:
            raise CSVError("Choose a traffic CSV file to inspect.")
        result = store.inspect(file.read(MAX_BYTES + 1), file.filename,
                               request.form.get("name", ""), request.form.get("location", ""))
        return jsonify(result), 201

    @app.post("/api/datasets/train")
    def train_dataset():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise CSVError("Send the inspected dataset ID as a JSON object.")
        try:
            bundle = store.train(payload.get("id"), payload.get("temperature_unit"), payload.get("location"))
        except CSVError:
            raise
        except (ValueError, TypeError) as error:
            raise CSVError("This dataset could not train a model. Check that the training period contains varying date/time features and usable numeric readings.") from error
        return jsonify(message="Dataset trained and activated.", dataset=bundle.summary["dataset"], metrics=bundle.summary["metrics"])

    @app.post("/api/datasets/activate")
    def activate_dataset():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise CSVError("Choose a saved dataset to activate.")
        bundle = store.activate(payload.get("id"))
        return jsonify(message="Active dataset updated.", dataset=bundle.summary["dataset"])

    @app.get("/download/dataset.csv")
    def download_dataset():
        dataset_id = request.args.get("dataset_id", store.active.id)
        bundle = store.load(dataset_id)
        if bundle.id != "builtin":
            return send_file(bundle.directory / "source.csv", as_attachment=True, download_name="traffic-dataset.csv")
        import pandas as pd
        source = ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv.gz"
        content = pd.read_csv(source, keep_default_na=False).to_csv(index=False).encode("utf-8")
        return send_file(io.BytesIO(content), mimetype="text/csv", as_attachment=True, download_name="original-uci-traffic.csv")

    @app.get("/download/sample-traffic.csv")
    def download_sample():
        import pandas as pd
        source = ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv.gz"
        frame = pd.read_csv(source, keep_default_na=False).drop_duplicates("date_time").head(168)
        sample = pd.DataFrame({"DateTime": frame.date_time, "Vehicles": frame.traffic_volume,
                               "temperature_c": (frame.temp - 273.15).round(2), "rainfall": frame.rain_1h,
                               "snowfall": frame.snow_1h, "cloud_cover": frame.clouds_all, "weather": frame.weather_main,
                               "holiday": frame.holiday})
        return send_file(io.BytesIO(sample.to_csv(index=False).encode("utf-8")), mimetype="text/csv",
                         as_attachment=True, download_name="sample-traffic.csv")

    @app.errorhandler(CSVError)
    def invalid_dataset(error):
        return jsonify(message=str(error)), 400

    @app.errorhandler(DatasetBusy)
    def dataset_busy(error):
        return jsonify(message=str(error)), 409

    @app.errorhandler(413)
    def too_large(_error):
        if request.path == "/api/datasets/inspect":
            return jsonify(message="Choose a CSV file no larger than 10 MB."), 413
        return jsonify(message="The request is too large. Send only the prediction inputs.", errors={"form": "Request exceeds 16 KB."}), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
