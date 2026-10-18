"""Local dataset lifecycle: inspect, train, save, and atomically select a model."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from uuid import uuid4

import joblib

from dataset_import import CSVError, inspect_csv, prepare_uploaded
from traffic_model import FEATURES
from train import fit_dataset

FEATURE_LABELS = {"hour": "Hour of day", "weekday": "Day of week", "month": "Month", "holiday": "Holiday indicator",
                  "temperature": "Temperature", "rainfall": "Rainfall", "snowfall": "Snowfall", "cloud_cover": "Cloud cover", "weather": "Weather category"}


class DatasetBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetBundle:
    id: str
    directory: Path
    model: object
    summary: dict


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class DatasetStore:
    def __init__(self, builtin_dir, root):
        self.builtin_dir = Path(builtin_dir)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.operation_lock = threading.Lock()
        self.selection_lock = threading.Lock()
        self.notice = ""
        self._active = self.load("builtin")
        selection = self.root / "active.json"
        if selection.exists():
            try:
                self._active = self.load(json.loads(selection.read_text(encoding="utf-8"))["id"])
            except (OSError, ValueError, KeyError, EOFError):
                self.notice = "The previously selected dataset could not be loaded. The original UCI dataset is active; you can select another saved dataset below."

    @property
    def active(self):
        with self.selection_lock:
            return self._active

    def directory(self, dataset_id):
        if dataset_id == "builtin":
            return self.builtin_dir
        if not isinstance(dataset_id, str) or not re.fullmatch(r"[0-9a-f]{32}", dataset_id):
            raise CSVError("Choose a valid saved dataset.")
        return self.root / dataset_id

    def load(self, dataset_id):
        directory = self.directory(dataset_id)
        if dataset_id == "builtin":
            info = {"id": "builtin", "name": "Original UCI dataset", "location": "I-94 westbound · Station 301, Minnesota",
                    "builtin": True, "filename": "Metro_Interstate_Traffic_Volume.csv"}
        else:
            try:
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                if metadata.get("status") != "ready":
                    raise CSVError("Train this dataset successfully before selecting it.")
                info = metadata["dataset"]
            except FileNotFoundError as error:
                raise CSVError("That saved dataset could not be found.") from error
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        model = joblib.load(directory / "model.joblib")
        summary["dataset"] = info
        summary.setdefault("feature_columns", FEATURES.copy())
        summary["feature_labels"] = [FEATURE_LABELS[key] for key in summary["feature_columns"]]
        summary["weather_features"] = [key for key in ["temperature", "rainfall", "snowfall", "cloud_cover", "weather"] if key in summary["feature_columns"]]
        improvement = summary["mae_improvement"]
        summary["mae_comparison_label"] = ("Baseline MAE is zero" if improvement is None else
                                            f"{abs(improvement):.1f}% {'lower' if improvement >= 0 else 'higher'} MAE")
        return DatasetBundle(dataset_id, directory, model, summary)

    def catalog(self):
        base = self.load_catalog_item("builtin", self.builtin_dir)
        items = [base]
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and re.fullmatch(r"[0-9a-f]{32}", directory.name):
                try:
                    item = self.load_catalog_item(directory.name, directory)
                    if item:
                        items.append(item)
                except (OSError, ValueError, KeyError):
                    continue
        return {"active_id": self.active.id, "datasets": items, "notice": self.notice}

    def load_catalog_item(self, dataset_id, directory):
        if dataset_id == "builtin":
            info = {"id": "builtin", "name": "Original UCI dataset", "location": "I-94 westbound · Station 301, Minnesota", "builtin": True}
        else:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") != "ready":
                return None
            info = metadata["dataset"].copy()
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        return {**info, "records": summary["records"], "metrics": summary["metrics"], "start": summary["start"], "end": summary["end"]}

    def inspect(self, content, filename, name="", location=""):
        if not filename.lower().endswith(".csv"):
            raise CSVError("Choose a CSV file with a .csv extension.")
        filename = Path(filename.replace("\\", "/")).name[:160]
        if len(name) > 80 or len(location) > 120:
            raise CSVError("Dataset names can have up to 80 characters and location labels up to 120.")
        inspection, _raw = inspect_csv(content)
        dataset_id = uuid4().hex
        info = {"id": dataset_id, "name": name.strip() or Path(filename).stem[:80],
                "location": location.strip() or "Location not supplied", "builtin": False, "filename": filename}
        directory = self.directory(dataset_id)
        directory.mkdir()
        (directory / "source.csv").write_bytes(content)
        metadata = {"status": "inspected", "dataset": info, "inspection": inspection,
                    "sha256": hashlib.sha256(content).hexdigest()}
        write_json(directory / "metadata.json", metadata)
        return {"id": dataset_id, "dataset": info, **inspection}

    def _activate(self, bundle):
        with self.selection_lock:
            write_json(self.root / "active.json", {"id": bundle.id})
            self._active = bundle
            self.notice = ""

    def activate(self, dataset_id):
        if not self.operation_lock.acquire(blocking=False):
            raise DatasetBusy("Another dataset is being trained or selected. Wait for it to finish, then try again.")
        try:
            bundle = self.load(dataset_id)
            self._activate(bundle)
            return bundle
        finally:
            self.operation_lock.release()

    def train(self, dataset_id, temperature_unit=None, location=None):
        if dataset_id == "builtin":
            raise CSVError("The original dataset is already trained. Select it from Saved datasets.")
        if not self.operation_lock.acquire(blocking=False):
            raise DatasetBusy("Another dataset is being trained or selected. Wait for it to finish, then try again.")
        try:
            directory = self.directory(dataset_id)
            try:
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise CSVError("Upload and inspect a CSV before training it.") from error
            if metadata["status"] == "ready":
                bundle = self.load(dataset_id)
            else:
                inspection, raw = inspect_csv((directory / "source.csv").read_bytes())
                frame, record_count, selected = prepare_uploaded(raw, inspection, temperature_unit, location)
                info = metadata["dataset"]
                if selected is not None:
                    column = inspection["mapping"]["location"]
                    label = f"{column}: {selected}"
                    info["location"] = f"{info['location']} · {label}" if info["location"] != "Location not supplied" else label
                source = {"name": info["filename"], "url": f"/download/dataset.csv?dataset_id={dataset_id}",
                          "author": "User-provided CSV", "license": "As supplied by the dataset owner", "archive_sha256": metadata["sha256"]}
                fit_dataset(frame, directory, source, record_count, inspection["feature_columns"])
                metadata.update(status="ready", selected_location=selected,
                                temperature_unit=temperature_unit or inspection["temperature_unit"])
                write_json(directory / "metadata.json", metadata)
                bundle = self.load(dataset_id)
            self._activate(bundle)
            return bundle
        finally:
            self.operation_lock.release()
