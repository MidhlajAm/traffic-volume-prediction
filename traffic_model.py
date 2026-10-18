"""Shared feature engineering and inference for training and Flask."""
from datetime import date
import math
import re

import numpy as np
import pandas as pd

NUMERIC = ["temperature", "rainfall", "snowfall", "cloud_cover"]
CATEGORICAL = ["hour", "weekday", "month", "weather", "holiday"]
FEATURES = NUMERIC + CATEGORICAL


def prepare_data(raw):
    """Keep the first source row per timestamp; never split a repeated hour."""
    frame = raw.copy()
    frame["date_time"] = pd.to_datetime(frame["date_time"], errors="raise")
    frame = frame.drop_duplicates("date_time", keep="first")
    frame = frame.sort_values("date_time", kind="stable").reset_index(drop=True)
    frame["temperature"] = pd.to_numeric(frame["temp"], errors="coerce") - 273.15
    # Implausible sensor readings (including zero Kelvin) are not outdoor weather.
    frame.loc[~frame["temperature"].between(-60, 60), "temperature"] = np.nan
    for source, target in [("rain_1h", "rainfall"), ("snow_1h", "snowfall"), ("clouds_all", "cloud_cover")]:
        frame[target] = pd.to_numeric(frame[source], errors="coerce")
        frame.loc[frame[target] < 0, target] = np.nan
    frame.loc[frame["cloud_cover"] > 100, "cloud_cover"] = np.nan
    frame["hour"] = frame.date_time.dt.hour
    frame["weekday"] = frame.date_time.dt.weekday
    frame["month"] = frame.date_time.dt.month
    frame["weather"] = frame.weather_main.astype(str)
    frame["holiday"] = frame.holiday.ne("None").astype(int)
    return frame


def chronological_split(frame):
    boundary = int(len(frame) * 0.8)
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy()


def predict_volume(model, features):
    """The exact same nonnegative, unrounded rule is used in both paths."""
    columns = list(getattr(model, "feature_names_in_", FEATURES))
    return np.maximum(0.0, model.predict(features[columns]))


class InputError(ValueError):
    def __init__(self, errors):
        super().__init__("Check the highlighted inputs and try again.")
        self.errors = errors


def validate_input(payload, weather_categories, feature_columns=None):
    feature_columns = FEATURES if feature_columns is None else feature_columns
    if not isinstance(payload, dict):
        raise InputError({"form": "Send a JSON object containing the prediction inputs."})
    errors = {}
    cleaned = {}
    value = payload.get("date")
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        selected_date = date.fromisoformat(value)
        cleaned["date"] = selected_date.isoformat()
    except (ValueError, TypeError):
        errors["date"] = "Choose a valid date in YYYY-MM-DD format."
    limits = {"hour": (0, 23), "temperature": (-60, 60), "rainfall": (0, None),
              "snowfall": (0, None), "cloud_cover": (0, 100)}
    for key, (low, high) in limits.items():
        if key != "hour" and key not in feature_columns:
            continue
        value = payload.get(key)
        try:
            if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
                raise ValueError
            number = float(value)
            if not math.isfinite(number) or number < low or (high is not None and number > high):
                raise ValueError
            if key == "hour" and not number.is_integer():
                raise ValueError
            cleaned[key] = int(number) if key == "hour" else number
        except (TypeError, ValueError, OverflowError):
            bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
            errors[key] = f"Enter a {'whole ' if key == 'hour' else ''}number {bounds}."
    if "weather" in feature_columns and payload.get("weather") not in weather_categories:
        errors["weather"] = "Select one of the available weather conditions."
    elif "weather" in feature_columns:
        cleaned["weather"] = payload["weather"]
    if "holiday" in feature_columns and type(payload.get("holiday")) is not bool:
        errors["holiday"] = "Holiday must be true or false."
    elif "holiday" in feature_columns:
        cleaned["holiday"] = payload["holiday"]
    if errors:
        raise InputError(errors)
    features = {key: value for key, value in cleaned.items() if key in feature_columns}
    features.update(weekday=selected_date.weekday(), month=selected_date.month)
    if "holiday" in feature_columns:
        features["holiday"] = int(cleaned["holiday"])
    return cleaned, pd.DataFrame([features], columns=feature_columns)
