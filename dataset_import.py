"""Recognize supported traffic CSV columns and validate hourly observations."""
import csv
import io
import re

import numpy as np
import pandas as pd

from traffic_model import CATEGORICAL, NUMERIC

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MIN_HOURS = 100
ALIASES = {
    "timestamp": ["date_time", "datetime", "timestamp", "recorded_at", "observation_time"],
    "date": ["date", "observation_date"], "time": ["time", "hour", "hour_of_day"],
    "traffic_volume": ["traffic_volume", "vehicles", "vehicle_count", "traffic_count", "vehicle_volume", "volume"],
    "temperature": ["temp", "temperature", "temp_c", "temperature_c", "temperature_celsius", "temp_celsius", "temp_k", "temperature_k", "temperature_kelvin"],
    "rainfall": ["rain_1h", "rainfall", "rainfall_mm", "rain_mm", "rain"],
    "snowfall": ["snow_1h", "snowfall", "snowfall_mm", "snow_mm", "snow"],
    "cloud_cover": ["clouds_all", "cloud_cover", "cloud_cover_percent", "clouds"],
    "weather": ["weather_main", "weather", "weather_condition"],
    "holiday": ["holiday", "is_holiday"],
    "location": ["junction", "junction_id", "station", "station_id", "location", "road", "sensor_id", "intersection"],
}


class CSVError(ValueError):
    pass


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_csv(content):
    if not content or len(content) > MAX_BYTES:
        raise CSVError("Choose a non-empty CSV file no larger than 10 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CSVError("Save the CSV using UTF-8 encoding, then upload it again.") from error
    try:
        separator = csv.Sniffer().sniff(text[:8192], delimiters=",;\t").delimiter
    except csv.Error:
        separator = ","
    headers = next(csv.reader(io.StringIO(text), delimiter=separator), [])
    if not headers or any(not normalized(name) for name in headers) or len({normalized(name) for name in headers}) != len(headers):
        raise CSVError("The CSV needs a header row with distinct, non-empty column names.")
    try:
        raw = pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise CSVError("This file could not be read as a CSV. Check the separators and row lengths.") from error
    if raw.empty or len(raw) > MAX_ROWS:
        raise CSVError("Use between 100 and 100,000 data rows per CSV.")
    return raw


def detect_columns(raw):
    mapping = {}
    for field, aliases in ALIASES.items():
        candidates = [column for column in raw.columns if normalized(column) in {normalized(name) for name in aliases}]
        if len(candidates) > 1:
            raise CSVError(f"More than one column matches {field}: {', '.join(candidates)}. Keep one matching column.")
        if candidates:
            mapping[field] = candidates[0]
    if "traffic_volume" not in mapping:
        raise CSVError("No traffic-volume column found. Use traffic_volume, vehicles, vehicle_count, traffic_count, or volume.")
    if "timestamp" not in mapping and not all(key in mapping for key in ["date", "time"]):
        raise CSVError("No hourly date/time column found. Use DateTime/date_time/timestamp, or separate Date and Time/Hour columns.")
    return mapping


def timestamps(raw, mapping):
    if "timestamp" in mapping:
        text = raw[mapping["timestamp"]].str.strip()
        if not text.str.contains(r"[ T]\d{1,2}:\d{2}", regex=True).all():
            raise CSVError("Timestamps must include a time, for example 2024-01-15 08:00:00. For date-only data, also provide an Hour column.")
    else:
        dates = raw[mapping["date"]].str.strip()
        times = raw[mapping["time"]].str.strip()
        if times.str.fullmatch(r"\d{1,2}").all():
            if not pd.to_numeric(times).between(0, 23).all():
                raise CSVError("Hour values must be integers from 0 to 23.")
            times = times.str.zfill(2) + ":00:00"
        text = dates + " " + times
    if not text.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}[ T]").all():
        raise CSVError("Use unambiguous year-first dates: YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD with a separate Hour column.")
    if text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any():
        raise CSVError("Use the road's local date/time without timezone suffixes in the CSV.")
    values = pd.to_datetime(text, format="mixed", errors="coerce")
    if values.isna().any():
        raise CSVError(f"{int(values.isna().sum())} rows contain invalid dates or times. Correct them before uploading.")
    if ((values.dt.minute != 0) | (values.dt.second != 0) | (values.dt.microsecond != 0)).any():
        raise CSVError("This website predicts hourly counts. Aggregate sub-hourly records to whole-hour timestamps first.")
    return values


def inspect_csv(content):
    raw = read_csv(content)
    mapping = detect_columns(raw)
    times = timestamps(raw, mapping)
    target = pd.to_numeric(raw[mapping["traffic_volume"]], errors="coerce")
    if not (np.isfinite(target) & (target >= 0)).all():
        raise CSVError("Every traffic-volume value must be a finite, nonnegative number. Remove missing counts or correct the affected rows.")
    if "weather" in mapping and raw[mapping["weather"]].nunique() > 32:
        raise CSVError("Use at most 32 weather categories. Check that the detected weather column contains conditions rather than IDs or descriptions.")
    locations = []
    if "location" in mapping:
        location_values = raw[mapping["location"]].str.strip()
        if location_values.eq("").any() or location_values.nunique() > 100:
            raise CSVError("Location values must be non-empty, with at most 100 distinct roads or stations per CSV.")
        locations = [{"value": str(value), "hours": int(times[location_values == value].nunique())}
                     for value in sorted(location_values.unique())]
        if max(item["hours"] for item in locations) < MIN_HOURS:
            raise CSVError("At least one location needs 100 distinct recorded hours for training and evaluation.")
    elif times.nunique() < MIN_HOURS:
        raise CSVError("Provide at least 100 distinct recorded hours for training and evaluation.")
    unit = None
    if "temperature" in mapping:
        name = normalized(mapping["temperature"])
        if name.endswith(("kelvin", "tempk", "temperaturek")):
            unit = "K"
        elif name.endswith(("celsius", "tempc", "temperaturec")):
            unit = "C"
        elif name == "temp" and {"rain_1h", "snow_1h", "clouds_all", "weather_main"} <= set(raw.columns):
            unit = "K"
    features = [key for key in NUMERIC if key in mapping] + ["hour", "weekday", "month"]
    features += [key for key in ["weather", "holiday"] if key in mapping]
    keys = pd.DataFrame({"time": times})
    if locations:
        keys["location"] = raw[mapping["location"]].str.strip()
    return {"mapping": mapping, "rows": len(raw), "unique_hours": int(len(keys.drop_duplicates())),
            "duplicates": int(keys.duplicated().sum()), "start": times.min().isoformat(), "end": times.max().isoformat(),
            "temperature_unit": unit, "feature_columns": features, "locations": locations,
            "preview": [{"date_time": stamp.isoformat(), "traffic_volume": float(count)}
                        for stamp, count in zip(times.head(5), target.head(5))]}, raw


def prepare_uploaded(raw, inspection, temperature_unit=None, location=None):
    mapping = inspection["mapping"]
    frame = pd.DataFrame({"date_time": timestamps(raw, mapping),
                          "traffic_volume": pd.to_numeric(raw[mapping["traffic_volume"]])})
    if "location" in mapping:
        choices = [item["value"] for item in inspection["locations"]]
        if location is None and len(choices) == 1:
            location = choices[0]
        if location not in choices:
            raise CSVError("Choose one road or station from the detected locations before training.")
        selected = raw[mapping["location"]].str.strip().eq(location)
        raw, frame = raw.loc[selected].copy(), frame.loc[selected].copy()
    records = len(frame)
    for key in NUMERIC:
        if key in mapping:
            frame[key] = pd.to_numeric(raw[mapping[key]], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if key == "temperature":
                unit = temperature_unit or inspection["temperature_unit"]
                if unit not in ["C", "K"]:
                    raise CSVError("Confirm whether the temperature column uses Celsius or Kelvin before training.")
                if unit == "K":
                    frame[key] -= 273.15
                frame.loc[~frame[key].between(-60, 60), key] = np.nan
            else:
                frame.loc[frame[key] < 0, key] = np.nan
                if key == "cloud_cover":
                    frame.loc[frame[key] > 100, key] = np.nan
    if "weather" in mapping:
        frame["weather"] = raw[mapping["weather"]].str.strip().replace("", "Unknown")
    if "holiday" in mapping:
        frame["holiday"] = (~raw[mapping["holiday"]].str.strip().str.lower().isin(["", "none", "no", "false", "0", "0.0"])).astype(int)
    frame = frame.drop_duplicates("date_time", keep="first").sort_values("date_time", kind="stable").reset_index(drop=True)
    if len(frame) < MIN_HOURS:
        raise CSVError("The selected location needs at least 100 distinct recorded hours.")
    for key, values in [("hour", frame.date_time.dt.hour), ("weekday", frame.date_time.dt.weekday), ("month", frame.date_time.dt.month)]:
        frame[key] = values
    if frame.iloc[:int(len(frame) * .8)].traffic_volume.nunique() < 2:
        raise CSVError("Training traffic counts must vary. This file's training period contains only one traffic-volume value.")
    return frame, records, location
