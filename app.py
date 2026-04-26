import os
import logging
import io
import zipfile
from datetime import timedelta, datetime
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from predict_new_river import predict_new_usgs_river

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

REG_BUNDLE_PATH = os.path.join(MODEL_DIR, "lightgbm_direct_recursive_regression_bundle.pkl")
CLF_PATH = os.path.join(MODEL_DIR, "final_low_leakage_is_flood_classifier.pkl")
CLF_MEDIANS_PATH = os.path.join(MODEL_DIR, "final_low_leakage_feature_medians.pkl")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "final_low_leakage_thresholds.csv")
CLF_SUMMARY_PATH = os.path.join(MODEL_DIR, "final_low_leakage_summary.csv")
DATA_PATH = os.path.join(DATA_DIR, "regression_dataset.csv")

LSTM_BUNDLE_PATH = os.path.join(MODEL_DIR, "peak_lstm_direct_recursive_bundle.pkl")
LSTM_FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "peak_lstm_feature_scaler.pkl")
LSTM_DIRECT_TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "peak_lstm_direct_target_scaler.pkl")
LSTM_RECURSIVE_TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "peak_lstm_recursive_target_scaler.pkl")
LSTM_FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "peak_lstm_feature_medians.pkl")

SAFETY_THRESHOLD = 0.30
TEST_SPLIT_FRACTION = 0.20
ALLOWED_WEB_RIVER_IDS = {
    "1464500",
    "1487000",
    "1493500",
    "1503000",
    "1516500",
    "1520000",
    "1531000",
    "1536000",
    "1539000",
    "1544000",
}

artifacts = {
    "regression_bundle": None,
    "classifier": None,
    "classifier_medians": None,
    "thresholds": None,
    "classifier_summary": None,
    "df": None,
    "test_df": None,

    "lstm_bundle": None,
    "lstm_direct_model": None,
    "lstm_recursive_model": None,
    "lstm_feature_scaler": None,
    "lstm_direct_target_scaler": None,
    "lstm_recursive_target_scaler": None,
    "lstm_feature_medians": None,

    "status": {},
}


def make_time_ordered_test_split(df, fraction=TEST_SPLIT_FRACTION):
    bundle_path = os.path.join(MODEL_DIR, "lightgbm_direct_recursive_regression_bundle.pkl")
    feature_cols = []
    target_cols = [f"flow_target_day{i}" for i in range(1, 11)]

    if os.path.exists(bundle_path):
        try:
            feature_cols = list(joblib.load(bundle_path).get("features", []))
        except Exception:
            feature_cols = []

    required_cols = [
        col for col in ["river_id", "date", "flow"] + feature_cols + target_cols
        if col in df.columns
    ]
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()

    test_parts = []
    for _, river_df in clean_df.sort_values("date").groupby(clean_df["river_id"].astype(str), sort=False):
        split_index = int(np.floor(len(river_df) * (1 - fraction)))
        split_index = min(max(split_index, 0), len(river_df) - 1)
        test_parts.append(river_df.iloc[split_index:].copy())

    if not test_parts:
        return clean_df.iloc[0:0].copy()

    return pd.concat(test_parts, ignore_index=True).sort_values(["river_id", "date"])


def clean_number(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def clean_number_list(values):
    return [clean_number(value) for value in values]


def safe_load_joblib(key, path):
    try:
        artifacts[key] = joblib.load(path)
        artifacts["status"][key] = "loaded"
    except Exception as e:
        artifacts["status"][key] = f"missing/error: {e}"


def safe_load_csv(key, path):
    try:
        artifacts[key] = pd.read_csv(path)
        artifacts["status"][key] = "loaded"
    except Exception as e:
        artifacts["status"][key] = f"missing/error: {e}"


def model_artifact_path(path_value):
    if not path_value:
        return ""

    if os.path.isabs(path_value):
        return path_value

    candidate = os.path.join(BASE_DIR, path_value)
    if os.path.exists(candidate):
        return candidate

    return os.path.join(MODEL_DIR, path_value)


def patch_keras_dense_quantization_config():
    try:
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import InputLayer
        old_from_config = Dense.from_config
        old_input_from_config = InputLayer.from_config

        @classmethod
        def patched_from_config(cls, config):
            if isinstance(config, dict):
                config.pop("quantization_config", None)
            return old_from_config(config)

        @classmethod
        def patched_input_from_config(cls, config):
            if isinstance(config, dict):
                config = dict(config)
                config.pop("optional", None)
                if "batch_shape" in config and "batch_input_shape" not in config:
                    config["batch_input_shape"] = config.pop("batch_shape")
            return old_input_from_config(config)

        Dense.from_config = patched_from_config
        InputLayer.from_config = patched_input_from_config
        artifacts["status"]["keras_compat_patch"] = "applied"
    except Exception as e:
        artifacts["status"]["keras_compat_patch"] = f"patch warning: {e}"


def build_peak_lstm_model(seq_len, feature_count, output_units):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(seq_len, feature_count)),
        tf.keras.layers.LSTM(160, return_sequences=True),
        tf.keras.layers.LayerNormalization(epsilon=0.001),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(80, return_sequences=False),
        tf.keras.layers.LayerNormalization(epsilon=0.001),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(output_units, activation="linear"),
    ])
    model(np.zeros((1, seq_len, feature_count), dtype=np.float32))
    return model


def load_peak_lstm_from_keras3_archive(path, output_units):
    import h5py

    bundle = artifacts["lstm_bundle"]
    seq_len = int(bundle.get("seq_len", 21))
    feature_count = len(bundle["features"])
    model = build_peak_lstm_model(seq_len, feature_count, output_units)

    with zipfile.ZipFile(path) as archive:
        weights_bytes = archive.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as weights_file:
        layer_map = [
            (model.layers[0], "layers/lstm/cell/vars"),
            (model.layers[1], "layers/layer_normalization/vars"),
            (model.layers[3], "layers/lstm_1/cell/vars"),
            (model.layers[4], "layers/layer_normalization_1/vars"),
            (model.layers[6], "layers/dense/vars"),
            (model.layers[8], "layers/dense_1/vars"),
            (model.layers[9], "layers/dense_2/vars"),
        ]

        for layer, weight_path in layer_map:
            weights = [
                weights_file[f"{weight_path}/{index}"][()]
                for index in range(len(layer.get_weights()))
            ]
            layer.set_weights(weights)

    return model


def load_lstm_model_with_fallback(path, output_units):
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False), "loaded"
    except Exception as primary_error:
        try:
            model = load_peak_lstm_from_keras3_archive(path, output_units)
            return model, "loaded via Keras 3 archive fallback"
        except Exception as fallback_error:
            raise RuntimeError(f"{primary_error}; fallback failed: {fallback_error}") from fallback_error


def load_artifacts():
    safe_load_joblib("regression_bundle", REG_BUNDLE_PATH)
    safe_load_joblib("classifier", CLF_PATH)
    safe_load_joblib("classifier_medians", CLF_MEDIANS_PATH)
    safe_load_csv("thresholds", THRESHOLDS_PATH)
    safe_load_csv("classifier_summary", CLF_SUMMARY_PATH)

    try:
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        artifacts["df"] = df
        artifacts["test_df"] = make_time_ordered_test_split(df)
        artifacts["status"]["df_model"] = "loaded"
        artifacts["status"]["test_split"] = f"loaded last {int(TEST_SPLIT_FRACTION * 100)}% per river"
    except Exception as e:
        artifacts["status"]["df_model"] = f"missing/error: {e}"

    safe_load_joblib("lstm_bundle", LSTM_BUNDLE_PATH)
    safe_load_joblib("lstm_feature_scaler", LSTM_FEATURE_SCALER_PATH)
    safe_load_joblib("lstm_direct_target_scaler", LSTM_DIRECT_TARGET_SCALER_PATH)
    safe_load_joblib("lstm_recursive_target_scaler", LSTM_RECURSIVE_TARGET_SCALER_PATH)
    safe_load_joblib("lstm_feature_medians", LSTM_FEATURE_MEDIANS_PATH)

    if TF_AVAILABLE and isinstance(artifacts["lstm_bundle"], dict):
        patch_keras_dense_quantization_config()

        try:
            direct_path = model_artifact_path(artifacts["lstm_bundle"].get("direct_model_path", ""))
            artifacts["lstm_direct_model"], status = load_lstm_model_with_fallback(direct_path, 10)
            artifacts["status"]["lstm_direct_model"] = status
        except Exception as e:
            artifacts["status"]["lstm_direct_model"] = f"missing/error: {e}"

        try:
            recursive_path = model_artifact_path(artifacts["lstm_bundle"].get("recursive_model_path", ""))
            artifacts["lstm_recursive_model"], status = load_lstm_model_with_fallback(recursive_path, 1)
            artifacts["status"]["lstm_recursive_model"] = status
        except Exception as e:
            artifacts["status"]["lstm_recursive_model"] = f"missing/error: {e}"
    else:
        artifacts["status"]["tensorflow"] = "not available, LSTM disabled"


load_artifacts()


# =========================================================
# Existing dataset mode
# =========================================================
def get_latest_row(river_id, selected_date=None):
    df = artifacts["test_df"]
    if df is None:
        return None

    river_df = df[df["river_id"].astype(str) == str(river_id)].copy()
    if river_df.empty:
        return None

    river_df = river_df.sort_values("date")

    if selected_date:
        selected_date = pd.to_datetime(selected_date, errors="coerce")
        if pd.notna(selected_date):
            river_df = river_df[river_df["date"] <= selected_date]

    if river_df.empty:
        return None

    return river_df.iloc[-1]


def get_river_history(river_id, selected_date=None):
    df = artifacts["df"]
    if df is None:
        return None

    river_df = df[df["river_id"].astype(str) == str(river_id)].copy()
    if river_df.empty:
        return None

    river_df = river_df.sort_values("date")

    if selected_date:
        selected_date = pd.to_datetime(selected_date, errors="coerce")
        if pd.notna(selected_date):
            river_df = river_df[river_df["date"] <= selected_date]

    if river_df.empty:
        return None

    return river_df


# =========================================================
# USGS + Weather live mode
# =========================================================
def fetch_usgs_site_metadata(site_id):
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": str(site_id).zfill(8),
        "siteOutput": "expanded",
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    lines = [line for line in r.text.splitlines() if not line.startswith("#")]
    text = "\n".join(lines)

    site_df = pd.read_csv(StringIO(text), sep="\t")

    if len(site_df) > 1:
        site_df = site_df.iloc[1:].copy()

    if site_df.empty:
        raise ValueError("No USGS site metadata found.")

    row = site_df.iloc[0]

    lat = pd.to_numeric(row.get("dec_lat_va", row.get("lat_va", np.nan)), errors="coerce")
    lon = pd.to_numeric(row.get("dec_long_va", row.get("long_va", np.nan)), errors="coerce")
    drainage = pd.to_numeric(row.get("drain_area_va", np.nan), errors="coerce")
    elev = pd.to_numeric(row.get("alt_va", np.nan), errors="coerce")

    return {
        "river_latitude": float(lat) if pd.notna(lat) else np.nan,
        "river_longitude": float(lon) if pd.notna(lon) else np.nan,
        "river_drainage_area_sqmi": float(drainage) if pd.notna(drainage) else np.nan,
        "river_elevation_ft": float(elev) if pd.notna(elev) else np.nan,
    }


def fetch_usgs_flow_history(site_id, days=120, selected_date=None):
    if selected_date:
        end = pd.to_datetime(selected_date, errors="coerce").date()
    else:
        end = datetime.today().date()

    start = end - timedelta(days=days)

    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": str(site_id).zfill(8),
        "parameterCd": "00060",
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
        "siteStatus": "all",
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError("No USGS flow data found for this site/date.")

    values = series[0]["values"][0]["value"]
    flow_df = pd.DataFrame(values)

    flow_df["date"] = pd.to_datetime(flow_df["dateTime"]).dt.tz_localize(None)
    flow_df["flow"] = pd.to_numeric(flow_df["value"], errors="coerce")

    return flow_df[["date", "flow"]].dropna().sort_values("date")


def fetch_weather_history(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "auto",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("No weather data found.")

    w = pd.DataFrame({"date": pd.to_datetime(daily["time"])})

    rain_values = daily.get("rain_sum")
    if rain_values is None:
        rain_values = daily.get("precipitation_sum")

    w["rain"] = pd.to_numeric(rain_values, errors="coerce")
    w["weather_temperature_2m_mean"] = pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce")
    w["weather_temperature_2m_max"] = pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce")
    w["weather_temperature_2m_min"] = pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce")
    w["weather_precipitation_hours"] = pd.to_numeric(daily.get("precipitation_hours"), errors="coerce")
    w["weather_wind_speed_10m_max"] = pd.to_numeric(daily.get("wind_speed_10m_max"), errors="coerce")
    w["weather_wind_gusts_10m_max"] = pd.to_numeric(daily.get("wind_gusts_10m_max"), errors="coerce")
    w["weather_et0_fao_evapotranspiration"] = pd.to_numeric(daily.get("et0_fao_evapotranspiration"), errors="coerce")

    w["weather_relative_humidity_2m_mean"] = np.nan
    w["weather_surface_pressure_mean"] = np.nan

    return w


def build_usgs_feature_table(flow_df, weather_df, meta):
    df = flow_df.merge(weather_df, on="date", how="left")
    df = df.sort_values("date").copy()

    for k, v in meta.items():
        df[k] = v

    for lag in range(1, 15):
        df[f"flow_lag{lag}"] = df["flow"].shift(lag)
        df[f"rain_lag{lag}"] = df["rain"].shift(lag)

    df["flow_3d_avg"] = df["flow"].rolling(3).mean()
    df["flow_7d_avg"] = df["flow"].rolling(7).mean()
    df["flow_14d_avg"] = df["flow"].rolling(14).mean()

    df["rain_3d"] = df["rain"].rolling(3).sum()
    df["rain_7d"] = df["rain"].rolling(7).sum()
    df["rain_14d"] = df["rain"].rolling(14).sum()

    df["flow_diff"] = df["flow"] - df["flow_lag1"]
    df["flow_acc"] = df["flow_diff"] - (df["flow_lag1"] - df["flow_lag2"])

    dry_streak = []
    count = 0
    for r in df["rain"].fillna(0):
        if r == 0:
            count += 1
        else:
            count = 0
        dry_streak.append(count)

    df["dry_streak"] = dry_streak

    month = df["date"].dt.month
    doy = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["rain_flow_interaction"] = df["rain_3d"] * df["flow_lag1"]
    df["rain_flow_ratio"] = df["rain_3d"] / (df["flow_lag1"] + 1)
    df["rain_per_area"] = df["rain_3d"] / (df["river_drainage_area_sqmi"] + 1e-6)

    return df


def build_live_usgs_pipeline(site_id, selected_date=None):
    meta = fetch_usgs_site_metadata(site_id)
    flow_df = fetch_usgs_flow_history(site_id, days=120, selected_date=selected_date)

    if flow_df.empty or len(flow_df) < 21:
        raise ValueError("Not enough USGS flow history. Need at least 21 daily records before selected date.")

    weather_df = fetch_weather_history(
        meta["river_latitude"],
        meta["river_longitude"],
        flow_df["date"].min(),
        flow_df["date"].max(),
    )

    feature_df = build_usgs_feature_table(flow_df, weather_df, meta)
    feature_df = feature_df.dropna(subset=["flow_lag14"]).copy()

    if feature_df.empty:
        raise ValueError("Feature engineering failed: not enough lagged history.")

    latest_row = feature_df.iloc[-1]
    return latest_row, feature_df


# =========================================================
# Prediction helpers
# =========================================================
def exact_model_features(model, fallback_features):
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    if hasattr(model, "booster_"):
        try:
            return list(model.booster_.feature_name())
        except Exception:
            pass

    return list(fallback_features)


def build_exact_input(row, features):
    df = artifacts["df"]
    X = pd.Series(index=features, dtype=float)

    for f in features:
        X[f] = row[f] if f in row.index else np.nan

    X = pd.to_numeric(X, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    existing = [f for f in features if df is not None and f in df.columns]
    medians = df[existing].median(numeric_only=True) if existing else pd.Series(dtype=float)

    for f in features:
        if f not in medians.index:
            medians[f] = 0.0

    X = X.fillna(medians.reindex(features).fillna(0.0))
    return pd.DataFrame([X], columns=features)


def predict_lightgbm(row):
    bundle = artifacts["regression_bundle"]

    if not isinstance(bundle, dict):
        raise ValueError("LightGBM regression bundle is not loaded.")

    direct_models = bundle["direct_models"]
    fallback_features = bundle["features"]
    horizon = int(bundle.get("horizon", 10))

    preds = []

    for h in range(1, horizon + 1):
        model = direct_models[h]
        model_features = exact_model_features(model, fallback_features)
        X_df = build_exact_input(row, model_features)

        pred_log = model.predict(X_df)[0]
        pred = np.expm1(pred_log)
        preds.append(max(float(pred), 0.0))

    return preds, []


def get_lstm_sequence_from_existing(river_id, selected_date):
    history_df = get_river_history(river_id, selected_date)
    if history_df is None:
        raise ValueError("No LSTM history found for this river/date.")

    return get_lstm_sequence_from_feature_df(history_df)


def get_lstm_sequence_from_feature_df(feature_df):
    bundle = artifacts["lstm_bundle"]

    if not isinstance(bundle, dict):
        raise ValueError("LSTM bundle is missing.")

    seq_len = int(bundle.get("seq_len", 21))
    features = list(bundle["features"])

    feature_df = feature_df.sort_values("date").copy()

    if len(feature_df) < seq_len:
        raise ValueError(f"LSTM needs at least {seq_len} engineered rows.")

    seq_df = feature_df.tail(seq_len).copy()

    X = seq_df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    medians = artifacts["lstm_feature_medians"]
    if medians is not None:
        X = X.fillna(medians.reindex(features))

    X = X.fillna(0.0)

    scaler = artifacts["lstm_feature_scaler"]
    X_scaled = scaler.transform(X) if scaler is not None else X.values

    return X_scaled.reshape(1, seq_len, len(features))


def inverse_lstm_target(values, scaler, cap=30000):
    arr = np.array(values).reshape(1, -1)
    target_transform = "log"

    if isinstance(artifacts["lstm_bundle"], dict):
        target_transform = artifacts["lstm_bundle"].get("target_transform", "log")

    if scaler is not None:
        try:
            arr = scaler.inverse_transform(arr)
        except Exception:
            pass

    if target_transform in {"log", "log1p"}:
        arr = np.clip(arr, -5, 10)
        arr = np.expm1(arr)

    arr = np.clip(arr, 0, cap)

    return arr.flatten().astype(float).tolist()


def predict_lstm_direct_existing(river_id, selected_date):
    model = artifacts["lstm_direct_model"]
    if model is None:
        raise ValueError("LSTM Direct model is not loaded. Check Loaded Artifacts.")

    X_seq = get_lstm_sequence_from_existing(river_id, selected_date)
    raw = model.predict(X_seq, verbose=0)
    raw = np.array(raw).reshape(1, -1)

    preds = inverse_lstm_target(raw, artifacts["lstm_direct_target_scaler"])
    return preds[:10], []


def predict_lstm_direct_live(feature_df):
    model = artifacts["lstm_direct_model"]
    if model is None:
        raise ValueError("LSTM Direct model is not loaded. Check Loaded Artifacts.")

    X_seq = get_lstm_sequence_from_feature_df(feature_df)
    raw = model.predict(X_seq, verbose=0)
    raw = np.array(raw).reshape(1, -1)

    preds = inverse_lstm_target(raw, artifacts["lstm_direct_target_scaler"])
    return preds[:10], []


def predict_lstm_recursive_live(feature_df, row=None):
    model = artifacts["lstm_recursive_model"]
    if model is None:
        raise ValueError("LSTM Recursive model is not loaded. Check Loaded Artifacts.")

    X_seq = get_lstm_sequence_from_feature_df(feature_df)

    raw_preds = []
    for _ in range(10):
        raw = model.predict(X_seq, verbose=0)
        raw_preds.append(float(np.array(raw).reshape(-1)[0]))

    preds = inverse_lstm_target(
        raw_preds,
        artifacts["lstm_recursive_target_scaler"],
        cap=30000
    )[:10]

    warnings = []

    if row is not None:
        last_flow = float(row.get("flow", 0))

        # Ø¥Ø°Ø§ Ø£ÙˆÙ„ ÙŠÙˆÙ… Ø·Ø§Ø± ÙƒØªÙŠØ±ØŒ Ø§Ø³ØªØ¹Ù…Ù„ LightGBM fallback
        if preds[0] > max(5000, last_flow * 5):
            lgb_preds, _ = predict_lightgbm(row)
            preds[0] = lgb_preds[0]
            warnings.append("First recursive day looked unstable, replaced with LightGBM day-1.")

    return preds, warnings


def make_classification_prediction(row):
    clf = artifacts["classifier"]
    medians = artifacts["classifier_medians"]

    if clf is None:
        raise ValueError("Classification model is not loaded.")

    if hasattr(clf, "feature_names_in_"):
        clf_features = list(clf.feature_names_in_)
    else:
        clf_features = list(medians.index)

    clean_medians = pd.Series(dtype=float)
    for f in clf_features:
        clean_medians[f] = medians[f] if medians is not None and f in medians.index else 0.0

    X = pd.Series(index=clf_features, dtype=float)

    for f in clf_features:
        X[f] = row[f] if f in row.index else np.nan

    X = pd.to_numeric(X, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(clean_medians)

    X_df = pd.DataFrame([X], columns=clf_features)

    prob = float(clf.predict_proba(X_df)[0][1])
    alert = prob >= SAFETY_THRESHOLD

    return prob, alert, []


def get_classifier_metrics():
    summary = artifacts["classifier_summary"]

    if summary is None or summary.empty:
        return {"precision": None, "recall": None, "f1": None, "roc_auc": None}

    row = summary.iloc[0]

    return {
        "precision": float(row["precision"]) if "precision" in row and pd.notna(row["precision"]) else None,
        "recall": float(row["recall"]) if "recall" in row and pd.notna(row["recall"]) else None,
        "f1": float(row["f1"]) if "f1" in row and pd.notna(row["f1"]) else None,
        "roc_auc": float(row["roc_auc"]) if "roc_auc" in row and pd.notna(row["roc_auc"]) else None,
    }


def fetch_usgs_observed(river_id, start_date, days=10):
    try:
        site = str(int(float(river_id))).zfill(8)
        end_date = start_date + timedelta(days=days)

        url = "https://waterservices.usgs.gov/nwis/dv/"
        params = {
            "format": "json",
            "sites": site,
            "parameterCd": "00060",
            "startDT": start_date.strftime("%Y-%m-%d"),
            "endDT": end_date.strftime("%Y-%m-%d"),
            "siteStatus": "all",
        }

        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()

        data = r.json()
        series = data.get("value", {}).get("timeSeries", [])

        if not series:
            return {}

        values = series[0]["values"][0]["value"]
        return {v["dateTime"][:10]: float(v["value"]) for v in values}

    except Exception as e:
        logging.warning(f"USGS observed data unavailable: {e}")
        return {}


def compute_metrics(pred_dates, preds, observed_dict):
    observed_list = []
    diffs = []
    y_true = []
    y_pred = []

    for d, p in zip(pred_dates, preds):
        key = d.strftime("%Y-%m-%d")
        obs = observed_dict.get(key)

        if obs is None:
            observed_list.append(None)
            diffs.append(None)
        else:
            observed_list.append(obs)
            diff = p - obs
            diffs.append(float(diff))
            y_true.append(obs)
            y_pred.append(p)

    if not y_true:
        return observed_list, diffs, None, None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return observed_list, diffs, mae, rmse


def observed_targets_from_row(row, horizon):
    observed = []
    for h in range(1, horizon + 1):
        observed.append(clean_number(row.get(f"flow_target_day{h}")))
    return observed


def compute_metrics_from_observed(preds, observed_list):
    diffs = []
    y_true = []
    y_pred = []

    for pred, obs in zip(preds, observed_list):
        clean_pred = clean_number(pred)
        clean_obs = clean_number(obs)

        if clean_pred is None or clean_obs is None:
            diffs.append(None)
            continue

        diffs.append(float(clean_pred - clean_obs))
        y_true.append(clean_obs)
        y_pred.append(clean_pred)

    if not y_true:
        return diffs, None, None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return diffs, mae, rmse


def add_forecast_diagnostic_warnings(warnings, preds, observed_list):
    clean_preds = [value for value in clean_number_list(preds) if value is not None]
    clean_observed = [value for value in clean_number_list(observed_list) if value is not None]

    if not clean_preds or not clean_observed:
        return

    peak_pred = max(clean_preds)
    peak_obs = max(clean_observed)

    if peak_obs >= 2 * max(peak_pred, 1.0):
        warnings.append(
            "Large gap detected: this testing example contains a sharp flood peak, and the selected model is underpredicting the peak."
        )


# =========================================================
# Routes
# =========================================================
@app.route("/")
def index():
    df = artifacts["test_df"]
    regression_bundle_loaded = isinstance(artifacts["regression_bundle"], dict)
    model_options = [
        {
            "value": "lightgbm",
            "label": "LightGBM Direct",
            "available": regression_bundle_loaded,
            "note": "loaded" if regression_bundle_loaded else "missing",
        },
        {
            "value": "lstm_direct",
            "label": "LSTM Direct",
            "available": artifacts["lstm_direct_model"] is not None,
            "note": "loaded" if artifacts["lstm_direct_model"] is not None else "unavailable",
        },
        {
            "value": "lstm_recursive",
            "label": "LSTM Recursive",
            "available": artifacts["lstm_recursive_model"] is not None,
            "note": "loaded" if artifacts["lstm_recursive_model"] is not None else "unavailable",
        },
    ]

    if df is None:
        rivers = []
        min_date = ""
        max_date = ""
    else:
        rivers = [
            river_id
            for river_id in sorted(df["river_id"].dropna().astype(str).unique())
            if river_id in ALLOWED_WEB_RIVER_IDS
        ]
        min_date = df["date"].min().strftime("%Y-%m-%d")
        max_date = df["date"].max().strftime("%Y-%m-%d")

    return render_template(
        "index.html",
        rivers=rivers,
        artifact_status=artifacts["status"],
        model_options=model_options,
        min_date=min_date,
        max_date=max_date,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        river_id = data.get("river_id")
        selected_date = data.get("selected_date")
        model_choice = data.get("model_choice", "lightgbm")
        use_live_usgs = bool(data.get("use_live_usgs", False))

        if not river_id:
            return jsonify({"error": "river_id is required"}), 400

        if use_live_usgs:
            warnings = []
            forecast_df = predict_new_usgs_river(river_id, selected_date=selected_date)
            preds = forecast_df["predicted_flow"].astype(float).tolist()
            pred_dates = pd.to_datetime(forecast_df["date"]).tolist()
            row = pd.Series(forecast_df.attrs.get("latest_features", {}))

            if model_choice != "lightgbm":
                warnings.append("New USGS river prediction uses the saved LightGBM bundle.")

            warnings.append("New USGS mode: live USGS/Open-Meteo data was fetched and engineered for prediction only.")
            warnings.append("Processed datasets were used as-is; missing live values were filled from regression dataset medians.")

            try:
                flood_prob, flood_alert, clf_warnings = make_classification_prediction(row)
                warnings.extend(clf_warnings)
            except Exception as e:
                logging.warning(f"Classification unavailable for live USGS request: {e}")
                flood_prob = 0.0
                flood_alert = False
                warnings.append("Flood classification was unavailable for this live USGS request.")

            observed_dict = fetch_usgs_observed(river_id, pred_dates[0], len(preds))
            observed_list, diffs, mae, rmse = compute_metrics(pred_dates, preds, observed_dict)
            classifier_metrics = get_classifier_metrics()

            if not observed_dict:
                warnings.append("No external observed USGS data available for these prediction dates. Forecast still shown.")

            warnings.append("Flood threshold 0.30 is safety-first and prioritizes recall.")

            return jsonify({
                "river_id": str(river_id),
                "mode": "live_usgs",
                "model_used": "LightGBM Direct Live USGS",
                "selected_input_date": forecast_df.attrs.get("latest_input_date", ""),
                "prediction_dates": [d.strftime("%Y-%m-%d") for d in pred_dates],
                "predicted_flows": clean_number_list(preds),
                "flood_probability": clean_number(flood_prob) or 0.0,
                "flood_alert": bool(flood_alert),
                "threshold": SAFETY_THRESHOLD,
                "observed_flows": clean_number_list(observed_list),
                "differences": clean_number_list(diffs),
                "mae": clean_number(mae),
                "rmse": clean_number(rmse),
                "classifier_metrics": classifier_metrics,
                "warnings": warnings,
            })

        warnings = []
        reg_warnings = []

        if str(river_id) not in ALLOWED_WEB_RIVER_IDS:
            return jsonify({"error": "This river ID is not enabled in the web prediction list."}), 400

        row = get_latest_row(river_id, selected_date)

        if row is None:
            return jsonify({"error": "No data found for this river/date. Enter a USGS Site ID for live mode."}), 404

        if not selected_date:
            selected_date = row["date"].strftime("%Y-%m-%d")

        if model_choice == "lstm_direct" and artifacts["lstm_direct_model"] is not None:
            direct_preds, direct_warnings = predict_lstm_direct_existing(river_id, selected_date)
            lgb_preds, lgb_warnings = predict_lightgbm(row)
            direct_is_bad = (
                np.std(direct_preds) < 50 or
                np.mean(direct_preds) > max(5000, float(row["flow"]) * 5)
            )

            if direct_is_bad:
                preds = lgb_preds
                reg_warnings = direct_warnings + lgb_warnings
                reg_warnings.append("LSTM Direct was unstable; fallback to LightGBM.")
                model_label = "LightGBM Fallback"
            else:
                preds = direct_preds
                reg_warnings = direct_warnings
                model_label = "LSTM Direct"
        elif model_choice == "lstm_recursive" and artifacts["lstm_recursive_model"] is not None:
            history_df = get_river_history(river_id, selected_date)
            rec_preds, rec_warnings = predict_lstm_recursive_live(history_df, row)
            lgb_preds, lgb_warnings = predict_lightgbm(row)

            if rec_preds[0] > 5000:
                preds = lgb_preds
                reg_warnings.append("LSTM Recursive was unstable; fallback to LightGBM.")
            else:
                preds = rec_preds[:1] + lgb_preds[1:]
                reg_warnings = rec_warnings + lgb_warnings
                reg_warnings.append("Hybrid mode: first day uses LSTM Recursive; remaining days use LightGBM.")
            model_label = "Hybrid LSTM Recursive + LightGBM"
        else:
            if model_choice in {"lstm_direct", "lstm_recursive"}:
                warnings.append("Selected LSTM model is unavailable in this environment, so LightGBM was used.")
            preds, reg_warnings = predict_lightgbm(row)
            model_label = "LightGBM Direct"

        warnings.extend(reg_warnings)
        flood_prob, flood_alert, clf_warnings = make_classification_prediction(row)
        warnings.extend(clf_warnings)

        start_date = pd.to_datetime(row["date"]) + timedelta(days=1)
        pred_dates = [start_date + timedelta(days=i) for i in range(len(preds))]
        observed_list = observed_targets_from_row(row, len(preds))
        diffs, mae, rmse = compute_metrics_from_observed(preds, observed_list)
        classifier_metrics = get_classifier_metrics()

        add_forecast_diagnostic_warnings(warnings, preds, observed_list)
        warnings.append("Observed flow comes from testing target columns in the processed regression dataset.")
        warnings.append("Flood threshold 0.30 is safety-first and prioritizes recall.")

        return jsonify({
            "river_id": str(river_id),
            "mode": "processed_dataset",
            "model_used": model_label,
            "selected_input_date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "prediction_dates": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "predicted_flows": clean_number_list(preds),
            "flood_probability": clean_number(flood_prob) or 0.0,
            "flood_alert": bool(flood_alert),
            "threshold": SAFETY_THRESHOLD,
            "observed_flows": clean_number_list(observed_list),
            "differences": clean_number_list(diffs),
            "mae": clean_number(mae),
            "rmse": clean_number(rmse),
            "classifier_metrics": classifier_metrics,
            "warnings": warnings,
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/api/artifacts")
def api_artifacts():
    bundle = artifacts["regression_bundle"]

    return jsonify({
        "status": artifacts["status"],
        "regression_keys": list(bundle.keys()) if isinstance(bundle, dict) else [],
        "regression_feature_count": len(bundle["features"]) if isinstance(bundle, dict) and "features" in bundle else None,
        "classifier_feature_count": len(artifacts["classifier_medians"]) if artifacts["classifier_medians"] is not None else None,
        "testing_rows": int(len(artifacts["test_df"])) if artifacts["test_df"] is not None else 0,
        "testing_river_count": int(artifacts["test_df"]["river_id"].nunique()) if artifacts["test_df"] is not None else 0,
        "classifier_metrics": get_classifier_metrics(),
        "tensorflow_available": TF_AVAILABLE,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
