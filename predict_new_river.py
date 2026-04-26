import os
from datetime import datetime, timedelta
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bundle_path = os.path.join(BASE_DIR, "models", "lightgbm_direct_recursive_regression_bundle.pkl")
regression_data_path = os.path.join(BASE_DIR, "data", "processed", "regression_dataset.csv")

bundle = joblib.load(bundle_path)
df_train = pd.read_csv(regression_data_path)

FEATURES = bundle["features"]
MODELS = bundle["direct_models"]
HORIZON = bundle.get("horizon", 10)


def fetch_usgs_site_metadata(site_id):
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": str(site_id).zfill(8),
        "siteOutput": "expanded",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    lines = [line for line in response.text.splitlines() if not line.startswith("#")]
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

    if pd.isna(lat) or pd.isna(lon):
        raise ValueError("USGS site metadata did not include usable latitude/longitude.")

    return {
        "river_latitude": float(lat),
        "river_longitude": float(lon),
        "river_drainage_area_sqmi": float(drainage) if pd.notna(drainage) else np.nan,
        "river_elevation_ft": float(elev) if pd.notna(elev) else np.nan,
    }


def fetch_usgs_flow_history(site_id, days=120, selected_date=None):
    if selected_date:
        end = pd.to_datetime(selected_date, errors="coerce")
        if pd.isna(end):
            raise ValueError("Invalid selected date for live USGS prediction.")
        end = end.date()
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

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError("No USGS flow data found for this site.")

    values = series[0].get("values", [{}])[0].get("value", [])
    flow_df = pd.DataFrame(values)
    if flow_df.empty:
        raise ValueError("USGS flow response did not include daily values.")

    flow_df["date"] = pd.to_datetime(flow_df["dateTime"]).dt.tz_localize(None).dt.normalize()
    flow_df["flow"] = pd.to_numeric(flow_df["value"], errors="coerce")

    return flow_df[["date", "flow"]].dropna().sort_values("date")


def fetch_weather_history(lat, lon, start_date, end_date):
    daily_vars = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "precipitation_hours",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "et0_fao_evapotranspiration",
        "surface_pressure_mean",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "daily": ",".join(daily_vars),
        "timezone": "auto",
    }

    url = "https://archive-api.open-meteo.com/v1/archive"
    response = requests.get(url, params=params, timeout=20)

    if response.status_code >= 400:
        fallback_vars = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "et0_fao_evapotranspiration",
        ]
        params["daily"] = ",".join(fallback_vars)
        response = requests.get(url, params=params, timeout=20)

    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("No weather data found.")

    weather_df = pd.DataFrame({"date": pd.to_datetime(pd.Series(daily["time"])).dt.normalize()})

    rain_values = daily.get("rain_sum")
    if rain_values is None:
        rain_values = daily.get("precipitation_sum")

    weather_df["rain"] = pd.to_numeric(rain_values, errors="coerce")
    weather_df["weather_temperature_2m_mean"] = pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce")
    weather_df["weather_temperature_2m_max"] = pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce")
    weather_df["weather_temperature_2m_min"] = pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce")
    weather_df["weather_precipitation_hours"] = pd.to_numeric(daily.get("precipitation_hours"), errors="coerce")
    weather_df["weather_relative_humidity_2m_mean"] = pd.to_numeric(daily.get("relative_humidity_2m_mean"), errors="coerce")
    weather_df["weather_wind_speed_10m_max"] = pd.to_numeric(daily.get("wind_speed_10m_max"), errors="coerce")
    weather_df["weather_wind_gusts_10m_max"] = pd.to_numeric(daily.get("wind_gusts_10m_max"), errors="coerce")
    weather_df["weather_et0_fao_evapotranspiration"] = pd.to_numeric(
        daily.get("et0_fao_evapotranspiration"),
        errors="coerce",
    )
    weather_df["weather_surface_pressure_mean"] = pd.to_numeric(daily.get("surface_pressure_mean"), errors="coerce")

    return weather_df


def build_feature_table(flow_df, weather_df, meta):
    flow_df = flow_df.copy()
    weather_df = weather_df.copy()
    flow_df["date"] = pd.to_datetime(flow_df["date"]).dt.normalize()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()

    df = flow_df.merge(weather_df, on="date", how="left")
    df = df.sort_values("date").copy()

    if "rain" not in df.columns:
        df["rain"] = np.nan

    for key, value in meta.items():
        df[key] = value

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

    dry_streaks = []
    count = 0
    for rain in df["rain"].fillna(0):
        if rain == 0:
            count += 1
        else:
            count = 0
        dry_streaks.append(count)
    df["dry_streak"] = dry_streaks

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


def _training_medians():
    existing = [c for c in FEATURES if c in df_train.columns]
    medians = df_train[existing].median(numeric_only=True)

    for f in FEATURES:
        if f not in medians.index:
            medians[f] = 0.0

    return medians


def _build_model_input(latest):
    X = latest.reindex(FEATURES)
    X = pd.to_numeric(X, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    medians = _training_medians()
    X = X.fillna(medians.reindex(FEATURES).fillna(0.0))

    return pd.DataFrame([X], columns=FEATURES)


def _latest_feature_row(site_id, selected_date=None):
    meta = fetch_usgs_site_metadata(site_id)
    flow_df = fetch_usgs_flow_history(site_id, days=120, selected_date=selected_date)

    if flow_df.empty:
        raise ValueError("No usable USGS flow history found.")

    weather_df = fetch_weather_history(
        meta["river_latitude"],
        meta["river_longitude"],
        flow_df["date"].min(),
        flow_df["date"].max(),
    )
    feature_df = build_feature_table(flow_df, weather_df, meta)

    valid_rows = feature_df.dropna(subset=["flow_lag14"])
    if valid_rows.empty:
        raise ValueError("Not enough historical flow data. Need at least 14 valid previous days.")

    return valid_rows.iloc[-1], feature_df


def predict_new_usgs_river(site_id, selected_date=None):
    latest, feature_df = _latest_feature_row(site_id, selected_date=selected_date)
    X_df = _build_model_input(latest)

    preds = []
    for h in range(1, int(HORIZON) + 1):
        pred_log = MODELS[h].predict(X_df)[0]
        pred = np.expm1(pred_log)
        pred = max(float(pred), 0.0)
        preds.append(pred)

    forecast_dates = [pd.to_datetime(latest["date"]) + timedelta(days=i) for i in range(1, int(HORIZON) + 1)]
    forecast_df = pd.DataFrame({
        "date": [date.strftime("%Y-%m-%d") for date in forecast_dates],
        "predicted_flow": preds,
    })
    forecast_df.attrs["latest_input_date"] = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")
    forecast_df.attrs["latest_features"] = latest.to_dict()
    forecast_df.attrs["feature_row_count"] = len(feature_df)

    return forecast_df


def _print_forecast(site_id, forecast_df):
    print(f"\nUSGS Site: {site_id}")
    latest_date = forecast_df.attrs.get("latest_input_date", "unknown")
    print(f"Latest input date: {latest_date}")
    print("Forecast:")
    for _, row in forecast_df.iterrows():
        print(f"{row['date']}: {row['predicted_flow']:.2f}")


if __name__ == "__main__":
    site = input("Enter USGS site ID: ").strip()
    forecast_df = predict_new_usgs_river(site)
    _print_forecast(site, forecast_df)
    print(forecast_df)
