import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# =========================
# Load trained artifacts
# =========================
bundle = joblib.load("lightgbm_direct_recursive_regression_bundle.pkl")
df_train = pd.read_csv("df_model.csv")

FEATURES = bundle["features"]
MODELS = bundle["direct_models"]
HORIZON = bundle.get("horizon", 10)


# =========================
# 1) Get USGS site metadata
# =========================
def fetch_usgs_site_metadata(site_id):
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": str(site_id).zfill(8),
        "siteOutput": "expanded"
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    lines = [line for line in r.text.splitlines() if not line.startswith("#")]
    text = "\n".join(lines)

    from io import StringIO
    df = pd.read_csv(StringIO(text), sep="\t")

    # second row often contains data types, remove it
    if len(df) > 1:
        df = df.iloc[1:].copy()

    if df.empty:
        raise ValueError("No USGS site metadata found.")

    row = df.iloc[0]

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


# =========================
# 2) Get USGS flow history
# =========================
def fetch_usgs_flow_history(site_id, days=90):
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
        raise ValueError("No USGS flow data found for this site.")

    values = series[0]["values"][0]["value"]
    df = pd.DataFrame(values)

    df["date"] = pd.to_datetime(df["dateTime"]).dt.tz_localize(None)
    df["flow"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[["date", "flow"]].dropna().sort_values("date")
    return df


# =========================
# 3) Get weather/rain history
# =========================
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
            "et0_fao_evapotranspiration"
        ]),
        "timezone": "auto"
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("No weather data found.")

    w = pd.DataFrame({"date": pd.to_datetime(daily["time"])})

    # Match your model feature names
    w["rain"] = pd.to_numeric(daily.get("rain_sum", daily.get("precipitation_sum")), errors="coerce")
    w["weather_temperature_2m_mean"] = pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce")
    w["weather_temperature_2m_max"] = pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce")
    w["weather_temperature_2m_min"] = pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce")
    w["weather_precipitation_hours"] = pd.to_numeric(daily.get("precipitation_hours"), errors="coerce")
    w["weather_wind_speed_10m_max"] = pd.to_numeric(daily.get("wind_speed_10m_max"), errors="coerce")
    w["weather_wind_gusts_10m_max"] = pd.to_numeric(daily.get("wind_gusts_10m_max"), errors="coerce")
    w["weather_et0_fao_evapotranspiration"] = pd.to_numeric(daily.get("et0_fao_evapotranspiration"), errors="coerce")

    # Not requested from Open-Meteo daily above; fill later by medians.
    w["weather_relative_humidity_2m_mean"] = np.nan
    w["weather_surface_pressure_mean"] = np.nan

    return w


# =========================
# 4) Feature engineering
# =========================
def build_feature_table(flow_df, weather_df, meta):
    df = flow_df.merge(weather_df, on="date", how="left")
    df = df.sort_values("date").copy()

    # static metadata
    for k, v in meta.items():
        df[k] = v

    # flow/rain lags
    for lag in range(1, 15):
        df[f"flow_lag{lag}"] = df["flow"].shift(lag)
        df[f"rain_lag{lag}"] = df["rain"].shift(lag)

    # rolling features
    df["flow_3d_avg"] = df["flow"].rolling(3).mean()
    df["flow_7d_avg"] = df["flow"].rolling(7).mean()
    df["flow_14d_avg"] = df["flow"].rolling(14).mean()

    df["rain_3d"] = df["rain"].rolling(3).sum()
    df["rain_7d"] = df["rain"].rolling(7).sum()
    df["rain_14d"] = df["rain"].rolling(14).sum()

    # dynamics
    df["flow_diff"] = df["flow"] - df["flow_lag1"]
    df["flow_acc"] = df["flow_diff"] - (df["flow_lag1"] - df["flow_lag2"])

    # dry streak
    dry_streaks = []
    count = 0
    for r in df["rain"].fillna(0):
        if r == 0:
            count += 1
        else:
            count = 0
        dry_streaks.append(count)
    df["dry_streak"] = dry_streaks

    # seasonality
    month = df["date"].dt.month
    doy = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # interactions
    df["rain_flow_interaction"] = df["rain_3d"] * df["flow_lag1"]
    df["rain_flow_ratio"] = df["rain_3d"] / (df["flow_lag1"] + 1)
    df["rain_per_area"] = df["rain_3d"] / (df["river_drainage_area_sqmi"] + 1e-6)

    return df


# =========================
# 5) Predict
# =========================
def predict_new_usgs_river(site_id):
    meta = fetch_usgs_site_metadata(site_id)
    flow_df = fetch_usgs_flow_history(site_id, days=120)

    start_date = flow_df["date"].min()
    end_date = flow_df["date"].max()

    weather_df = fetch_weather_history(
        meta["river_latitude"],
        meta["river_longitude"],
        start_date,
        end_date
    )

    feature_df = build_feature_table(flow_df, weather_df, meta)

    latest = feature_df.dropna(subset=["flow_lag14"]).iloc[-1]

    X = latest.reindex(FEATURES)
    X = pd.to_numeric(X, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    existing = [c for c in FEATURES if c in df_train.columns]
    medians = df_train[existing].median(numeric_only=True)

    for f in FEATURES:
        if f not in medians.index:
            medians[f] = 0.0

    X = X.fillna(medians.reindex(FEATURES).fillna(0.0))
    X_df = pd.DataFrame([X], columns=FEATURES)

    preds = []
    for h in range(1, HORIZON + 1):
        pred_log = MODELS[h].predict(X_df)[0]
        pred = np.expm1(pred_log)
        preds.append(max(float(pred), 0.0))

    forecast_dates = [
        (latest["date"] + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, HORIZON + 1)
    ]

    print("\nUSGS Site:", site_id)
    print("Latest input date:", latest["date"].strftime("%Y-%m-%d"))
    print("Warning: new river prediction uses engineered USGS/weather features and filled missing values with training medians.")
    print("\nForecast:")
    for d, p in zip(forecast_dates, preds):
        print(d, round(p, 2))


# =========================
# Run test
# =========================
if __name__ == "__main__":
    site = input("Enter USGS site ID: ").strip()
    predict_new_usgs_river(site)