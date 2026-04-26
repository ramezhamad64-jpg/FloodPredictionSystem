# FloodSense

FloodSense is a Flask-based machine learning web application for flood risk classification and 10-day streamflow forecasting.

The project uses processed river and weather datasets, saved trained machine learning models, and live API data for new river prediction. For a new USGS river, the user enters a USGS site ID, then the app fetches recent streamflow and weather data, creates the same features used during training, and predicts future streamflow using the saved LightGBM model.

---

## Project Overview

FloodSense supports three main tasks:

1. Flood classification using a trained Random Forest classifier.
2. Streamflow forecasting for the next 10 days using regression models.
3. New USGS river prediction using live data from USGS and Open-Meteo APIs.

The main goal of this project is to provide an end-to-end machine learning system that can classify flood risk and forecast future river discharge using hydrological and weather-based features.

---

## Data Sources

The project uses real-world environmental data from the following sources:

### USGS Water Services

USGS data is used to collect:

- River site metadata
- River location
- Drainage area
- Elevation
- Historical and recent streamflow measurements

### Open-Meteo Archive API

Open-Meteo data is used to collect:

- Rainfall
- Temperature
- Wind features
- Evapotranspiration
- Other weather-related variables

These sources allow the system to combine river behavior with weather conditions for better flood and streamflow prediction.

---

## Models Used

The project includes several machine learning models and saved artifacts.

---

### 1. Flood Classification

A Random Forest classifier was used for flood risk classification.

The goal of the classifier is to predict whether a given river condition represents a flood risk.

Main classification artifacts:

```text
models/final_low_leakage_is_flood_classifier.pkl
models/final_low_leakage_feature_medians.pkl
models/final_low_leakage_thresholds.csv
models/final_low_leakage_summary.csv
models/final_low_leakage_feature_importance.csv
The classification dataset is:

data/processed/classification_dataset.csv

It contains the final processed classification features and the target column:

is_flood

The classification model prioritizes recall because flood prediction is safety-critical. Missing a flood is more dangerous than producing a false alarm. Therefore, threshold tuning was used to improve flood detection and reduce the chance of missing dangerous events.

2. Regression Baseline

Linear Regression was used as an initial baseline for streamflow forecasting.

The baseline helped compare simple linear behavior against more advanced nonlinear models. It also showed why stronger models were needed, since streamflow behavior depends on nonlinear hydrological patterns, rainfall accumulation, seasonality, and previous flow conditions.

3. LightGBM Streamflow Forecasting

LightGBM was used for direct and recursive 10-day streamflow forecasting.

Main LightGBM artifact:

models/lightgbm_direct_recursive_regression_bundle.pkl

This bundle contains:

direct_models
recursive_model
features
target_cols
horizon
max_lag

The deployed new-river prediction mainly uses the saved LightGBM direct models.

The regression dataset is:

data/processed/regression_dataset.csv

It contains:

river_id
date
59 LightGBM features
flow_target_day1
flow_target_day2
flow_target_day3
flow_target_day4
flow_target_day5
flow_target_day6
flow_target_day7
flow_target_day8
flow_target_day9
flow_target_day10

The LightGBM model predicts future streamflow values for multiple forecast horizons.

Direct forecasting trains separate models for each future day. Recursive forecasting uses one model step-by-step, where earlier predictions are used as inputs for later predictions.

4. LSTM Experiments

LSTM models were tested for sequence-based streamflow forecasting.

Main LSTM artifacts:

models/peak_lstm_direct_recursive_bundle.pkl
models/best_peak_lstm_direct_10day.keras
models/best_peak_lstm_recursive_day1.keras
models/peak_lstm_feature_scaler.pkl
models/peak_lstm_feature_medians.pkl
models/peak_lstm_direct_target_scaler.pkl
models/peak_lstm_recursive_target_scaler.pkl

The LSTM models are included as experiment artifacts, while the main deployed forecasting path uses LightGBM.

Important Concept

The processed datasets are already cleaned and feature-engineered.

Do not rebuild the datasets during app prediction.

Do not retrain models during app prediction.

Do not overwrite saved model files.

Feature engineering is only performed when predicting a new live USGS river, because the app must convert raw live API data into the exact feature format expected by the trained model.

For live prediction, the app loads the required feature list directly from the LightGBM bundle:

FEATURES = bundle["features"]

This ensures the live input matches the trained LightGBM model.