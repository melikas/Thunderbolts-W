# Thunderbolts-W
Decode the invisible activity and location recognition challenge in care facility


Papers: 
1. Models for Time Series Forecasting: https://arxiv.org/abs/2401.13912


# BLE-Based Location Prediction

## Overview
This project implements a BLE-based location prediction pipeline using RSSI and transmission power measurements. The objective is to infer the most probable location over time by processing raw BLE scan data, estimating distances to beacons, and applying window-based aggregation.

---

## Pipeline

### Step 1: Data Merging
All relevant data sources (e.g., BLE scans, RSSI values, transmission power) are merged into a single unified dataset to ensure temporal alignment and consistency.

---

### Step 2: Data Cleaning
The dataset is cleaned to improve reliability and robustness:
- Out-of-range transmission power values are replaced with valid in-range values.
- Missing values are handled using appropriate imputation techniques.

---

### Step 3: Distance Estimation
For each data point (row), the distance \( d \) to each BLE beacon is estimated using RSSI and transmission power, based on standard RSSI-based distance estimation models.

Example formula:  d = 10 ^ ((TxPower - RSSI) / (10 * n))
where `n` is the path-loss exponent.

---

### Step 4: Window-Based Location Inference
Location is inferred using fixed-length sliding windows:
- A predefined window length is used (e.g., 10 seconds, approximately 600 data points).
- Within each window, the dominant BLE beacon is identified using aggregated RSSI and power values.
- The window is assigned the location corresponding to the dominant BLE beacon.
- Mean RSSI and mean transmission power are computed for each window.
- This procedure is repeated sequentially until the end of the dataset.

(## Possible Improvements
- Probabilistic and Bayesian location inference
- Temporal smoothing across windows
- Multi-beacon fusion and confidence estimation)

---

## Output
For each time window, the pipeline outputs:
- Estimated location (BLE beacon ID)
- Mean RSSI
- Mean transmission power
- Time window index
- Maybe added accelerometer

---

## Configuration
- Window length can be adjusted based on the sampling rate and application requirements.
- Distance estimation parameters (e.g., path-loss exponent) can be tuned for the environment.

---
