# Solar Flare Prediction using Machine Learning

## Overview

Solar flares are sudden bursts of energy released from the sun that can disrupt GPS accuracy, satellite operations, radio communication, and power grids on Earth. Real-world examples — the 1989 Hydro-Québec blackout, the ~$500B agricultural losses from the May 2024 geomagnetic storm, and the 2022 Starlink satellite re-entries — show that although significant flares are relatively rare, their impact can be severe.

This project explores how much **predictive signal exists in publicly available daily solar indices alone**, and builds a machine learning pipeline to forecast the likelihood and scale of significant solar flare activity (M-class and X-class) using historical solar activity data.

> **Note:** This project is not intended to replace NOAA's operational forecasts (which use satellite imagery, magnetograms, and physics-based models). It is an academic exploration of how far a reproducible ML pipeline can go using only tabular daily indices.

---

## Objective

- Predict whether a significant solar flare (M-class or X-class) is likely to occur on the **next day**, using solar indices from the current/past day(s).
- Predict the approximate **count** of such flares, conditional on occurrence.
- Study the sun's **long-term activity pattern** through monthly and yearly trend analysis, since solar activity follows an approximately 11-year cycle.
- Validate the trained model against a freshly downloaded, genuinely unseen portion of the dataset at the end of the project.

---

## Dataset

- **Source**: [NOAA Space Weather Prediction Center — Daily Solar Data](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/annual_reports/daily_solar_indices_summaries/daily_solar_data/)
- **Range used**: 1999 – present (column structure is consistent from 1999 onward; earlier years have a different format).
- **Format**: Yearly fixed-width text files (`YYYY_DSD.txt`).

### Raw Columns

| Column | Description |
|---|---|
| Date | Year, Month, Day |
| Radio Flux (10.7cm) | Solar radio emission — a proxy for overall solar activity |
| SESC Sunspot Number | Daily sunspot count |
| Sunspot Area (10E-6 Hemisphere) | Total area covered by sunspots |
| New Regions | Count of newly formed active regions |
| Stanford Solar Mean Field | Solar magnetic field strength (contains missing values: `*` or `-999`) |
| X-Ray Bkgd Flux | Background X-ray class (e.g. B6.3, C1.0, M2.5) |
| Flares: C, M, X | Daily count of C/M/X-class X-ray flares |
| Flares: S, 1, 2, 3 | Daily count of Optical flares by importance (Subflare, Small, Medium, Large) |

### Flare Classification Reference

**X-ray classes** (weakest → strongest): `A → B → C → M → X` — each class is ~10x more intense than the previous. M and X classes are the primary target since they correspond to significant, Earth-impacting events.

**Optical classes** (weakest → strongest): `S → 1 → 2 → 3` — an older, visible-light-based classification, used here only as a supporting input feature, not as a prediction target.

---

## Approach

### 1. Data Pipeline
1. Scrape and combine all yearly `_DSD.txt` files (1999–present) into one chronological dataset.
2. Clean missing values (standardize `*` / `-999` → NaN, forward-fill), convert X-ray Bkgd Flux to numeric.
3. Shift target columns by one day (Day T's data → Day T+1's outcome) to avoid data leakage.
4. Engineer rolling-window features (3-day, 7-day, 27-day averages) on key indices — the 27-day window aligns with the sun's rotation period.

### 2. Modeling — Two-Stage Design
- **Stage 1 (Classification)**: Predict whether an M/X-class flare will occur the next day (binary).
- **Stage 2 (Regression)**: For days predicted positive in Stage 1, predict the expected count of M-class and X-class flares.

**Models used**: Logistic Regression (baseline), Random Forest, XGBoost; optional LSTM/sequence model for comparison.

### 3. Evaluation
- **Classification**: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix (Recall prioritized — missing a flare is costlier than a false alarm).
- **Regression**: MAE, RMSE.
- **Split strategy**: Chronological (time-based) train/validation/test split — never random shuffling, since this is time-series data.
- **Class imbalance**: Addressed via class weights / SMOTE, since M/X-class flares are rare events.

### 4. Final Validation
At the end of the project, a fresh copy of the dataset is manually re-downloaded from NOAA. The portion beyond the original download cutoff is used as a genuinely unseen test set to check real-world generalization — this is a one-time manual step, not a live/automated pipeline.

---

## Project Structure

```
├── data/
│   ├── raw/                # Original downloaded _DSD.txt files
│   └── processed/          # Cleaned, feature-engineered CSV
├── notebooks/               # EDA, feature engineering, modeling notebooks
├── src/
│   ├── data_collection.py   # Scraping & parsing raw NOAA files
│   ├── preprocessing.py     # Cleaning, missing value handling, flux conversion
│   ├── feature_engineering.py  # Row shifting, rolling windows, targets
│   ├── train_stage1.py       # Classification model training
│   ├── train_stage2.py       # Regression model training
│   └── evaluate.py           # Evaluation metrics & baseline comparison
├── reports/
│   └── figures/               # EDA & trend visualizations
├── README.md
└── requirements.txt
```

---

## Team

| Member | Role |
|---|---|
| **Ankit** | Team Lead — Modeling (Stage 1 & 2), evaluation, coordination |
| **Apoorva** | Feature engineering, class imbalance handling, feature importance analysis |
| **Anjali** | Data collection, cleaning, EDA & trend analysis |

---

## Limitations & Future Scope

- Uses only tabular daily indices — does not include solar wind, coronal mass ejection (CME), or magnetogram imaging data, which operational forecasting systems rely on heavily.
- Daily granularity only — does not predict *when* within a day a flare might occur.
- Rare-event nature of M/X-class flares makes exact count prediction (Stage 2) inherently harder than occurrence prediction (Stage 1).
- Future work could incorporate higher-resolution GOES X-ray flux data or solar wind parameters for improved accuracy.

---

## Disclaimer

This project is an academic exercise and is **not a substitute for official space weather forecasts**. For real-time, authoritative space weather alerts, refer to [NOAA's Space Weather Prediction Center](https://www.swpc.noaa.gov/).
