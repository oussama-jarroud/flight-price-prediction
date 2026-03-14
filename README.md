# ✈️ Flight Price Prediction — CMN → MAD

> Web Scraping · Data Science · Machine Learning  
> **Master SDIA** — Projet de Fin de Module · UMI FSM · 2025/2026

---

## Overview

End-to-end data science project that scrapes real flight prices from Skyscanner (Casablanca → Madrid), builds a clean dataset, performs exploratory analysis, and trains ML models to predict prices and forecast price drops.

**Best results:**
- Regression (price prediction): **R² = 0.496**, MAE = 66€ — Random Forest
- Classification (price drop in 3 days): **Accuracy = 90.6%** — Random Forest

---

## Project Structure

```
flight-price-prediction/
│
├── data/
│   ├── flights_raw.csv          # Raw scraped data (300 rows)
│   └── flights_clean.csv        # Cleaned dataset (294 rows)
│
├── figures/
│   ├── 01_distribution_prix.png
│   ├── 02_evolution_prix.png
│   ├── 03_prix_par_compagnie.png
│   ├── 04_prix_par_jour.png
│   ├── 05_prix_escales.png
│   ├── 06_correlation.png
│   ├── 07_serie_temporelle_min.png
│   ├── 08_nb_vols_par_date.png
│   ├── 09_comparaison_regression.png
│   ├── 10_predit_vs_reel.png
│   ├── 11_feature_importance_reg.png
│   ├── 12_confusion_matrix.png
│   └── 13_feature_importance_clf.png
│
├── models/
│   ├── rf_regressor.pkl         # Random Forest regressor
│   ├── rf_classifier.pkl        # Random Forest classifier
│   ├── scaler.pkl
│   └── le_airline.pkl
│
├── scraper/
│   ├── skyscanner_scraper_v3.py # Main Playwright scraper
│   └── skyscanner_debug.py      # Diagnostic tool
│
├── dashboard/
│   └── flight_dashboard.html    # Interactive prototype (no install needed)
│
├── phase3_nettoyage.py          # Data cleaning
├── phase4_eda.py                # Exploratory analysis
├── phase5_ml.py                 # ML modeling
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Skyscanner (Playwright)
        ↓
  flights_raw.csv       ← phase 2 : scraping
        ↓
 flights_clean.csv      ← phase 3 : cleaning
        ↓
   8 EDA figures        ← phase 4 : analysis
        ↓
  ML models (.pkl)      ← phase 5 : modeling
        ↓
dashboard prototype     ← phase 6 : recommendation
```

---

## Key Results

| Model | Task | Metric | Score |
|---|---|---|---|
| Linear Regression | Price prediction | R² | 0.159 |
| Decision Tree | Price prediction | R² | 0.206 |
| **Random Forest** | **Price prediction** | **R²** | **0.496** |
| Gradient Boosting | Price prediction | R² | 0.450 |
| Logistic Regression | Drop classification | Accuracy | 71.7% |
| **Random Forest** | **Drop classification** | **Accuracy** | **90.6%** |

---

## Dataset

| Field | Description |
|---|---|
| `date` | Departure date |
| `price` | Price in EUR |
| `airline` | Carrier name (9 airlines) |
| `stops` | Number of stops |
| `duration_min` | Total flight duration |
| `departure` / `arrival` | Datetime |
| `days_until_dep` | Days between scraping and departure |
| `is_direct` | Binary (84% direct flights) |
| `day_of_week` | Day name |

**Stats:** 294 flights · 30 days · 9 airlines · price range: 55€ – 781€ · avg: 217€

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/flight-price-prediction
cd flight-price-prediction
pip install -r requirements.txt
playwright install chromium
```

---

## Usage

```bash
# 1. Scrape data
python scraper/skyscanner_scraper_v3.py

# 2. Clean
python phase3_nettoyage.py

# 3. EDA
python phase4_eda.py

# 4. Train models
python phase5_ml.py

# 5. Open dashboard (no server needed)
open dashboard/flight_dashboard.html
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Playwright](https://img.shields.io/badge/Playwright-scraping-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-data-purple)
![Chart.js](https://img.shields.io/badge/Chart.js-dashboard-red)

- **Scraping**: Playwright, Requests
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Dashboard**: Vanilla HTML/CSS/JS

---

## Anti-bot Strategy

Skyscanner uses **PerimeterX** protection. The scraper handles this via:
- Real browser session cookies injection (`_pxvid`, `_px3`)
- Human-like behavior simulation (random delays, mouse movement)
- Playwright headless with stealth patches

---

## Limitations

- Dataset limited to 30 days and one route (CMN→MAD)
- R² = 0.496 would improve with 3+ months of historical data
- Cookie-based authentication requires manual refresh every few hours

---

## Author

**[Your Name]** — Master SDIA 2025/2026, UMI FSM  
Supervised by: [Professor name]

---

*Academic project — data collected for educational purposes only.*
