# Sales and Customer Insights Web Application

A full-stack retail analytics platform built with Python Flask, integrating machine learning models for sales forecasting, customer segmentation, inventory simulation, and marketing ROI analysis. Developed as a final year project for the MSc in Big Data Management and Analytics at Griffith College Dublin.

---

## Live Demo

https://saleswebapp-1hki.onrender.com/)

---

## Screenshots

> Add screenshots of your dashboard here once deployed

---

## What It Does

This application transforms raw retail sales data into actionable business insights through an interactive, role-based web interface. It is designed for retail managers and analysts who need to make data-driven decisions without requiring enterprise-level software or advanced technical skills.

**Core features:**

- **Sales Forecasting** — Overall and sub-category level forecasting using ARIMA/SARIMAX and Facebook Prophet, with side-by-side model comparison and backtesting (MAE, RMSE, MAPE)
- **Customer Segmentation** — K-Means clustering to identify customer groups (Loyal, New, At Risk) based on purchasing behaviour
- **Inventory Simulation** — Monte Carlo simulation with configurable lead time, service level, and safety stock to estimate Reorder Points (ROP) and stockout probability
- **Marketing ROI Analysis** — Region and sub-category level ROI calculation with interactive filtering and chart export
- **Discount Impact Analysis** — Scatter analysis with optional trendline (linear/quadratic), binning, and 3-band Simple Moving Average
- **KPI Tracker** — Monthly sales vs target with automatic target generation (rolling average + uplift) or CSV upload
- **Interactive Sales Analysis** — Sub-category trend explorer with YoY comparison and smoothing
- **Role-Based Access Control** — Admin, Manager, and Analyst roles with separate dashboards and feature access

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask |
| Frontend | Bootstrap 5, Bootstrap Icons |
| Machine Learning | scikit-learn (K-Means), statsmodels (ARIMA/SARIMAX), Prophet |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib |
| Database | SQLite (via Flask-Login) |
| Authentication | Flask-Login, Werkzeug password hashing |

---

## Project Structure

```
sales-customer-insights/
│
├── app.py                          # Main Flask application
├── superstore_extended.csv         # Extended Superstore dataset (2014–2025)
├── sales_model.pkl                 # Trained regression model
├── sales_forecast_model_monthly.pkl
├── sales_forecast_model_yearly.pkl
├── segmented_customers.csv         # K-Means segmentation output
├── app_users.db                    # SQLite users database
├── requirements.txt
├── logs/
│   └── app.log
└── static/
    └── favicon.ico
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/sales-customer-insights.git
cd sales-customer-insights
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

**Default login credentials:**
```
Username: admin1
Password: pass123
```

---

## Requirements

```
flask
flask-login
werkzeug
pandas
numpy
matplotlib
scikit-learn
statsmodels
prophet
scipy
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## Key Technical Highlights

- **Monte Carlo Inventory Simulation** — Simulates 200 demand paths using historical sales as a proxy, computing Reorder Point using the formula: `ROP = μD × L + z × σD × √L`
- **Dual Forecasting Models** — ARIMA/SARIMAX with log-transformation for variance stabilisation vs Prophet with automatic seasonality detection; both backtested on hold-out periods
- **Regression-Based Marketing Spend Predictor** — Trained model predicts expected sales from marketing budget, discount rate, and region (one-hot encoded)
- **Role-Based Access Control** — Three user roles with route-level enforcement via custom `roles_required` decorator
- **Data Export** — All charts and tables exportable as PNG and CSV directly from the UI

---

## Dataset

Built on the **Superstore dataset** extended with synthetic data from 2019 to 2025, simulating realistic sales trends, seasonal patterns, regional variation, and marketing spend figures.

---

## Author

**Surya Sri Sundara**  
Postgraduate Diploma in Big Data Management and Analytics  
Griffith College Dublin, 2025  

LinkedIn: https://www.linkedin.com/in/surya-sri-sundara-4ab45b331/  
Email: suryasrisundara549@gmail.com

---

## Acknowledgements

Supervised by John Hannon, Griffith College Dublin.  
Dataset: Superstore (Kaggle) extended with synthetic data.
