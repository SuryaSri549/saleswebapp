from flask import Flask, render_template, request, render_template_string, redirect, url_for, flash, g
import sqlite3
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.cluster import KMeans
import datetime
import secrets, string
import os

import time
import sys
import json
import logging
from logging.handlers import RotatingFileHandler


# --- Where to write logs (next to app.py) ---
BASEDIR = os.path.abspath(os.path.dirname(__file__))

DATA_CSV = os.path.join(BASEDIR, "superstore_extended.csv")
USERS_DB = os.path.join(BASEDIR, "app_users.db")   # for SQLite users DB
MODEL_MONTHLY    = os.path.join(BASEDIR, "sales_forecast_model_monthly.pkl")
MODEL_YEARLY     = os.path.join(BASEDIR, "sales_forecast_model_yearly.pkl")
SUPERSTORE_DB = os.path.join(BASEDIR, "superstore.db")
SEGMENTED_CSV = os.path.join(BASEDIR, "segmented_customers.csv")
MODEL_PATH    = os.path.join(BASEDIR, "sales_model.pkl")
LOG_DIR  = os.path.join(BASEDIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


# ensure file exists so you can tail immediately
open(LOG_FILE, "a", encoding="utf-8").close()

# --- Handlers/format ---
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s %(funcName)s:%(lineno)d — %(message)s"
)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Force global logging config (overrides any prior handlers/config)
logging.basicConfig(handlers=[file_handler, console_handler], level=logging.INFO, force=True)


# --- Root logger (captures most libraries too) ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# avoid double-adding if auto-reloads
if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
    root_logger.addHandler(file_handler)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
           for h in root_logger.handlers):
    root_logger.addHandler(console_handler)

# Log *all* SQL executed by sqlite3 (Python 3.12+ only)
if hasattr(sqlite3, "set_trace_callback"):
    try:
        def _sqlite_trace(statement):
            logging.getLogger("sql").info("SQL: %s", statement)
        sqlite3.set_trace_callback(_sqlite_trace)
    except Exception as e:
        logging.getLogger("sql").warning("Could not enable sqlite trace: %s", e)
else:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    logging.getLogger("sql").warning(
        "sqlite3.set_trace_callback not available in Python %s (requires 3.12+)", ver
    )

# --- Redaction helpers for sensitive fields ---
_SENSITIVE_KEYS = {"password", "passwd", "pwd", "secret", "token", "access_token", "authorization"}

def _redact_mapping(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        out[k] = "***REDACTED***" if k and k.lower() in _SENSITIVE_KEYS else v
    return out

def _safe_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
try:
    from product_analysis import (
        generate_region_chart_base64,
        get_monthly_sales_chart,
        get_top_products_chart,
        generate_filtered_chart
    )
except Exception:
    def generate_region_chart_base64(*a, **k): return None
    def get_monthly_sales_chart(*a, **k): return None
    def get_top_products_chart(*a, **k): return None
    def generate_filtered_chart(*a, **k): return None
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    current_user, logout_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException
from functools import wraps
from flask import abort

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

# --- Debug routes (add right after app is created) ---
@app.get("/health")
def _health():
    return "OK from " + __file__

@app.get("/routes")
def _routes():
    return "<pre>" + "\n".join(sorted(r.rule for r in app.url_map.iter_rules())) + "</pre>"

@app.get("/log-path", strict_slashes=False)
def _log_path():
    import os
    return f"LOG_FILE: {LOG_FILE}<br>exists: {os.path.exists(LOG_FILE)}"


# Attach logging handler to app
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.propagate = False   # prevent duplicate lines
app.logger.info("🚀 App started. Logging to: %s", LOG_FILE)



# --- Global request/response logging ---
@app.route("/log-test")
def log_test():
    app.logger.info("Manual test log from /log-test route")
    return "Logged a test line. Check app.log."

def _dump_routes_now():
    rules = sorted(r.rule for r in app.url_map.iter_rules())
    text = "Registered routes:\n" + "\n".join(rules)
    app.logger.info(text)
    print(text)


@app.before_request
def _log_request():
    g._start_ts = time.perf_counter()

    # Collect inputs
    args = request.args.to_dict(flat=False) or {}
    form = request.form.to_dict(flat=False) or {}
    try:
        body_json = request.get_json(silent=True)
    except Exception:
        body_json = None

    # Redact
    form_flat = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in form.items()}
    args_red = {k: ["***REDACTED***" if k.lower() in _SENSITIVE_KEYS else v for v in vs] for k, vs in args.items()}
    form_red = _redact_mapping(form_flat)
    json_red = _redact_mapping(body_json) if isinstance(body_json, dict) else body_json

    # Try user info if flask-login is present
    uname = ""
    urole = ""
    try:
        from flask_login import current_user
        if getattr(current_user, "is_authenticated", False):
            uname = getattr(current_user, "username", "")
            urole = getattr(current_user, "role", "")
    except Exception:
        pass

    app.logger.info(
        "➡️ %s %s | ip=%s | user=%s role=%s | args=%s | form=%s | json=%s",
        request.method,
        request.path,
        request.headers.get("X-Forwarded-For") or request.remote_addr,
        uname, urole,
        _safe_json(args_red),
        _safe_json(form_red),
        _safe_json(json_red),
    )

@app.after_request
def _log_response(resp):
    dur_ms = (time.perf_counter() - getattr(g, "_start_ts", time.perf_counter())) * 1000.0
    app.logger.info(
        "⬅️ %s %s — %s in %.1f ms — len=%s",
        request.method,
        request.path,
        resp.status,
        dur_ms,
        resp.content_length,
    )
    return resp

@app.errorhandler(Exception)
def _log_exception(e):
    # Let Flask handle HTTP errors (404, 405, etc.) correctly
    if isinstance(e, HTTPException):
        app.logger.warning("HTTP %s at %s %s", e.code, request.method, request.path)
        return e

    # Everything else = 500 + stack trace
    app.logger.exception("💥 Unhandled exception at %s %s", request.method, request.path)
    return ("<h3>Internal Server Error</h3><p>Check logs/app.log</p>", 500)




# ---- Shared helper: plot -> base64 PNG ----
def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return out
def find_col(df, wanted, aliases=None):
    """
    Return an existing column name from df matching 'wanted' or any alias (case-insensitive).
    Example: find_col(df, 'Profit', ['profit', 'Total Profit'])
    """
    if aliases is None:
        aliases = []
    wanted_all = [wanted] + aliases
    # normalize once
    norm = {c.lower(): c for c in df.columns}
    for name in wanted_all:
        if name.lower() in norm:
            return norm[name.lower()]
    return None
def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return out
def build_daily_demand(df, subcat=None, region=None):
    """
    Returns a daily demand pandas Series (index=day, values=demand units).
    Uses Sales as 'units' proxy; resamples to daily sum; fills missing days with 0.
    """
    if "Order Date" not in df.columns or "Sales" not in df.columns:
        raise ValueError("CSV must contain 'Order Date' and 'Sales' columns.")

    d = df.copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
    d = d.dropna(subset=["Order Date", "Sales"])
    if subcat and "Sub-Category" in d.columns:
        d = d[d["Sub-Category"] == subcat]
    if region and "Region" in d.columns:
        d = d[d["Region"] == region]

    if d.empty:
        return pd.Series(dtype=float)

    daily = (d.set_index("Order Date")["Sales"]
               .astype(float)
               .resample("D").sum()
               .fillna(0.0))
    return daily


def reorder_point_normal(mean_d, std_d, lead_time_days, service_level):
    """
    ROP = mean_d * L + z * std_d * sqrt(L)
    service_level ∈ (0,1). If std=0 or very short history, safety stock becomes 0.
    """
    from scipy.stats import norm
    z = norm.ppf(min(max(service_level, 0.50), 0.999))  # clamp
    mu_L = mean_d * lead_time_days
    sigma_L = std_d * (lead_time_days ** 0.5)
    return max(0.0, mu_L + z * sigma_L)


def simulate_inventory_mc(
    demand_series,
    initial_stock=1000.0,
    lead_time_days=7,
    review_period_days=1,
    service_level=0.95,
    order_up_to_multiplier=1.5,
    runs=200,
    horizon_days=90,
):
    """
    Monte Carlo inventory sim (continuous review approximated daily).
    - Demand is sampled by bootstrapping from historical daily demand distribution.
    - Policy: reorder when on-hand + on-order - backorder < ROP, order up to (ROP*order_up_to_multiplier).
    - Backorders allowed (track stockouts).
    Returns: dict with KPIs, per-run inventory paths, and per-day percentiles.
    """
    if len(demand_series) < 14:
        raise ValueError("Not enough history to simulate (need at least ~14 days).")

    mean_d = demand_series.mean()
    std_d = demand_series.std(ddof=1) if demand_series.std(ddof=1) > 0 else 0.0
    rop = reorder_point_normal(mean_d, std_d, lead_time_days, service_level)
    order_up_to = max(rop * order_up_to_multiplier, rop + mean_d * lead_time_days)

    hist = demand_series.values
    n_hist = len(hist)

    # Storage for runs
    paths = []      # list of DataFrames per run
    stockout_flags = []

    rng = np.random.default_rng(seed=42)

    for r in range(runs):
        days = pd.date_range(start=pd.Timestamp.today().normalize(), periods=horizon_days, freq="D")

        on_hand = initial_stock
        on_order = []  # list of tuples (arrival_day_index, qty)
        backorder = 0.0

        inv = []
        ords = []
        bkgs = []
        outs = []

        for t, day in enumerate(days):
            # Receive any orders arriving today
            arrivals = [qty for (arrive_t, qty) in on_order if arrive_t == t]
            if arrivals:
                on_hand += sum(arrivals)
            # keep only future
            on_order = [(arrive_t, qty) for (arrive_t, qty) in on_order if arrive_t > t]

            # Sample today's demand (bootstrap)
            d_t = float(hist[rng.integers(0, n_hist)])
            # Serve demand
            available = max(0.0, on_hand - backorder)
            if d_t <= available:
                # all demand met; first cover backlog then new demand
                serve_backlog = min(backorder, on_hand)
                on_hand -= serve_backlog
                backorder -= serve_backlog
                on_hand -= (d_t)
                stockout_today = 0
            else:
                # stockout occurs
                needed = d_t - available
                on_hand = max(0.0, on_hand - backorder)
                backorder = needed
                stockout_today = 1

            # Place order if it's a review day and net position below ROP
            if (t % review_period_days) == 0:
                net_pos = on_hand + sum(q for _, q in on_order) - backorder
                if net_pos < rop:
                    target = order_up_to
                    order_qty = max(0.0, target - net_pos)
                    if order_qty > 0:
                        arrival_t = t + max(1, int(lead_time_days))
                        on_order.append((arrival_t, order_qty))
                else:
                    order_qty = 0.0
            else:
                order_qty = 0.0

            inv.append(on_hand)
            ords.append(order_qty)
            bkgs.append(backorder)
            outs.append(stockout_today)

        df_run = pd.DataFrame({
            "date": days,
            "on_hand": inv,
            "order_qty": ords,
            "backorder": bkgs,
            "stockout": outs
        })
        paths.append(df_run)
        stockout_flags.append(int(any(df_run["stockout"] == 1)))

    # KPIs
    stockout_prob = np.mean(stockout_flags)
    # Fill rate approx (orders met / total demand) — we can estimate from last run
    last = paths[-1]
    # Approx demand per day as served+backorder delta; better: keep demand in loop; here estimate:
    # We'll just compute average backorder and on_hand:
    avg_on_hand = float(np.mean([p["on_hand"].mean() for p in paths]))
    avg_backorders = float(np.mean([p["backorder"].mean() for p in paths]))

    # Percentile bands across runs (on_hand)
    df_concat = pd.concat([p.set_index("date")["on_hand"].rename(i) for i, p in enumerate(paths)], axis=1)
    pct = df_concat.quantile([0.1, 0.5, 0.9], axis=1).T  # columns: 0.1, 0.5, 0.9

    return {
        "rop": rop,
        "order_up_to": order_up_to,
        "mean_demand": mean_d,
        "std_demand": std_d,
        "stockout_prob": float(stockout_prob),
        "avg_on_hand": avg_on_hand,
        "avg_backorders": avg_backorders,
        "paths": paths,
        "percentiles": pct.reset_index().rename(columns={"index": "date", 0.1: "p10", 0.5: "p50", 0.9: "p90"})
    }

login_manager = LoginManager(app)
login_manager.login_view = "login"  # where to redirect if not logged in
def get_all_users():
    with get_db() as con:
        rows = con.execute("SELECT id, username, role FROM users ORDER BY role, username").fetchall()
    return rows

def set_user_password(user_id, new_password):
    with get_db() as con:
        con.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new_password), user_id)
        )

def delete_user(user_id):
    with get_db() as con:
        con.execute("DELETE FROM users WHERE id=?", (user_id,))

# ---------- Users DB (SQLite) ----------
def get_db():
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    with get_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT CHECK(role IN ('admin','manager','analyst')) NOT NULL
            );
        """)
        # seed default admin (admin1 / pass123)
        if not con.execute("SELECT 1 FROM users WHERE username=?", ('admin1',)).fetchone():
            con.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                ('admin1', generate_password_hash('pass123'), 'admin')
            )

# initialize the users DB at startup
init_user_db()
class User(UserMixin):
    def __init__(self, id, username, password_hash, role):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    @staticmethod
    def from_row(row):
        return User(row["id"], row["username"], row["password_hash"], row["role"])

@login_manager.user_loader
def load_user(user_id):
    with get_db() as con:
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return User.from_row(row) if row else None


def roles_required(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                # flask-login will redirect to login page if you also use @login_required
                return abort(401)
            if current_user.role not in roles:
                return abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# --------------------------
# 📈 Compare ARIMA vs Prophet
# --------------------------
def compare_forecast_models(df_grouped, periods=6):
    
    # Prophet
    from prophet import Prophet
    prophet_df = df_grouped.rename(columns={"Month": "ds", "Sales": "y"})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods, freq='MS')
    forecast_prophet = m.predict(future)
    prophet_plot = forecast_prophet[['ds', 'yhat']].tail(periods)

    # ARIMA
    from statsmodels.tsa.arima.model import ARIMA
    from pandas.tseries.offsets import DateOffset
    arima_model = ARIMA(df_grouped['Sales'], order=(1, 1, 1))
    arima_fit = arima_model.fit()
    forecast_arima = arima_fit.forecast(steps=periods)

    last_date = df_grouped['Month'].max()
    arima_dates = [last_date + DateOffset(months=i) for i in range(1, periods + 1)]

    # Plot both
    plt.figure(figsize=(10, 5))
    plt.plot(df_grouped['Month'], df_grouped['Sales'], label='Historical Sales')
    plt.plot(prophet_plot['ds'], prophet_plot['yhat'], label='Prophet Forecast', linestyle='--')
    plt.plot(arima_dates, forecast_arima, label='ARIMA Forecast', linestyle='--')
    plt.title("ARIMA vs Prophet Forecast")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return chart_base64





@app.route("/", methods=["GET"])
def home():
    df = pd.read_csv(DATA_CSV)
    subcategories = sorted(df['Sub-Category'].unique())

    return render_template_string("""
    <!DOCTYPE html>
    <html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>📊 Sales & Customer Insights</title>

  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root {
      --brand: #0d6efd; /* Bootstrap primary */
      --card-radius: 14px;
    }
    body { padding: 24px; background-color: #f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,0.05); }
    .card h4 { margin-bottom: 14px; }
    .section-title { display: flex; align-items: center; gap: 8px; }
    .section-title .bi { opacity: .8; }
    .btn { border-radius: 10px; }
    .subtle { color:#6c757d; }
    footer { color:#6c757d; }
    /* Dark mode support */
    [data-bs-theme="dark"] body { background-color:#0b0f14; }
  </style>
</head>

    <body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/">
    <i class="bi bi-graph-up-arrow me-1"></i> Sales Insights
  </a>
  <div class="ms-auto d-flex align-items-center gap-2">
  {% if current_user.is_authenticated %}
    <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>

    {# Admin-only button #}
    {% if current_user.role == 'admin' %}
      <a class="btn btn-sm btn-outline-primary" href="/admin">
        <i class="bi bi-gear"></i> Admin Dashboard
      </a>
    {% endif %}

    {# Manager-only button #}
    {% if current_user.role == 'manager' %}
      <a class="btn btn-sm btn-outline-success" href="/manager-dashboard">
        <i class="bi bi-speedometer2"></i> Manager Dashboard
      </a>
    {% endif %}
                                  
    {# Analyst-only button #}
    {% if current_user.is_authenticated and (current_user.role|lower == 'analyst') %}
      <a class="btn btn-sm btn-outline-dark" href="/analyst-dashboard">
      <i class="bi bi-speedometer2"></i> Analyst Dashboard
      </a>
    {% endif %}

    <a class="btn btn-sm btn-outline-secondary" href="/logout">
      <i class="bi bi-box-arrow-right"></i> Logout
    </a>
  {% else %}
    <a class="btn btn-sm btn-primary" href="/login">
      <i class="bi bi-box-arrow-in-right"></i> Login
    </a>
  {% endif %}

  <button id="themeToggle" class="btn btn-sm btn-outline-secondary" type="button">
    <i class="bi bi-moon-stars"></i>
  </button>
</div>
</nav>

<script>
  // Simple dark-mode toggle using Bootstrap 5.3 data-bs-theme
  (function () {
    const html = document.documentElement;
    const saved = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-bs-theme', saved);
    document.getElementById('themeToggle').addEventListener('click', () => {
      const next = html.getAttribute('data-bs-theme') === 'light' ? 'dark' : 'light';
      html.setAttribute('data-bs-theme', next);
      localStorage.setItem('theme', next);
    });
  })();
</script>

<div class="container">

  <h1 class="text-center mb-4">📊 Welcome to Sales & Customer Insights Web App 🎉</h1>

  <!-- top status/login strip -->

  <!-- ROW 1 -->
  <div class="row g-4">
    <!-- LEFT: Predict + Forecasting (single card; no nested cards) -->
    <div class="col-lg-8">
      <div class="card h-100">
        <div class="card-body">
          <h4 class="section-title"><i class="bi bi-calculator"></i> Predict Sales</h4>
          <form method="post" action="/predict" class="row g-3">
            <div class="col-md-4">
              <label class="form-label">Marketing Spend ($)</label>
              <input type="number" name="marketing_spend" step="0.01" class="form-control" required>
            </div>
            <div class="col-md-4">
              <label class="form-label">Discount (%)</label>
              <input type="number" name="discount" step="0.01" class="form-control" required>
            </div>
            <div class="col-md-4">
              <label class="form-label">Region</label>
              <select name="region" class="form-select" required>
                <option value="" disabled selected>Select Region</option>
                <option value="Central">Central</option>
                <option value="East">East</option>
                <option value="South">South</option>
                <option value="West">West</option>
              </select>
            </div>
            <div class="col-12">
              <button type="submit" class="btn btn-primary mt-2">Predict Sales</button>
            </div>
          </form>

          <hr class="my-4">

          {% if current_user.is_authenticated and (current_user.role == 'analyst' or current_user.role == 'manager') %}
            <h4 class="section-title mb-3"><i class="bi bi-bar-chart-steps"></i> Forecast Future Sales</h4>

            <!-- Tabs -->
            <ul class="nav nav-tabs" id="forecastTabs" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overall-tab" data-bs-toggle="tab" data-bs-target="#overall" type="button" role="tab" aria-controls="overall" aria-selected="true">Overall</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="subcat-tab" data-bs-toggle="tab" data-bs-target="#subcat" type="button" role="tab" aria-controls="subcat" aria-selected="false">Sub-Category</button>
              </li>
            </ul>

            <div class="tab-content pt-3">
<!-- OVERALL -->
<div class="tab-pane fade show active" id="overall" role="tabpanel" aria-labelledby="overall-tab">
  <form action="/forecast" method="post" class="row row-cols-1 row-cols-md-3 g-3 align-items-end" id="overallForm">
    
    <!-- Periods -->
    <div class="col">
      <label class="form-label">Periods</label>
      <div class="input-group">
        <input type="number" name="periods" min="1" value="6" class="form-control" required>
        <span class="input-group-text" id="overallUnit">months</span>
      </div>
      <div class="form-text">How far ahead to predict.</div>
    </div>

    <!-- Forecast Type -->
    <div class="col">
      <label class="form-label">Forecast Type</label>
      <select name="forecast_type" class="form-select" id="forecastTypeOverall">
        <option value="monthly" selected>Monthly</option>
        <option value="yearly">Yearly</option>
      </select>
      <div class="form-text">Choose monthly or yearly forecast.</div>
    </div>

    <!-- Quick presets -->
    <div class="col">
      <label class="form-label d-block">Quick presets</label>
      <div class="btn-group w-100" role="group" aria-label="Quick presets">
        <button class="btn btn-outline-secondary btn-sm preset" data-val="3" type="button">3</button>
        <button class="btn btn-outline-secondary btn-sm preset" data-val="6" type="button">6</button>
        <button class="btn btn-outline-secondary btn-sm preset" data-val="12" type="button">12</button>
      </div>
      <div class="form-text text-end text-md-start">Adjust periods quickly.</div>
    </div>

    <!-- Submit (full row, right aligned) -->
    <div class="col-12 text-end">
      <button type="submit" class="btn btn-success" id="overallSubmit">
        <span class="spinner-border spinner-border-sm me-1 d-none" id="overallSpinner" aria-hidden="true"></span>
        Forecast
      </button>
    </div>
  </form>
</div>


              <!-- SUB-CATEGORY -->
              <div class="tab-pane fade" id="subcat" role="tabpanel" aria-labelledby="subcat-tab">
                <form action="/forecast-subcat" method="post" class="row g-3 align-items-end" id="subcatForm">
                  <div class="col-md-5">
                    <label class="form-label">Sub-Category</label>
                    <select name="subcategory" class="form-select" required>
                      {% for subcat in subcategories %}
                        <option value="{{ subcat }}">{{ subcat }}</option>
                      {% endfor %}
                    </select>
                  </div>

                  <div class="col-sm-6 col-md-3">
                    <label class="form-label">Type</label>
                    <select name="forecast_type" class="form-select">
                      <option value="months" selected>Monthly</option>
                      <option value="years">Yearly</option>
                    </select>
                  </div>

                  <div class="col-sm-6 col-md-3">
                    <label class="form-label">Periods</label>
                    <input type="number" name="periods" min="1" max="24" value="6" class="form-control" required>
                  </div>

                  <div class="col-12">
                    <button type="submit" class="btn btn-outline-success" id="subcatSubmit">
                      <span class="spinner-border spinner-border-sm me-1 d-none" id="subcatSpinner" aria-hidden="true"></span>
                      Forecast Sub-Category
                    </button>
                    <div class="form-text">Linear trend forecast from historical sales in the selected sub-category.</div>
                  </div>
                </form>
              </div>
            </div>
          {% else %}
            <div class="alert alert-info mt-3 mb-0">
              Please <a href="/login">log in</a> to access forecasting features.
            </div>
          {% endif %}
{% if current_user.is_authenticated and current_user.role == 'manager' %}
  <div class="card mt-3">
    <div class="card-body d-flex justify-content-between align-items-center">
      <div>
        <h5 class="mb-1">Manager Dashboard</h5>
        <p class="text-muted mb-0">Profitability (estimated), KPI tracker, discount impact, stock alerts.</p>
      </div>
      <a class="btn btn-primary" href="/manager-dashboard">Open</a>
    </div>
  </div>
{% endif %}
       {# Admin quick entry card - mirrors the Manager card #}
{% if current_user.is_authenticated and (current_user.role|lower == 'admin') %}
  <div class="card mt-3">
    <div class="card-body d-flex justify-content-between align-items-center">
      <div>
        <h5 class="mb-1">Admin Dashboard</h5>
        <p class="text-muted mb-0">User management, role settings, data & system controls.</p>
      </div>
      <a class="btn btn-outline-primary" href="/admin">Open</a>
    </div>
  </div>
{% endif %}
                    
        </div>
      </div>
    </div>

<!-- RIGHT: Quick Reports + Other Options -->
<div class="col-lg-4">
  <div class="card h-100">
    <div class="card-body">
      <h4 class="section-title"><i class="bi bi-speedometer2"></i> Quick Reports</h4>
      <div class="d-grid gap-2">
        <a class="btn btn-warning" href="/sales-trends"><i class="bi bi-calendar2-week"></i> Monthly Sales Trend</a>
        <a class="btn btn-warning" href="/top-products"><i class="bi bi-trophy"></i> Top Selling Sub-Categories</a>
        <a class="btn btn-success" href="/recommend"><i class="bi bi-geo"></i> Recommend by Region</a>
        <a class="btn btn-success" href="/compare-models"><i class="bi bi-sliders2"></i> ARIMA vs Prophet</a>
        <a class="btn btn-info" href="/roi"><i class="bi bi-cash-coin"></i> Marketing ROI Analysis</a> <!-- NEW -->

      </div>


          <hr class="my-4">

          <h4 class="section-title"><i class="bi bi-tools"></i> Other Options</h4>
          <ul class="list-group list-group-flush">
            <li class="list-group-item px-0"><a href="/segments">View Customer Segments</a></li>
            <li class="list-group-item px-0"><a href="/customers">View Customer List</a></li>
            <li class="list-group-item px-0"><a href="/recommend">Recommend Products by Region</a></li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- ROW 2 -->
  <div class="row g-4 mt-1">
    <div class="col-lg-6">
      <div class="card h-100">
        <div class="card-body">
          <h4 class="section-title"><i class="bi bi-activity"></i> Interactive Sales Analysis</h4>
          <form method="post" action="/interactive-analysis" class="row g-3">
            <div class="col-12">
              <label class="form-label">Select Sub-Category</label>
              <select name="subcategory" class="form-select" required>
                {% for subcat in subcategories %}
                  <option value="{{ subcat }}">{{ subcat }}</option>
                {% endfor %}
              </select>
            </div>
            <div class="col-12">
              <button type="submit" class="btn btn-primary">View Trend</button>
            </div>
          </form>
        </div>
      </div>
    </div>

    <div class="col-lg-6">
      <div class="card h-100">
        <div class="card-body">
          <h4 class="section-title"><i class="bi bi-diagram-3"></i> Model Comparison</h4>
          <p class="text-muted mb-3">Compare ARIMA vs Prophet forecasts for any sub-category.</p>
          <a href="/compare-models" class="btn btn-dark">Compare ARIMA vs Prophet</a>
        </div>
      </div>
    </div>
  </div>

  <footer class="mt-5 text-center small">
    Superstore Forecasting &amp; Analytics Portal © 2025 · <span class="subtle">v1.0</span>
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<!-- Tiny JS for quick presets and loading spinners -->
<script>
(function () {
  // Quick presets for Overall tab
  document.querySelectorAll('#overall .preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.querySelector('#overall input[name="periods"]');
      if (input) input.value = btn.dataset.val;
    });
  });

  // Loading spinners on submit
  function hookSubmit(formId, btnId, spinnerId) {
    const form = document.getElementById(formId);
    if (!form) return;
    form.addEventListener('submit', () => {
      const btn = document.getElementById(btnId);
      const spin = document.getElementById(spinnerId);
      if (btn && spin) {
        btn.disabled = true;
        spin.classList.remove('d-none');
      }
    });
  }
  hookSubmit('overallForm', 'overallSubmit', 'overallSpinner');
  hookSubmit('subcatForm', 'subcatSubmit', 'subcatSpinner');
})();
</script>

    </body>
    </html>
    """, subcategories=subcategories)


@app.route("/data-diagnostics")
@login_required
def data_diagnostics():
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p>Error reading data: {e}</p><a href='/'>Back</a>"

    cols = df.columns.tolist()
    sample = df.head(10).to_dict(orient="records")
    return render_template_string("""
      <h2>🧪 Data Diagnostics</h2>
      <p><b>Columns found ({{ cols|length }}):</b></p>
      <pre>{{ cols|tojson(indent=2) }}</pre>
      <h5>First 10 rows</h5>
      <div class="table-responsive">
        <table class="table table-sm table-striped">
          <thead>
            <tr>
              {% for c in cols %}<th>{{ c }}</th>{% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for r in sample %}
              <tr>
                {% for c in cols %}<td>{{ r[c] }}</td>{% endfor %}
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <a href="/">Back to Home</a>
    """, cols=cols, sample=sample)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as con:
            row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            user = User.from_row(row)
            login_user(user)
            # ⬇️ redirect admins to /admin, others to home
            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("home"))
        return "<p>❌ Invalid credentials</p><a href='/login'>Try again</a>"

    return """
    <h2>Login</h2>
    <form method="post">
      <input name="username" placeholder="username" required>
      <input name="password" type="password" placeholder="password" required>
      <button type="submit">Login</button>
    </form>
    <p>Try admin: <b>admin1 / pass123</b></p>
    """


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return "<p>👋 Logged out</p><a href='/'>Back to Home</a>"


@app.route("/admin")
@login_required
@roles_required("admin")
def admin_dashboard():
    users = get_all_users()
    # If a “just created user” message was flashed, it will show here
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
      <title>Admin Dashboard</title>
    </head>
    <body class="p-4">
      <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h2>👑 Admin Dashboard</h2>
          <div>
            <a href="/" class="btn btn-outline-secondary btn-sm">Home</a>
            <a href="/logout" class="btn btn-outline-danger btn-sm">Logout</a>
          </div>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, msg in messages %}
              <div class="alert alert-{{category}}">{{ msg|safe }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <div class="row g-4">
          <!-- Add User -->
          <div class="col-lg-5">
            <div class="card">
              <div class="card-body">
                <h5 class="card-title">➕ Add User</h5>
                <form method="post" action="{{ url_for('admin_add_user') }}" class="row g-3">
                  <div class="col-12">
                    <label class="form-label">Username</label>
                    <input name="username" class="form-control" required>
                  </div>
                  <div class="col-12">
                    <label class="form-label">Password</label>
                    <input name="password" type="text" class="form-control" required>
                  </div>
                  <div class="col-12">
                    <label class="form-label">Role</label>
                    <select name="role" class="form-select">
                      <option value="analyst">Analyst</option>
                      <option value="manager">Manager</option>
                      <option value="admin">Admin</option>
                    </select>
                  </div>
                  <div class="col-12">
                    <button class="btn btn-primary" type="submit">Create</button>
                  </div>
                </form>
                <p class="text-muted mt-2 small">Share these credentials with the user to let them log in at <code>/login</code>.</p>
              </div>
            </div>
          </div>

          <!-- Users Table -->
          <div class="col-lg-7">
            <div class="card">
              <div class="card-body">
                <h5 class="card-title">👥 Users</h5>
                <div class="table-responsive">
                  <table class="table table-sm align-middle">
                    <thead>
                      <tr><th>ID</th><th>Username</th><th>Role</th><th class="text-end">Actions</th></tr>
                    </thead>
                    <tbody>
                      {% for u in users %}
                      <tr>
                        <td>{{ u.id }}</td>
                        <td>{{ u.username }}</td>
                        <td><span class="badge bg-secondary text-capitalize">{{ u.role }}</span></td>
                        <td class="text-end">
                          <form method="post" action="{{ url_for('admin_reset_password', uid=u.id) }}" class="d-inline">
                            <button class="btn btn-outline-warning btn-sm" type="submit">Reset Password</button>
                          </form>
                          {% if u.username != 'admin1' %}
                          <form method="post" action="{{ url_for('admin_delete_user', uid=u.id) }}" class="d-inline" onsubmit="return confirm('Delete user {{u.username}}?')">
                            <button class="btn btn-outline-danger btn-sm" type="submit">Delete</button>
                          </form>
                          {% endif %}
                        </td>
                      </tr>
                      {% endfor %}
                      {% if not users %}
                      <tr><td colspan="4" class="text-muted">No users yet.</td></tr>
                      {% endif %}
                    </tbody>
                  </table>
                </div>

                <div class="alert alert-info small mt-3">
                  <b>Tip:</b> After creating a user, copy this invite:
                  <pre class="mt-2 mb-0">Hi,<br>You've been granted access to the Sales & Insights portal.<br>Login: /login<br>Username: &lt;their username&gt;<br>Password: &lt;their password&gt;<br>Role: &lt;manager/analyst&gt;</pre>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """, users=users)
@app.route("/admin/add-user", methods=["POST"])
@login_required
@roles_required("admin")
def admin_add_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    role = request.form.get("role", "analyst")

    if role not in ("admin", "manager", "analyst"):
        flash("Invalid role.", "danger")
        return redirect(url_for("admin_dashboard"))

    if not username or not password:
        flash("Username and password required.", "danger")
        return redirect(url_for("admin_dashboard"))

    try:
        with get_db() as con:
            con.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                (username, generate_password_hash(password), role)
            )
        flash(
            f"✅ Created user <b>{username}</b> as <b>{role}</b>.<br>"
            f"<code>Username: {username}</code><br>"
            f"<code>Password: {password}</code>",
            "success"
        )
    except sqlite3.IntegrityError:
        flash("❌ Username already exists.", "danger")

    return redirect(url_for("admin_dashboard"))

def _generate_temp_password(length=10):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

@app.route("/admin/reset-password/<int:uid>", methods=["POST"])
@login_required
@roles_required("admin")
def admin_reset_password(uid):
    # don't allow resetting your own password from here (optional)
    # if uid == current_user.id: ...
    new_pw = _generate_temp_password()
    set_user_password(uid, new_pw)
    flash(
        f"🔑 Password reset. New temporary password:<br><code>{new_pw}</code>",
        "warning"
    )
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete-user/<int:uid>", methods=["POST"])
@login_required
@roles_required("admin")
def admin_delete_user(uid):
    # prevent removing the seeded admin (optional)
    with get_db() as con:
        row = con.execute("SELECT username, role FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        flash("User not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    if row["username"] == "admin1":
        flash("Cannot delete the primary admin.", "danger")
        return redirect(url_for("admin_dashboard"))
    delete_user(uid)
    flash(f"🗑️ Deleted user <b>{row['username']}</b>.", "info")
    return redirect(url_for("admin_dashboard"))

@app.route("/predict", methods=["POST"])
@login_required
@roles_required("analyst", "manager")
def predict():
    import numpy as np
    import pickle

    # ---- Read form values ----
    try:
        marketing_spend = float(request.form.get("marketing_spend", "0"))
    except:
        marketing_spend = 0.0

    # Accept both percent (e.g., 10) and fraction (e.g., 0.10)
    try:
        disc_raw = float(request.form.get("discount", "0"))
    except:
        disc_raw = 0.0
    discount_frac = disc_raw / 100.0 if disc_raw > 1.0 else disc_raw
    discount_pct = discount_frac * 100.0

    region = request.form.get("region", "")

    # ---- One-hot encode region (Central/East/South/West) ----
    known_regions = ["Central", "East", "South", "West"]
    region_encoded = [1 if region == r else 0 for r in known_regions]
    region_note = "" if region in known_regions else "Region not in training set; encoded as all zeros."

    # ---- Build feature vector ----
    # [Marketing Spend, Discount (fraction 0–1), Region_Central, Region_East, Region_South, Region_West]
    input_features = [marketing_spend, discount_frac] + region_encoded
    input_array = np.array([input_features])

    # ---- Load model & predict ----
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        predicted_sales = float(model.predict(input_array)[0])
        predict_error = ""
    except Exception as e:
        predicted_sales = None
        predict_error = str(e)

    # ---- Render UI ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Predicted Sales — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .kpi { border-radius: var(--card-radius); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-bullseye"></i> Predicted Sales Result</h3>
      <p class="text-muted mb-3">Your scenario was scored by the trained model. Review the inputs and predicted output below.</p>

      {% if predict_error %}
        <div class="alert alert-danger"><b>Prediction error:</b> {{ predict_error }}</div>
      {% endif %}

      <!-- KPI strip -->
      <div class="row g-3">
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">Marketing Spend</div>
            <div class="fs-4 fw-semibold">${{ '{:,.2f}'.format(marketing_spend) }}</div>
            <div class="small text-muted">Input</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">Discount</div>
            <div class="fs-4 fw-semibold">{{ '{:.1f}'.format(discount_pct) }}%</div>
            <div class="small text-muted">Interpreted for model as {{ '{:.3f}'.format(discount_frac) }}</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">Region</div>
            <div class="fs-5">{{ region if region else '—' }}</div>
            {% if region_note %}
              <div class="small text-warning"><i class="bi bi-exclamation-triangle-fill me-1"></i>{{ region_note }}</div>
            {% endif %}
          </div>
        </div>
      </div>

      <hr class="my-4">

      <!-- Prediction -->
      <div class="row g-4 align-items-center">
        <div class="col-lg-7">
          <div class="p-4 bg-white border rounded">
            <h5 class="mb-2"><i class="bi bi-flag"></i> Predicted Sales</h5>
            <div class="display-6 fw-bold">
              {% if predicted_sales is not none %}
                ${{ '{:,.2f}'.format(predicted_sales) }}
              {% else %}
                <span class="text-danger">Unavailable</span>
              {% endif %}
            </div>
            <div class="small text-muted mt-2">Point estimate from the trained model.</div>
          </div>
        </div>

        <div class="col-lg-5">
          <div class="p-3 bg-white border rounded">
            <h6 class="mb-2">Model Inputs (encoded)</h6>
            <div class="table-responsive">
              <table class="table table-sm align-middle mono mb-0">
                <thead class="table-light">
                  <tr>
                    <th>Feature</th><th class="text-end">Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>Marketing Spend</td><td class="text-end">{{ '{:,.2f}'.format(marketing_spend) }}</td></tr>
                  <tr><td>Discount (fraction)</td><td class="text-end">{{ '{:.4f}'.format(discount_frac) }}</td></tr>
                  <tr><td>Region_Central</td><td class="text-end">{{ region_encoded[0] }}</td></tr>
                  <tr><td>Region_East</td><td class="text-end">{{ region_encoded[1] }}</td></tr>
                  <tr><td>Region_South</td><td class="text-end">{{ region_encoded[2] }}</td></tr>
                  <tr><td>Region_West</td><td class="text-end">{{ region_encoded[3] }}</td></tr>
                </tbody>
              </table>
            </div>
            <div class="small text-muted mt-2">Vector order: [Spend, Discount, Central, East, South, West]</div>
          </div>
        </div>
      </div>

      <div class="mt-4 d-flex gap-2">
        <a href="/" class="btn btn-primary"><i class="bi bi-arrow-left"></i> Back to Home</a>
        <a href="/" class="btn btn-outline-secondary">Try another prediction</a>
      </div>
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    # Jinja vars
    marketing_spend=marketing_spend,
    discount_frac=discount_frac,
    discount_pct=discount_pct,
    region=region,
    region_encoded=region_encoded,
    region_note=region_note,
    predicted_sales=predicted_sales,
    predict_error=predict_error
    )



@app.route('/segments')
def segments():
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # ---- Load & validate ----
    try:
        df = pd.read_csv(SEGMENTED_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading segmented_customers.csv:</b> {e}</p><a href='/'>Back to Home</a>"

    if 'Segment Label' not in df.columns:
        return "<p style='color:red'><b>Column 'Segment Label' not found in segmented_customers.csv</b></p><a href='/'>Back to Home</a>"

    # ---- Aggregate ----
    counts = df['Segment Label'].value_counts().sort_values(ascending=False)
    total_customers = int(counts.sum())
    n_segments = int(counts.shape[0])

    # CSV for download
    summary_csv = counts.rename_axis('Segment').reset_index(name='Customers')
    csv_b64 = base64.b64encode(summary_csv.to_csv(index=False).encode('utf-8')).decode('utf-8')

    # ---- Chart (pie; fallback to bar if only 1 segment) ----
    chart_b64 = None
    try:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if len(counts) > 1:
            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Customer Segments Share')
        else:
            ax.bar(counts.index, counts.values)
            ax.set_title('Customer Segments')
            ax.set_ylabel('Customers')
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        chart_b64 = None

    # ---- Build rows for table ----
    rows_html = ""
    for seg, cnt in counts.items():
        pct = (cnt / total_customers) * 100 if total_customers else 0
        rows_html += f"""
          <tr>
            <td><span class="badge text-bg-primary">{seg}</span></td>
            <td class="text-end">{cnt:,}</td>
            <td class="text-end">{pct:.1f}%</td>
          </tr>
        """

    # ---- Render ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Customer Segments — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .kpi { font-weight:600; font-size:1.05rem; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card mb-4">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-people"></i> Customer Segments</h3>
      <p class="text-muted mb-3">Distribution of customers across identified segments.</p>

      <!-- KPIs -->
      <div class="row g-3 mb-3">
        <div class="col-sm-6 col-lg-3">
          <div class="p-3 bg-light rounded border">
            <div class="text-muted">Total Customers</div>
            <div class="kpi">{{ total_customers|int }}</div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="p-3 bg-light rounded border">
            <div class="text-muted"># Segments</div>
            <div class="kpi">{{ n_segments|int }}</div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-6 d-flex align-items-center">
          <div>
            <a class="btn btn-sm btn-outline-secondary me-2" href="data:text/csv;base64,{{ csv_b64 }}" download="segment_summary.csv">
              ⬇️ Download CSV
            </a>
            {% if chart_b64 %}
            <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ chart_b64 }}" download="segment_share.png">
              ⬇️ Download PNG
            </a>
            {% endif %}
          </div>
        </div>
      </div>

      <!-- Content -->
      <div class="row g-4">
        <div class="col-lg-6">
          <div class="table-responsive">
            <table class="table table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th>Segment</th>
                  <th class="text-end">Customers</th>
                  <th class="text-end">% Share</th>
                </tr>
              </thead>
              <tbody>
                {{ rows_html|safe }}
              </tbody>
            </table>
          </div>
        </div>

        <div class="col-lg-6">
          {% if chart_b64 %}
            <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_b64 }}" alt="Segment Chart">
          {% else %}
            <div class="alert alert-warning">Chart could not be generated. You can still download the CSV.</div>
          {% endif %}
        </div>
      </div>

      <div class="mt-3">
        <a href="/" class="btn btn-link">&larr; Back to Home</a>
      </div>
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """, 
    total_customers=total_customers, 
    n_segments=n_segments,
    rows_html=rows_html,
    chart_b64=chart_b64,
    csv_b64=csv_b64)



@app.route('/customers')
def customers():
    import sqlite3
    import pandas as pd
    from io import StringIO
    import base64

    # ---- Load from SQLite ----
    try:
        conn = sqlite3.connect(SUPERSTORE_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT CustomerID, CustomerName, Segment, Region
            FROM Customers
            ORDER BY CustomerName COLLATE NOCASE
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        return f"<p style='color:red'><b>Error reading database:</b> {e}</p><a href='/'>Back to Home</a>"

    # ---- Build DataFrame ----
    df = pd.DataFrame(rows, columns=["CustomerID", "CustomerName", "Segment", "Region"])
    total_customers = len(df)
    unique_segments = sorted([x for x in df["Segment"].dropna().unique()])
    unique_regions  = sorted([x for x in df["Region"].dropna().unique()])

    # CSV (full dataset) for download
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    csv_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("utf-8")

    # ---- Render nice UI ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Customers — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .chip { font-size: .8rem; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-people"></i> Customer List</h3>
      <p class="text-muted mb-3">Browse customers with quick search and filters. Download full list as CSV.</p>

      <!-- KPIs + Download -->
      <div class="row g-3 mb-3">
        <div class="col-sm-6 col-lg-3">
          <div class="p-3 bg-light rounded border">
            <div class="text-muted">Total Customers</div>
            <div class="fs-5 fw-semibold">{{ total_customers }}</div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="p-3 bg-light rounded border">
            <div class="text-muted">Segments</div>
            <div class="fs-5 fw-semibold">{{ unique_segments|length }}</div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="p-3 bg-light rounded border">
            <div class="text-muted">Regions</div>
            <div class="fs-5 fw-semibold">{{ unique_regions|length }}</div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3 d-flex align-items-center">
          <a class="btn btn-outline-secondary ms-auto" href="data:text/csv;base64,{{ csv_b64 }}" download="customers.csv">
            ⬇️ Download CSV
          </a>
        </div>
      </div>

      <!-- Controls -->
      <div class="row g-3 mb-3">
        <div class="col-md-4">
          <label class="form-label">Search</label>
          <input id="search" type="text" class="form-control" placeholder="Search by name, ID…">
        </div>
        <div class="col-md-4">
          <label class="form-label">Filter by Segment</label>
          <select id="segmentFilter" class="form-select">
            <option value="">All Segments</option>
            {% for s in unique_segments %}
              <option value="{{ s }}">{{ s }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-md-4">
          <label class="form-label">Filter by Region</label>
          <select id="regionFilter" class="form-select">
            <option value="">All Regions</option>
            {% for r in unique_regions %}
              <option value="{{ r }}">{{ r }}</option>
            {% endfor %}
          </select>
        </div>
      </div>

      <!-- Table -->
      <div class="table-responsive">
        <table id="custTable" class="table table-hover align-middle">
          <thead class="table-light">
            <tr>
              <th>Customer</th>
              <th>Customer ID</th>
              <th>Segment</th>
              <th>Region</th>
            </tr>
          </thead>
          <tbody>
            {% for row in rows %}
              <tr>
                <td>{{ row[1] }}</td>
                <td class="text-muted">{{ row[0] }}</td>
                <td><span class="badge text-bg-primary chip">{{ row[2] }}</span></td>
                <td><span class="badge text-bg-secondary chip">{{ row[3] }}</span></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>

      <div class="mt-3 d-flex justify-content-between align-items-center">
        <small class="text-muted" id="visibleCount"></small>
        <a href="/" class="btn btn-link">&larr; Back to Home</a>
      </div>
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<!-- Tiny client-side filtering -->
<script>
(function () {
  const table = document.getElementById('custTable');
  const rows  = Array.from(table.querySelectorAll('tbody tr'));
  const q     = document.getElementById('search');
  const seg   = document.getElementById('segmentFilter');
  const reg   = document.getElementById('regionFilter');
  const visibleCount = document.getElementById('visibleCount');

  function norm(s){ return (s||'').toString().toLowerCase(); }

  function applyFilters(){
    const term = norm(q.value);
    const segv = seg.value;
    const regv = reg.value;
    let shown = 0;

    rows.forEach(tr => {
      const name   = norm(tr.cells[0].textContent);
      const custid = norm(tr.cells[1].textContent);
      const segtx  = tr.cells[2].textContent.trim();
      const regtx  = tr.cells[3].textContent.trim();

      const matchesText = !term || name.includes(term) || custid.includes(term);
      const matchesSeg  = !segv || segtx === segv;
      const matchesReg  = !regv || regtx === regv;

      const ok = matchesText && matchesSeg && matchesReg;
      tr.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });

    visibleCount.textContent = shown + " of " + rows.length + " customers shown";
  }

  q.addEventListener('input', applyFilters);
  seg.addEventListener('change', applyFilters);
  reg.addEventListener('change', applyFilters);
  applyFilters();
})();
</script>
</body>
</html>
    """,
    rows=rows,
    total_customers=total_customers,
    unique_segments=unique_segments,
    unique_regions=unique_regions,
    csv_b64=csv_b64)


@app.route('/forecast', methods=['GET', 'POST'])
@login_required
@roles_required("manager", "analyst")
def forecast():
    import base64
    import pickle
    import numpy as np
    from io import StringIO, BytesIO
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # ----- UI state -----
    selected_type = "monthly"    # 'monthly' or 'yearly'
    periods_val = 6              # 1..24
    results = []
    error_msg = ""
    csv_b64 = None
    chart_b64 = None
    label = "Period"

    # ----- helpers -----
    def fig_to_base64():
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        out = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return out

    # ----- POST -----
    if request.method == 'POST':
        try:
            periods_val = max(1, min(24, int(request.form.get('periods', 6))))
            selected_type = (request.form.get('forecast_type') or 'monthly').strip().lower()

            if selected_type == "monthly":
              model_path = MODEL_MONTHLY
              label = "Month"
            else:
                model_path = MODEL_YEARLY
                label = "Year"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Simple index-based prediction as in your code
            start_index = 1
            future_indexes = np.array([[start_index + i] for i in range(47, 47 + periods_val)])
            preds = model.predict(future_indexes)

            results = [{"label": f"{label} {i+1}", "value": float(p)} for i, p in enumerate(preds)]

            # CSV
            df_out = pd.DataFrame(results).rename(columns={"label": label, "value": "Predicted Sales"})
            csv_io = StringIO()
            df_out.to_csv(csv_io, index=False)
            csv_b64 = base64.b64encode(csv_io.getvalue().encode("utf-8")).decode("utf-8")

            # Chart
            x_labels = [r["label"] for r in results]
            y_vals = [r["value"] for r in results]
            plt.figure(figsize=(10, 4.5))
            plt.plot(range(1, len(y_vals) + 1), y_vals, marker='o')
            plt.title(f"{selected_type.capitalize()} Forecast — Next {periods_val} {label.lower()}s")
            plt.xlabel(label)
            plt.ylabel("Predicted Sales")
            plt.xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels, rotation=45, ha='right')
            plt.tight_layout()
            chart_b64 = fig_to_base64()

        except Exception as e:
            error_msg = str(e)

    # ----- TEMPLATE -----
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forecast (Overall) — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    /* Light theme */
    body { background:#f7f8fa; color:#212529; padding:24px; }
    .card { background:#ffffff; border-radius:14px; box-shadow:0 6px 20px rgba(0,0,0,.06); }
    .chart-wrap img { max-width:100%; height:auto; border:1px solid #e9ecef; border-radius:8px; background:#fff; }
    .result-item { display:flex; justify-content:space-between; padding:.6rem .85rem; border-bottom:1px dashed #e9ecef; }
    .result-item:last-child { border-bottom:none; }
    .result-left { color:#495057; }
    .result-right { color:#212529; font-weight:700; }

    /* Form alignment helpers (baseline on lg+, stacks on small) */
    .ff-row.row { align-items:stretch; }
    .ff-field { display:flex; flex-direction:column; height:100%; }
    .ff-actions { display:grid; align-content:end; gap:.5rem; height:100%; }
    .ff-label { margin-bottom:.35rem; font-weight:600; }
    .presets .btn { min-width:3rem; }
    .w-grow { flex: 1 1 auto; min-width: 10rem; } /* ensure input has space */

    @media (min-width:992px){
      .ff-row .col { display:flex; }
      .ff-row .ff-field, .ff-row .ff-actions { width:100%; }
    }
  </style>
</head>
<body>

<!-- Top navbar with Sales Insights home link -->
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="mb-3"><i class="bi bi-bar-chart-steps me-1"></i> Forecast (Overall)</h3>

      <!-- Baseline-aligned, spacious form -->
      <form method="post" class="row ff-row row-cols-1 row-cols-lg-4 g-3" id="forecastForm">
        <!-- Periods (suffix only), presets below so input has room -->
        <div class="col">
          <div class="ff-field">
            <label class="ff-label">Periods</label>
            <div class="input-group">
              <input type="number" name="periods" min="1" max="24" value="{{ periods_val }}"
                     class="form-control w-grow" required>
              <span class="input-group-text">{{ 'months' if selected_type == 'monthly' else 'years' }}</span>
            </div>
            <div class="presets mt-2">
              <div class="btn-group" role="group" aria-label="Quick presets">
                <button class="btn btn-outline-secondary btn-sm preset" data-val="3"  type="button">3</button>
                <button class="btn btn-outline-secondary btn-sm preset" data-val="6"  type="button">6</button>
                <button class="btn btn-outline-secondary btn-sm preset" data-val="12" type="button">12</button>
              </div>
              <div class="form-text">1–24 {{ 'months' if selected_type == 'monthly' else 'years' }}</div>
            </div>
          </div>
        </div>

        <!-- Forecast type -->
        <div class="col">
          <div class="ff-field">
            <label class="ff-label">Forecast Type</label>
            <select name="forecast_type" class="form-select">
              <option value="monthly" {% if selected_type == 'monthly' %}selected{% endif %}>Monthly</option>
              <option value="yearly"  {% if selected_type == 'yearly'  %}selected{% endif %}>Yearly</option>
            </select>
            <div class="form-text">Choose monthly or yearly forecast.</div>
          </div>
        </div>

        <!-- Info spacer (keeps 4-up layout even) -->
        <div class="col d-none d-lg-block">
          <div class="ff-field">
            <label class="ff-label">&nbsp;</label>
            <div class="text-muted">Set periods & type, then run forecast.</div>
          </div>
        </div>

        <!-- Action -->
        <div class="col">
          <div class="ff-actions">
            <button type="submit" class="btn btn-success w-100" id="submitBtn">Forecast</button>
            <a href="/" class="btn btn-link text-decoration-none">Back to Home</a>
          </div>
        </div>
      </form>

      {% if error_msg %}
        <div class="alert alert-danger mt-3"><b>Error:</b> {{ error_msg }}</div>
      {% endif %}

      {% if results and not error_msg %}
        <hr class="my-4">

        <div class="row g-4">
          <div class="col-12 col-lg-6">
            <div class="d-flex justify-content-between align-items-center">
              <h5 class="mb-2">Table</h5>
              {% if csv_b64 %}
                <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ csv_b64 }}"
                   download="forecast_{{ selected_type }}_{{ periods_val }}.csv">
                  <i class="bi bi-download me-1"></i>Download CSV
                </a>
              {% endif %}
            </div>
            <div class="border rounded bg-white">
              {% for r in results %}
                <div class="result-item">
                  <div class="result-left">{{ r.label }}</div>
                  <div class="result-right">${{ "{:,.2f}".format(r.value) }}</div>
                </div>
              {% endfor %}
            </div>
          </div>

          <div class="col-12 col-lg-6">
            <div class="d-flex justify-content-between align-items-center">
              <h5 class="mb-2">Chart</h5>
              {% if chart_b64 %}
                <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ chart_b64 }}"
                   download="forecast_{{ selected_type }}_{{ periods_val }}.png">
                  <i class="bi bi-download me-1"></i>Download PNG
                </a>
              {% endif %}
            </div>
            <div class="chart-wrap mt-2">
              {% if chart_b64 %}
                <img src="data:image/png;base64,{{ chart_b64 }}" alt="Forecast chart">
              {% endif %}
            </div>
          </div>
        </div>
      {% endif %}
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">Superstore Forecasting &amp; Analytics Portal © 2025</footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // Presets: set value without squashing the input
  document.querySelectorAll('#forecastForm .preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.querySelector('#forecastForm input[name="periods"]');
      if (input) input.value = btn.dataset.val;
    });
  });
</script>
</body>
</html>
    """,
    selected_type=selected_type,
    periods_val=periods_val,
    results=results,
    csv_b64=csv_b64,
    chart_b64=chart_b64,
    error_msg=error_msg)

@app.route('/forecast_subcat', methods=['GET', 'POST'])
@app.route('/forecast-subcat', methods=['GET', 'POST'])
@login_required
@roles_required("analyst", "manager")
def forecast_subcat():
    import base64
    import numpy as np
    from io import StringIO
    import pandas as pd
    import matplotlib.pyplot as plt

    periods_val = 6
    selected_subcat = None
    selected_region = "All"
    results = []
    error_msg = ""
    csv_b64 = None
    chart_b64 = None

    # Load data
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back</a>"

    required = {"Order Date", "Sales", "Sub-Category"}
    if not required.issubset(df.columns):
        miss = ", ".join(sorted(required - set(df.columns)))
        return f"<p style='color:red'><b>Missing columns:</b> {miss}</p><a href='/'>Back</a>"

    have_region = "Region" in df.columns
    subcategories = sorted(df["Sub-Category"].dropna().unique().tolist())
    regions = ["All"] + (sorted(df["Region"].dropna().unique().tolist()) if have_region else ["All"])

    # Clean
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    if request.method == "POST":
        selected_subcat = request.form.get("subcategory") or None
        selected_region = request.form.get("region", "All")
        try:
            periods_val = max(1, min(24, int(request.form.get("periods", 6))))
        except:
            periods_val = 6

        if not selected_subcat:
            error_msg = "Please choose a sub-category."
        elif selected_subcat not in subcategories:
            error_msg = f"Unknown sub-category: {selected_subcat}"
        else:
            q = df[df["Sub-Category"] == selected_subcat].copy()
            if have_region and selected_region != "All":
                q = q[q["Region"] == selected_region]

            if q.empty:
                error_msg = "No data for the chosen sub-category/region."
            else:
                # Monthly aggregate
                q["Month"] = q["Order Date"].dt.to_period("M").dt.to_timestamp()
                monthly = (
                    q.groupby("Month", as_index=False)["Sales"]
                     .sum()
                     .sort_values("Month")
                )
                y = monthly.set_index("Month")["Sales"].asfreq("MS")

                if y.dropna().empty or len(y.dropna()) < 2:
                    error_msg = "Not enough history to forecast."
                else:
                    # Seasonal-naive monthly forecast
                    hist = y.fillna(method="ffill").fillna(method="bfill")
                    last_vals = hist.tail(12).values
                    if len(last_vals) < 12:
                        last_vals = np.tile(np.mean(hist.tail(max(3, len(hist))).values), 12)

                    future_vals = [float(last_vals[i % 12]) for i in range(periods_val)]
                    results = [{"label": f"Month {i+1}", "value": v} for i, v in enumerate(future_vals)]

                    # CSV
                    out = pd.DataFrame(results).rename(columns={"label": "Month", "value": "Predicted Sales"})
                    buf = StringIO()
                    out.to_csv(buf, index=False)
                    csv_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

                    # PNG chart
                    x_labels = [r["label"] for r in results]
                    y_vals = [r["value"] for r in results]
                    plt.figure(figsize=(10, 4.5))
                    plt.plot(range(1, len(y_vals) + 1), y_vals, marker='o')
                    title_bits = [selected_subcat]
                    if have_region and selected_region != "All":
                        title_bits.append(f"({selected_region})")
                    plt.title(" ".join(title_bits) + f" — Next {periods_val} months")
                    plt.xlabel("Month")
                    plt.ylabel("Predicted Sales")
                    plt.xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels, rotation=45, ha='right')
                    plt.tight_layout()
                    chart_b64 = fig_to_base64()

    # UI
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forecast by Sub-Category — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    body { background:#f7f8fa; padding:24px; }
    .card { border-radius:14px; box-shadow:0 6px 20px rgba(0,0,0,.05); }
    .result-item { display:flex; justify-content:space-between; padding:.6rem .85rem; border-bottom:1px dashed #e9ecef; }
    .result-item:last-child { border-bottom:none; }
    .chart-wrap { text-align:center; }
    .chart-wrap img { max-width:100%; height:auto; border:1px solid #eee; border-radius:8px; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="mb-2"><i class="bi bi-diagram-3"></i> Sub-Category Forecast</h3>
      <p class="text-muted">Forecast monthly sales for a specific sub-category{{ ' and region' if regions|length>1 else '' }}.</p>

      <form method="post" class="row gy-3 gx-3 align-items-end" id="scForm">
        <div class="col-12 col-md-6">
          <label class="form-label">Sub-Category</label>
          <select name="subcategory" class="form-select" required>
            <option value="" disabled selected>Select Sub-Category</option>
            {% for s in subcategories %}
              <option value="{{ s }}" {% if selected_subcat == s %}selected{% endif %}>{{ s }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-12 col-md-6">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{ r }}" {% if selected_region == r %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-12 col-md-6">
          <label class="form-label">Forecast Periods (months)</label>
          <input type="number" name="periods" min="1" max="24" value="{{ periods_val }}" class="form-control" required>
          <div class="form-text">1–24</div>
        </div>

        <div class="col-12 col-md-6">
          <label class="form-label d-block">Quick presets</label>
          <div class="btn-group w-100" role="group" aria-label="Quick presets">
            <button class="btn btn-outline-secondary btn-sm preset" data-val="3" type="button">3</button>
            <button class="btn btn-outline-secondary btn-sm preset" data-val="6" type="button">6</button>
            <button class="btn btn-outline-secondary btn-sm preset" data-val="12" type="button">12</button>
          </div>
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-success" id="submitBtn">
            <span class="spinner-border spinner-border-sm me-1 d-none" id="spinner" aria-hidden="true"></span>
            Forecast
          </button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      {% if error_msg %}
        <div class="alert alert-danger mt-3"><b>Error:</b> {{ error_msg }}</div>
      {% endif %}

      {% if results and not error_msg %}
        <hr class="my-4">
        <div class="row g-4">
          <div class="col-12 col-lg-6">
            <div class="d-flex justify-content-between align-items-center">
              <h5 class="mb-2">Table</h5>
              {% if csv_b64 %}
                <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ csv_b64 }}" download="forecast_subcat_{{ selected_subcat|replace(' ','_')|lower }}.csv">
                  ⬇️ Download CSV
                </a>
              {% endif %}
            </div>
            <div class="border rounded bg-white">
              {% for r in results %}
                <div class="result-item">
                  <div class="text-muted">{{ r.label }}</div>
                  <div><b>${{ "{:,.2f}".format(r.value) }}</b></div>
                </div>
              {% endfor %}
            </div>
          </div>
          <div class="col-12 col-lg-6">
            <div class="d-flex justify-content-between align-items-center">
              <h5 class="mb-2">Chart</h5>
              {% if chart_b64 %}
                <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ chart_b64 }}" download="forecast_subcat_{{ selected_subcat|replace(' ','_')|lower }}.png">
                  ⬇️ Download PNG
                </a>
              {% endif %}
            </div>
            <div class="chart-wrap mt-2">
              {% if chart_b64 %}
                <img src="data:image/png;base64,{{ chart_b64 }}" alt="Sub-category forecast chart">
              {% endif %}
            </div>
          </div>
        </div>
      {% endif %}
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">Superstore Forecasting &amp; Analytics Portal © 2025</footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
(function () {
  document.querySelectorAll('.preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.querySelector('input[name="periods"]');
      input.value = btn.dataset.val;
    });
  });
  const form = document.getElementById('scForm');
  form.addEventListener('submit', () => {
    const btn = document.getElementById('submitBtn');
    const spin = document.getElementById('spinner');
    btn.disabled = true; spin.classList.remove('d-none');
  });
})();
</script>
</body>
</html>
    """,
    subcategories=subcategories,
    regions=regions,
    selected_subcat=selected_subcat,
    selected_region=selected_region,
    periods_val=periods_val,
    results=results,
    csv_b64=csv_b64,
    chart_b64=chart_b64,
    error_msg=error_msg)

@app.route('/roi', methods=['GET', 'POST'])
@login_required
@roles_required("manager", "analyst")
def roi_analysis():
    import pandas as pd
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    import numpy as np

    # ---------- data ----------
    df = pd.read_csv(DATA_CSV)
    regions = ["All"] + sorted(df['Region'].dropna().unique())
    subcats = ["All"] + sorted(df['Sub-Category'].dropna().unique())

    # form state
    selected_region = "All"
    selected_subcat = "All"

    # outputs
    csv_b64 = None
    chart_b64 = None
    roi_table = []
    kpi_total_sales = 0.0
    kpi_total_mkt   = 0.0
    kpi_roi_percent = None
    group_field = None
    title_suffix = ""

    # ---------- helpers ----------
    def df_to_csv_b64(dframe: pd.DataFrame) -> str:
        buf = BytesIO()
        dframe.to_csv(buf, index=False)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def fig_to_png_b64() -> str:
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def safe_roi(tsales, tmkt):
        return ((tsales - tmkt) / tmkt) * 100 if tmkt and tmkt != 0 else np.nan

    # ---------- filter + compute ----------
    if request.method == "POST":
        selected_region = request.form.get("region", "All")
        selected_subcat = request.form.get("subcat", "All")

    dff = df.copy()
    if selected_region != "All":
        dff = dff[dff["Region"] == selected_region]
    if selected_subcat != "All":
        dff = dff[dff["Sub-Category"] == selected_subcat]

    # KPIs for current filter (always shown)
    kpi_total_sales = float(pd.to_numeric(dff["Sales"], errors="coerce").fillna(0).sum())
    kpi_total_mkt   = float(pd.to_numeric(dff["Marketing Spend"], errors="coerce").fillna(0).sum())
    kpi_roi_percent = safe_roi(kpi_total_sales, kpi_total_mkt)

    # Decide grouping logic
    if selected_region == "All":
        group_field = "Region"
        title_suffix = "by Region"
    elif selected_subcat == "All":
        group_field = "Sub-Category"
        title_suffix = "by Sub-Category"
    else:
        group_field = None
        title_suffix = f"for {selected_region} / {selected_subcat}"

    if group_field:
        grouped = (
            dff.groupby(group_field)
               .agg(Total_Sales=('Sales', 'sum'),
                    Total_Marketing=('Marketing Spend', 'sum'))
               .reset_index()
        )
        grouped["ROI (%)"] = np.where(
            grouped["Total_Marketing"] > 0,
            ((grouped["Total_Sales"] - grouped["Total_Marketing"]) / grouped["Total_Marketing"]) * 100,
            np.nan
        )
        roi_table = grouped.to_dict(orient="records")
        csv_b64 = df_to_csv_b64(grouped)

        # chart
        plt.figure(figsize=(9, 5))
        plt.bar(grouped[group_field], grouped["ROI (%)"])
        plt.title(f"Marketing ROI {title_suffix}")
        plt.ylabel("ROI (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_b64 = fig_to_png_b64()
    else:
        # Single segment → one-row table, no bar chart
        single = pd.DataFrame([{
            "Segment": f"{selected_region} / {selected_subcat}",
            "Total_Sales": kpi_total_sales,
            "Total_Marketing": kpi_total_mkt,
            "ROI (%)": kpi_roi_percent
        }])
        roi_table = single.to_dict(orient="records")
        csv_b64 = df_to_csv_b64(single)
        chart_b64 = None

    # ---------- UI ----------
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Marketing ROI — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    body { background:#f7f8fa; }
    .page { padding:24px; }
    .card { border-radius:14px; box-shadow:0 6px 20px rgba(0,0,0,.05); }
    .kpi { background:#fff; border:1px solid #eee; border-radius:12px; padding:16px; }
    .kpi h6 { margin:0; color:#6c757d; }
    .kpi .val { font-size:1.25rem; font-weight:700; }
    .toolbar { gap:.5rem; }
  </style>
</head>
<body>

<nav class="navbar navbar-expand-lg bg-body-tertiary px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex toolbar">
    <a href="/" class="btn btn-sm btn-outline-secondary"><i class="bi bi-house"></i> Back to Home</a>
    <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
    <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
  </div>
</nav>

<div class="page container">
  <div class="card mb-3">
    <div class="card-body">
      <div class="d-flex justify-content-between align-items-center mb-1">
        <h3 class="mb-0"><i class="bi bi-cash-coin me-2"></i> Marketing ROI Analysis</h3>
        <div class="d-flex toolbar">
          {% if csv_b64 %}
            <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ csv_b64 }}" download="roi_analysis.csv">
              <i class="bi bi-download"></i> CSV
            </a>
          {% endif %}
          {% if chart_b64 %}
            <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ chart_b64 }}" download="roi_chart.png">
              <i class="bi bi-image"></i> PNG
            </a>
          {% endif %}
        </div>
      </div>
      <p class="text-muted mb-4">Analyze return on marketing spend {{ title_suffix }}. Filter below to focus the KPIs and table.</p>

      <!-- Filters -->
      <form method="post" class="row g-3 align-items-end">
        <div class="col-md-4">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{r}}" {% if r == selected_region %}selected{% endif %}>{{r}}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-md-4">
          <label class="form-label">Sub-Category</label>
          <select name="subcat" class="form-select">
            {% for s in subcats %}
              <option value="{{s}}" {% if s == selected_subcat %}selected{% endif %}>{{s}}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-md-4 d-flex gap-2">
          <button class="btn btn-primary ms-auto" type="submit">
            <i class="bi bi-arrow-repeat me-1"></i> Analyze
          </button>
          <a href="/roi" class="btn btn-outline-secondary">Reset</a>
        </div>
      </form>

      <!-- KPIs -->
      <div class="row g-3 mt-3">
        <div class="col-12 col-md-4">
          <div class="kpi">
            <h6>Total Sales</h6>
            <div class="val">${{ "{:,.2f}".format(kpi_total_sales) }}</div>
          </div>
        </div>
        <div class="col-12 col-md-4">
          <div class="kpi">
            <h6>Total Marketing Spend</h6>
            <div class="val">${{ "{:,.2f}".format(kpi_total_mkt) }}</div>
          </div>
        </div>
        <div class="col-12 col-md-4">
          <div class="kpi">
            <h6>ROI</h6>
            <div class="val">
              {% if kpi_roi_percent == kpi_roi_percent %}
                {{ "{:,.2f}".format(kpi_roi_percent) }}%
              {% else %} N/A {% endif %}
            </div>
          </div>
        </div>
      </div>

      <!-- Table + Chart -->
      {% if roi_table %}
        <hr class="my-4">
        <div class="row g-4">
          <div class="col-12 col-lg-6">
            <div class="table-responsive">
              <table class="table table-striped table-bordered">
                <thead class="table-light">
                  <tr>
                    {% for col in roi_table[0].keys() %}
                      <th>{{ col }}</th>
                    {% endfor %}
                  </tr>
                </thead>
                <tbody>
                  {% for row in roi_table %}
                    <tr>
                      {% for k, val in row.items() %}
                        <td>
                          {% if val is number %}
                            {{ "{:,.2f}".format(val) }}
                          {% else %}
                            {{ val }}
                          {% endif %}
                        </td>
                      {% endfor %}
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
          <div class="col-12 col-lg-6">
            {% if chart_b64 %}
              <img src="data:image/png;base64,{{ chart_b64 }}" class="img-fluid border rounded" alt="ROI chart">
            {% else %}
              <div class="alert alert-info">Single selection chosen — showing KPI summary and table only.</div>
            {% endif %}
          </div>
        </div>
      {% endif %}
    </div>
  </div>

  <div class="text-center text-muted small">Superstore Forecasting &amp; Analytics Portal © 2025</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    regions=regions,
    subcats=subcats,
    selected_region=selected_region,
    selected_subcat=selected_subcat,
    roi_table=roi_table,
    csv_b64=csv_b64,
    chart_b64=chart_b64,
    kpi_total_sales=kpi_total_sales,
    kpi_total_mkt=kpi_total_mkt,
    kpi_roi_percent=kpi_roi_percent,
    title_suffix=title_suffix)


@app.route("/recommend", methods=["GET", "POST"])
@login_required
@roles_required("analyst", "manager")
def recommend():
    import base64
    from io import StringIO
    import pandas as pd

    # ---- Load data ----
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back to Home</a>"

    required = {"Region", "Sub-Category", "Sales"}
    if not required.issubset(df.columns):
        return (
            "<p style='color:red'><b>Required columns missing.</b> "
            "Need 'Region', 'Sub-Category', 'Sales'.</p><a href='/'>Back to Home</a>"
        )

    # ---- Initial state ----
    regions = ["Central", "East", "South", "West"]
    selected_region = ""
    chart_base64 = None
    top_subcats = {}
    top_csv_b64 = None
    total_region_sales = None
    message = ""

    # ---- POST: compute results ----
    if request.method == "POST":
        selected_region = request.form.get("region", "")
        if selected_region not in regions:
            message = "Please choose a valid region."
        else:
            # Build chart
            chart_base64 = generate_region_chart_base64(selected_region, df)

            # Top 3 subcats by sales
            sub_df = df[df["Region"] == selected_region].copy()
            sub_df["Sales"] = pd.to_numeric(sub_df["Sales"], errors="coerce")
            sub_df = sub_df.dropna(subset=["Sales"])
            total_region_sales = float(sub_df["Sales"].sum())

            top_series = (
                sub_df.groupby("Sub-Category")["Sales"]
                      .sum()
                      .sort_values(ascending=False)
                      .head(3)
            )
            top_subcats = top_series.to_dict()

            # CSV download for top list
            if len(top_series) > 0:
                buf = StringIO()
                top_series.rename("Sales").reset_index().to_csv(buf, index=False)
                top_csv_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

    # ---- Render ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Recommendations by Region — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .bar {
      height: 8px; background: #e9ecef; border-radius: 999px; overflow: hidden;
    }
    .bar > span { display:block; height:100%; background: #0d6efd; }
    .kpi { background:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:.75rem 1rem; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-bullseye"></i> Product Recommendations by Region</h3>
      <p class="text-muted mb-3">Pick a region to see the top 3 sub-categories by total sales and a distribution chart.</p>

      <!-- Form -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Select Region</label>
          <select name="region" class="form-select" required>
            <option value="">Select Region</option>
            {% for r in regions %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-12 col-md-8">
          <label class="form-label d-block">Quick Select</label>
          <div class="btn-group" role="group" aria-label="Quick regions">
            {% for r in regions %}
              <button class="btn btn-outline-secondary btn-sm quickRegion" data-r="{{ r }}" type="button">{{ r }}</button>
            {% endfor %}
          </div>
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-primary">
            <i class="bi bi-magic"></i> Recommend
          </button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      {% if message %}
        <div class="alert alert-warning mt-3">{{ message }}</div>
      {% endif %}

      {% if top_subcats %}
        <hr class="my-4">

        <!-- KPIs -->
        <div class="row g-3 mb-3">
          <div class="col-sm-6 col-lg-4">
            <div class="kpi">
              <div class="text-muted">Region</div>
              <div class="fs-5 fw-semibold">{{ selected_region }}</div>
            </div>
          </div>
          <div class="col-sm-6 col-lg-4">
            <div class="kpi">
              <div class="text-muted">Top Picks</div>
              <div class="fs-5 fw-semibold">{{ top_subcats|length }}</div>
            </div>
          </div>
          <div class="col-sm-6 col-lg-4">
            <div class="kpi">
              <div class="text-muted">Total Region Sales</div>
              <div class="fs-5 fw-semibold">
                {% if total_region_sales is not none %}${{ "{:,.2f}".format(total_region_sales) }}{% else %}-{% endif %}
              </div>
            </div>
          </div>
        </div>

        <!-- Top 3 list + chart -->
        <div class="row g-4">
          <div class="col-lg-6">
            <h5 class="mb-3">Top 3 Sub-Categories</h5>
            <div class="list-group">
              {% set max_val = (top_subcats.values() | list | max) if top_subcats else 1 %}
              {% for subcat, sale in top_subcats.items() %}
                {% set pct = (sale / max_val * 100.0) if max_val else 0 %}
                <div class="list-group-item">
                  <div class="d-flex justify-content-between align-items-center">
                    <div class="fw-semibold">{{ subcat }}</div>
                    <div>${{ "{:,.2f}".format(sale) }}</div>
                  </div>
                  <div class="bar mt-2"><span style="width: {{ '%.0f'|format(pct) }}%"></span></div>
                </div>
              {% endfor %}
            </div>

            <div class="mt-3 d-flex gap-2">
              {% if top_csv_b64 %}
                <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ top_csv_b64 }}" download="top3_{{ selected_region|lower }}.csv">
                  ⬇️ Download CSV
                </a>
              {% endif %}
            </div>
          </div>

          <div class="col-lg-6">
            <h5 class="mb-3">Sales Distribution</h5>
            {% if chart_base64 %}
              <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Region Chart">
              <div class="mt-2">
                <a class="btn btn-sm btn-outline-secondary"
                   href="data:image/png;base64,{{ chart_base64 }}"
                   download="top_subcategories_{{ selected_region|lower }}.png">
                   ⬇️ Download PNG
                </a>
              </div>
            {% else %}
              <div class="alert alert-info">No chart available for this region.</div>
            {% endif %}
          </div>
        </div>
      {% endif %}
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<!-- Tiny JS: quick region buttons -->
<script>
(function () {
  document.querySelectorAll('.quickRegion').forEach(btn => {
    btn.addEventListener('click', () => {
      const sel = document.querySelector('select[name="region"]');
      sel.value = btn.dataset.r;
    });
  });
})();
</script>
</body>
</html>
    """,
    regions=regions,
    selected_region=selected_region,
    top_subcats=top_subcats,
    chart_base64=chart_base64,
    total_region_sales=total_region_sales,
    top_csv_b64=top_csv_b64,
    message=message)


@app.route("/sales-trends", methods=["GET", "POST"])
def sales_trends():
    import pandas as pd
    from io import BytesIO, StringIO
    import base64

    # ---- Load data safely ----
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back to Home</a>"

    required = {"Order Date", "Sales"}
    if not required.issubset(df.columns):
        return (
            "<p style='color:red'><b>Required columns missing.</b> "
            "Need 'Order Date' and 'Sales'.</p><a href='/'>Back to Home</a>"
        )

    # ---- Parse dates & numeric sales ----
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    # Region list (only if Region column exists)
    regions = []
    have_region = "Region" in df.columns
    if have_region:
        regions = sorted([r for r in df["Region"].dropna().unique()])
    regions_display = ["All"] + regions if have_region else ["All"]

    # ---- Read form selections ----
    selected_region = "All"
    sma_win = 0  # no smoothing by default
    if request.method == "POST":
        selected_region = request.form.get("region", "All")
        try:
            sma_win = int(request.form.get("sma", "0"))
            sma_win = sma_win if sma_win in (0, 3, 6, 12) else 0
        except:
            sma_win = 0

    # ---- Filter by region if applicable ----
    if have_region and selected_region != "All":
        df_plot = df[df["Region"] == selected_region].copy()
    else:
        df_plot = df.copy()

    # ---- Aggregate monthly ----
    if df_plot.empty:
        return (
            f"<p style='color:red'><b>No data"
            f"{' for region ' + selected_region if selected_region!='All' else ''}.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    df_plot["Month"] = df_plot["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df_plot.groupby("Month", as_index=False)["Sales"]
               .sum()
               .sort_values("Month")
    )

    # ---- Optional smoothing ----
    monthly["SMA"] = None
    if sma_win > 0 and len(monthly) >= sma_win:
        monthly["SMA"] = monthly["Sales"].rolling(window=sma_win, min_periods=sma_win).mean()

    # ---- Chart (fallback to helper if you prefer) ----
    # If you want to keep your original helper for the "All" view, uncomment:
    # chart_base64 = get_monthly_sales_chart(df_plot)  # if it already accounts for filtering
    # Otherwise generate here:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["Month"], monthly["Sales"], marker="o", label="Monthly Sales")
    if sma_win > 0 and monthly["SMA"].notna().any():
        plt.plot(monthly["Month"], monthly["SMA"], linestyle="--", label=f"{sma_win}-Month SMA")
    title_region = f" — {selected_region}" if have_region and selected_region != "All" else ""
    plt.title(f"Monthly Sales Trend{title_region}")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # ---- CSV download (Month, Sales, SMA if present) ----
    csv_df = monthly.copy()
    # prettier month label
    csv_df["Month"] = csv_df["Month"].dt.strftime("%Y-%m")
    out = StringIO()
    csv_df.to_csv(out, index=False)
    csv_b64 = base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    # ---- Build recent table (last 12 rows) ----
    recent = monthly.tail(12)
    table_rows = ""
    for _, r in recent.iterrows():
        month_label = r["Month"].strftime("%Y-%m")
        sales_val = r["Sales"]
        sma_val = r["SMA"] if pd.notna(r["SMA"]) else None
        table_rows += f"""
          <tr>
            <td>{month_label}</td>
            <td class="text-end">${sales_val:,.2f}</td>
            <td class="text-end">{('$' + format(sma_val, ',.2f')) if sma_val is not None else '-'}</td>
          </tr>
        """

    # ---- Render UI ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Monthly Sales Trend — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-calendar2-week"></i> Monthly Sales Trend</h3>
      <p class="text-muted mb-3">Visualize sales over time, optionally by region and with a rolling average.</p>

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions_display %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-4">
          <label class="form-label">Smoothing (Rolling Avg)</label>
          <select name="sma" class="form-select">
            <option value="0"  {% if sma_win == 0  %}selected{% endif %}>None</option>
            <option value="3"  {% if sma_win == 3  %}selected{% endif %}>3 months</option>
            <option value="6"  {% if sma_win == 6  %}selected{% endif %}>6 months</option>
            <option value="12" {% if sma_win == 12 %}selected{% endif %}>12 months</option>
          </select>
        </div>

        <div class="col-12 col-md-4">
          <button type="submit" class="btn btn-primary">
            <i class="bi bi-arrow-repeat"></i> Update
          </button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      <hr class="my-4">

      <!-- Chart + Downloads -->
      <div class="row g-4">
        <div class="col-lg-7">
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Monthly Sales Chart">
          <div class="mt-2 d-flex gap-2">
            <a class="btn btn-sm btn-outline-secondary"
               href="data:image/png;base64,{{ chart_base64 }}"
               download="monthly_sales_trend{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.png">
               ⬇️ Download PNG
            </a>
            <a class="btn btn-sm btn-outline-secondary"
               href="data:text/csv;base64,{{ csv_b64 }}"
               download="monthly_sales{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.csv">
               ⬇️ Download CSV
            </a>
          </div>
        </div>

        <div class="col-lg-5">
          <h5 class="mb-3">Recent Months</h5>
          <div class="table-responsive">
            <table class="table table-sm table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th>Month</th>
                  <th class="text-end">Sales</th>
                  <th class="text-end">SMA</th>
                </tr>
              </thead>
              <tbody>
                {{ table_rows|safe }}
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    chart_base64=chart_base64,
    csv_b64=csv_b64,
    regions_display=regions_display,
    selected_region=selected_region,
    sma_win=sma_win,
    table_rows=table_rows)



@app.route("/top-products", methods=["GET", "POST"])
def top_products():
    import pandas as pd
    from io import BytesIO, StringIO
    import base64

    # ---- Load data safely ----
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back to Home</a>"

    # ---- Validate required columns ----
    must_have_any = {"Sales", "Profit", "Quantity"}
    if "Sub-Category" not in df.columns or not any(c in df.columns for c in must_have_any):
        return (
            "<p style='color:red'><b>Required columns missing.</b> "
            "Need 'Sub-Category' and at least one of 'Sales', 'Profit', or 'Quantity'.</p>"
            "<a href='/'>Back to Home</a>"
        )

    # ---- Coerce numeric metrics if they exist ----
    for col in ["Sales", "Profit", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Region list (only if Region column exists)
    have_region = "Region" in df.columns
    regions_display = ["All"]
    if have_region:
        regions_display += sorted([r for r in df["Region"].dropna().unique()])

    # ---- Read form selections ----
    selected_region = "All"
    metric = "Sales" if "Sales" in df.columns else ("Profit" if "Profit" in df.columns else "Quantity")
    try:
        top_n = 10
        if request.method == "POST":
            selected_region = request.form.get("region", "All")
            metric = request.form.get("metric", metric)
            top_n = int(request.form.get("top_n", "10"))
            top_n = 5 if top_n < 5 else (25 if top_n > 25 else top_n)
    except:
        top_n = 10

    # ---- Filter by region if applicable ----
    df_plot = df.copy()
    if have_region and selected_region != "All":
        df_plot = df_plot[df_plot["Region"] == selected_region]

    if df_plot.empty:
        return (
            f"<p style='color:red'><b>No data"
            f"{' for region ' + selected_region if selected_region!='All' else ''}.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    # ---- Aggregate top N by Sub-Category on chosen metric ----
    if metric not in df_plot.columns:
        return (
            f"<p style='color:red'><b>Metric '{metric}' not found in data.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    grouped = (
        df_plot.groupby("Sub-Category", as_index=False)[metric]
               .sum()
               .sort_values(metric, ascending=False)
               .head(top_n)
    )

    if grouped.empty:
        return (
            "<p style='color:red'><b>No rows to aggregate for the selected filters.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    # ---- Chart (self-contained; no external helper needed) ----
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    # Sort again for nicer bar order left->right high->low
    grouped_sorted = grouped.sort_values(metric, ascending=True)
    plt.barh(grouped_sorted["Sub-Category"], grouped_sorted[metric])
    title_region = f" — {selected_region}" if have_region and selected_region != "All" else ""
    plt.title(f"Top {top_n} Sub-Categories by {metric}{title_region}")
    plt.xlabel(metric)
    plt.ylabel("Sub-Category")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # ---- CSV download for the table/chart ----
    out = StringIO()
    grouped.to_csv(out, index=False)
    csv_b64 = base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    # ---- Build table rows ----
    table_rows = ""
    for _, r in grouped.iterrows():
        val = r[metric]
        pretty_val = f"{val:,.0f}" if metric == "Quantity" else f"${val:,.2f}"
        table_rows += f"""
          <tr>
            <td>{r['Sub-Category']}</td>
            <td class="text-end">{pretty_val}</td>
          </tr>
        """

    # ---- Render UI ----
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top Sub-Categories — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-trophy"></i> Top Sub-Categories</h3>
      <p class="text-muted mb-3">See the top-performing sub-categories by Sales, Profit, or Quantity. Filter by region and choose how many to display.</p>

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Metric</label>
          <select name="metric" class="form-select">
            {% if 'Sales' in df.columns %}<option value="Sales"   {% if metric=='Sales'   %}selected{% endif %}>Sales</option>{% endif %}
            {% if 'Profit' in df.columns %}<option value="Profit" {% if metric=='Profit'  %}selected{% endif %}>Profit</option>{% endif %}
            {% if 'Quantity' in df.columns %}<option value="Quantity" {% if metric=='Quantity' %}selected{% endif %}>Quantity</option>{% endif %}
          </select>
        </div>

        <div class="col-sm-6 col-md-4">
          <label class="form-label">Top N (5–25)</label>
          <input type="number" name="top_n" class="form-control" min="5" max="25" value="{{ top_n }}">
        </div>

        <div class="col-sm-6 col-md-4">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions_display %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-primary">
            <i class="bi bi-arrow-repeat"></i> Update
          </button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      <hr class="my-4">

      <!-- Chart + Downloads -->
      <div class="row g-4">
        <div class="col-lg-7">
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Top Sub-Categories Chart">
          <div class="mt-2 d-flex gap-2">
            <a class="btn btn-sm btn-outline-secondary"
               href="data:image/png;base64,{{ chart_base64 }}"
               download="top_{{ top_n }}_subcategories_by_{{ metric|lower }}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.png">
               ⬇️ Download PNG
            </a>
            <a class="btn btn-sm btn-outline-secondary"
               href="data:text/csv;base64,{{ csv_b64 }}"
               download="top_{{ top_n }}_subcategories_by_{{ metric|lower }}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.csv">
               ⬇️ Download CSV
            </a>
          </div>
        </div>

        <div class="col-lg-5">
          <h5 class="mb-3">Ranking</h5>
          <div class="table-responsive">
            <table class="table table-sm table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th>Sub-Category</th>
                  <th class="text-end">{{ metric }}</th>
                </tr>
              </thead>
              <tbody>
                {{ table_rows|safe }}
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    df=df,  # for Jinja conditionals on available columns
    chart_base64=chart_base64,
    csv_b64=csv_b64,
    regions_display=regions_display,
    selected_region=selected_region,
    metric=metric,
    top_n=top_n,
    table_rows=table_rows)


@app.route("/profitability", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")
def profitability():
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO, StringIO
    import base64

    # ---------- Load data ----------
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back</a>"

    # ---------- Validate columns ----------
    col_subcat = "Sub-Category"
    col_region = "Region"
    col_sales  = "Sales"

    for c in [col_subcat, col_region, col_sales]:
        if c not in df.columns:
            return (f"<p style='color:red'><b>Missing column '{c}' in CSV.</b></p>"
                    "<a href='/data-diagnostics'>See Data Diagnostics</a>")

    # Coerce numeric
    df[col_sales] = pd.to_numeric(df[col_sales], errors="coerce")
    if "Profit" in df.columns:
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

    df = df.dropna(subset=[col_sales])

    # ---------- UI inputs ----------
    have_region = col_region in df.columns
    regions = sorted(df[col_region].dropna().unique().tolist()) if have_region else []
    margin_pct = 30.0
    region = ""
    top_n = 10
    use_actual_profit = False  # only meaningful if Profit exists

    if request.method == "POST":
        # Margin %
        try:
            margin_pct = float(request.form.get("margin_pct", "30"))
        except:
            margin_pct = 30.0

        # Region filter
        region = request.form.get("region") or ""

        # Top N clamp
        try:
            top_n = int(request.form.get("top_n", "10"))
        except:
            top_n = 10
        top_n = max(5, min(25, top_n))

        # Actual profit toggle
        use_actual_profit = request.form.get("use_actual_profit") == "on" and ("Profit" in df.columns)

    # ---------- Filter by region ----------
    df_plot = df.copy()
    if have_region and region:
        df_plot = df_plot[df_plot[col_region] == region]

    if df_plot.empty:
        return (
            f"<p style='color:red'><b>No data"
            f"{' for region ' + region if region else ''}.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    # ---------- Compute profitability ----------
    if use_actual_profit:
        profit_col = "Profit"
        df_plot = df_plot.dropna(subset=[profit_col])
        df_plot = df_plot.assign(ProfitToUse=df_plot[profit_col])
        info_banner = None  # no info banner when using actual profit
        metric_label = "Actual Profit"
        file_suffix_metric = "actual_profit"
    else:
        # Estimated Profit = Sales × Margin%
        df_plot = df_plot.assign(ProfitToUse=df_plot[col_sales] * (margin_pct / 100.0))
        info_banner = "No Profit column used or toggle off. Using Estimated Profit = Sales × Margin%."
        metric_label = f"Estimated Profit (Margin {margin_pct:.0f}%)"
        file_suffix_metric = "estimated_profit"

    # ---------- Aggregate ----------
    agg = (
        df_plot.groupby(col_subcat, as_index=False)["ProfitToUse"]
               .sum()
               .sort_values("ProfitToUse", ascending=False)
    )

    if agg.empty:
        return "<p>No data to display after filters.</p><a href='/'>Back</a>"

    top_df = agg.head(top_n).copy()
    bottom_df = agg.tail(top_n).copy()

    # ---------- Plot (single image with two panels) ----------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    plt.figure(figsize=(11, 7))

    # Top N
    plt.subplot(2, 1, 1)
    top_sorted = top_df.sort_values("ProfitToUse", ascending=True)
    plt.barh(top_sorted[col_subcat], top_sorted["ProfitToUse"])
    title_region = f" — {region}" if region else ""
    plt.title(f"Top {top_n} by {metric_label}{title_region}")
    plt.xlabel(metric_label)
    plt.ylabel("Sub-Category")

    # Bottom N
    plt.subplot(2, 1, 2)
    bottom_sorted = bottom_df.sort_values("ProfitToUse", ascending=True)
    plt.barh(bottom_sorted[col_subcat], bottom_sorted["ProfitToUse"])
    plt.title(f"Bottom {top_n} by {metric_label}{title_region}")
    plt.xlabel(metric_label)
    plt.ylabel("Sub-Category")

    plt.tight_layout()
    chart_base64 = fig_to_b64()
    plt.close()

    # ---------- CSVs for download ----------
    def df_to_csv_b64(xdf):
        out = StringIO()
        nice = xdf.rename(columns={"ProfitToUse": "Profit"})
        nice.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    top_csv_b64 = df_to_csv_b64(top_df)
    bottom_csv_b64 = df_to_csv_b64(bottom_df)

    # ---------- Build table rows ----------
    def build_rows(xdf):
        rows = ""
        for _, r in xdf.iterrows():
            rows += f"""
              <tr>
                <td>{r[col_subcat]}</td>
                <td class="text-end">${r['ProfitToUse']:,.2f}</td>
              </tr>
            """
        return rows

    top_rows = build_rows(top_df)
    bottom_rows = build_rows(bottom_df)

    # ---------- Render UI ----------
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Profitability — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-cash-coin"></i> Profitability Dashboard</h3>
      <p class="text-muted mb-3">Rank sub-categories by profitability. Use actual Profit if present or estimate using Sales × Margin%.</p>

      {% if info_banner %}
      <div class="alert alert-info">
        {{ info_banner }}
      </div>
      {% endif %}

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Margin % (global)</label>
          <input type="number" step="1" min="0" max="100" name="margin_pct"
                 value="{{ margin_pct }}" class="form-control">
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Top/Bottom N (5–25)</label>
          <input type="number" name="top_n" class="form-control" min="5" max="25" value="{{ top_n }}">
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Region (optional)</label>
          <select name="region" class="form-select">
            <option value="">All Regions</option>
            {% for r in regions %}
              <option value="{{ r }}" {% if r == region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <div class="form-check mt-4">
            <input class="form-check-input" type="checkbox" name="use_actual_profit" id="uap"
                   {% if use_actual_profit %}checked{% endif %}
                   {% if not df_has_profit %}disabled{% endif %}>
            <label class="form-check-label" for="uap">
              Use actual Profit{% if not df_has_profit %} (not available){% endif %}
            </label>
          </div>
        </div>

        <div class="col-12">
          <button class="btn btn-primary" type="submit"><i class="bi bi-arrow-repeat"></i> Apply</button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      <hr class="my-4">

      <!-- Chart + Downloads -->
      <div class="row g-4">
        <div class="col-lg-7">
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Profitability Chart">
          <div class="mt-2 d-flex flex-wrap gap-2">
            <a class="btn btn-sm btn-outline-secondary"
               href="data:image/png;base64,{{ chart_base64 }}"
               download="profitability_{{ file_suffix_metric }}{% if region %}_{{ region|lower }}{% endif %}.png">
               ⬇️ Download PNG
            </a>
            <a class="btn btn-sm btn-outline-secondary"
               href="data:text/csv;base64,{{ top_csv_b64 }}"
               download="top_{{ top_n }}_{{ file_suffix_metric }}{% if region %}_{{ region|lower }}{% endif %}.csv">
               ⬇️ Download Top {{ top_n }} CSV
            </a>
            <a class="btn btn-sm btn-outline-secondary"
               href="data:text/csv;base64,{{ bottom_csv_b64 }}"
               download="bottom_{{ top_n }}_{{ file_suffix_metric }}{% if region %}_{{ region|lower }}{% endif %}.csv">
               ⬇️ Download Bottom {{ top_n }} CSV
            </a>
          </div>
        </div>

        <div class="col-lg-5">
          <div class="row g-4">
            <div class="col-12">
              <h5 class="mb-3">Top {{ top_n }} — {{ metric_label }}</h5>
              <div class="table-responsive">
                <table class="table table-sm table-hover align-middle">
                  <thead class="table-light">
                    <tr>
                      <th>Sub-Category</th>
                      <th class="text-end">Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {{ top_rows|safe }}
                  </tbody>
                </table>
              </div>
            </div>
            <div class="col-12">
              <h5 class="mb-3">Bottom {{ top_n }} — {{ metric_label }}</h5>
              <div class="table-responsive">
                <table class="table table-sm table-hover align-middle">
                  <thead class="table-light">
                    <tr>
                      <th>Sub-Category</th>
                      <th class="text-end">Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {{ bottom_rows|safe }}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

      </div>

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    # Jinja variables
    regions=regions,
    region=region,
    margin_pct=margin_pct,
    top_n=top_n,
    use_actual_profit=use_actual_profit,
    df_has_profit=("Profit" in df.columns),
    info_banner=info_banner,
    metric_label=metric_label,
    chart_base64=chart_base64,
    file_suffix_metric=file_suffix_metric,
    top_rows=top_rows,
    bottom_rows=bottom_rows,
    top_csv_b64=top_csv_b64,
    bottom_csv_b64=bottom_csv_b64)



@app.route("/kpi-tracker", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")
def kpi_tracker():
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO, StringIO
    import base64

    # ---------- Helpers ----------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def df_to_csv_b64(xdf):
        out = StringIO()
        xdf.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    # ---------- Load data ----------
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back</a>"

    required = {"Order Date", "Sales"}
    if not required.issubset(df.columns):
        return "<p style='color:red'><b>Need columns: Order Date, Sales.</b></p><a href='/'>Back</a>"

    # Basic cleaning
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    have_region = "Region" in df.columns
    regions = ["All"] + (sorted(df["Region"].dropna().unique().tolist()) if have_region else [])

    # ---------- Read UI selections ----------
    # Defaults
    selected_region = "All"
    target_mode = "auto"  # 'auto' or 'file'
    uplift_pct = 5.0       # auto-target uplift above 3M rolling avg
    sma_win = 0            # 0, 3, 6, 12

    if request.method == "POST":
        selected_region = request.form.get("region", "All")
        target_mode = request.form.get("target_mode", "auto")
        try:
            uplift_pct = float(request.form.get("uplift_pct", "5"))
        except:
            uplift_pct = 5.0
        try:
            sma_win = int(request.form.get("sma", "0"))
            sma_win = sma_win if sma_win in (0, 3, 6, 12) else 0
        except:
            sma_win = 0

    # ---------- Filter by region ----------
    if have_region and selected_region != "All":
        df = df[df["Region"] == selected_region]

    if df.empty:
        return (
            f"<p style='color:red'><b>No data"
            f"{' for region ' + selected_region if have_region and selected_region!='All' else ''}.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    # ---------- Monthly aggregation ----------
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")

    # ---------- Targets ----------
    targets_loaded = None
    target_note = ""
    if target_mode == "file":
        try:
            tdf = pd.read_csv("sales_targets.csv")
            tdf["Month"] = pd.to_datetime(tdf["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            tdf["Target"] = pd.to_numeric(tdf["Target"], errors="coerce")
            tdf = tdf.dropna(subset=["Month", "Target"])
            targets_loaded = tdf
            target_note = "Targets loaded from sales_targets.csv."
        except Exception:
            # Fall back to auto if file missing/invalid
            target_mode = "auto"
            target_note = "Could not read sales_targets.csv — using Auto targets instead."

    if target_mode == "file" and targets_loaded is not None:
        monthly = monthly.merge(targets_loaded, on="Month", how="left")
        # Fill any missing targets with auto uplift
        mfill = monthly["Sales"].rolling(3, min_periods=1).mean() * (1 + uplift_pct/100.0)
        monthly["Target"] = monthly["Target"].fillna(mfill.round(2))
    else:
        # Auto target = 3M rolling avg × (1 + uplift)
        rolling = monthly["Sales"].rolling(3, min_periods=1).mean()
        monthly["Target"] = (rolling * (1 + uplift_pct/100.0)).round(2)
        if not target_note:
            target_note = f"Auto targets = 3-month rolling average × (1 + {uplift_pct:.0f}%)."

    # ---------- KPI calculations ----------
    monthly["Variance"] = monthly["Sales"] - monthly["Target"]
    # Hit rate: months where Sales >= Target
    hit_rate = 0.0
    if len(monthly) > 0:
        hit_rate = (monthly["Sales"] >= monthly["Target"]).mean() * 100.0

    # YTD (calendar year of latest month)
    if len(monthly) > 0:
        latest_month = monthly["Month"].max()
        ytd_mask = monthly["Month"].dt.year == latest_month.year
        ytd_actual = monthly.loc[ytd_mask, "Sales"].sum()
        ytd_target = monthly.loc[ytd_mask, "Target"].sum()
        ytd_var = ytd_actual - ytd_target
    else:
        ytd_actual = ytd_target = ytd_var = 0.0
        latest_month = pd.NaT

    # Last month variance
    last_actual = last_target = last_var = 0.0
    if len(monthly) > 0:
        lm = monthly.iloc[-1]
        last_actual = float(lm["Sales"])
        last_target = float(lm["Target"])
        last_var = float(lm["Variance"])

    # ---------- Optional smoothing for chart ----------
    monthly["SMA"] = None
    if sma_win > 0 and len(monthly) >= sma_win:
        monthly["SMA"] = monthly["Sales"].rolling(window=sma_win, min_periods=sma_win).mean()

    # ---------- Plot chart ----------
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["Month"], monthly["Sales"], marker="o", label="Actual Sales")
    plt.plot(monthly["Month"], monthly["Target"], marker="o", linestyle="--", label="Target")
    if sma_win > 0 and monthly["SMA"].notna().any():
        plt.plot(monthly["Month"], monthly["SMA"], linestyle=":", label=f"{sma_win}-Month SMA")
    title_region = f" — {selected_region}" if have_region and selected_region != "All" else ""
    plt.title(f"Sales vs Target (Monthly){title_region}")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    chart_base64 = fig_to_b64()
    plt.close()

    # ---------- Build recent table (last 12 rows) ----------
    recent = monthly.tail(12).copy()
    recent["MonthLabel"] = recent["Month"].dt.strftime("%Y-%m")
    table_rows = ""
    for _, r in recent.iterrows():
        var = float(r["Variance"])
        cls = "text-success" if var >= 0 else "text-danger"
        table_rows += f"""
          <tr>
            <td>{r['MonthLabel']}</td>
            <td class="text-end">${r['Sales']:,.2f}</td>
            <td class="text-end">${r['Target']:,.2f}</td>
            <td class="text-end {cls}">{'+' if var>=0 else ''}${var:,.2f}</td>
          </tr>
        """

    # ---------- Downloads ----------
    # Pretty export of the whole monthly table
    export_df = monthly[["Month", "Sales", "Target", "Variance"]].copy()
    export_df["Month"] = export_df["Month"].dt.strftime("%Y-%m")
    csv_b64 = df_to_csv_b64(export_df)

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>KPI Tracker — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .kpi { border-radius: var(--card-radius); }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-bullseye"></i> KPI Tracker — Sales vs Target</h3>
      <p class="text-muted mb-3">Compare monthly sales against targets. Choose region, target mode, uplift, and optional smoothing.</p>

      {% if target_note %}
      <div class="alert alert-info">{{ target_note }}</div>
      {% endif %}

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Target Mode</label>
          <select name="target_mode" class="form-select">
            <option value="auto" {% if target_mode=='auto' %}selected{% endif %}>Auto (3M avg × uplift)</option>
            <option value="file" {% if target_mode=='file' %}selected{% endif %}>From CSV (sales_targets.csv)</option>
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Uplift % (Auto mode)</label>
          <input type="number" step="1" min="0" max="50" name="uplift_pct" value="{{ uplift_pct }}" class="form-control">
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Smoothing (SMA)</label>
          <select name="sma" class="form-select">
            <option value="0"  {% if sma_win == 0  %}selected{% endif %}>None</option>
            <option value="3"  {% if sma_win == 3  %}selected{% endif %}>3 months</option>
            <option value="6"  {% if sma_win == 6  %}selected{% endif %}>6 months</option>
            <option value="12" {% if sma_win == 12 %}selected{% endif %}>12 months</option>
          </select>
        </div>

        <div class="col-12">
          <button class="btn btn-primary" type="submit"><i class="bi bi-arrow-repeat"></i> Update</button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      <!-- KPI Cards -->
      <div class="row g-3 mt-3">
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">Hit Rate</div>
            <div class="fs-4 fw-semibold">{{ '%.1f'|format(hit_rate) }}%</div>
            <div class="small text-muted">Share of months meeting/exceeding target</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">YTD Actual vs Target</div>
            <div class="fs-6">Actual: <b>${{ '{:,.2f}'.format(ytd_actual) }}</b></div>
            <div class="fs-6">Target: <b>${{ '{:,.2f}'.format(ytd_target) }}</b></div>
            <div class="small {% if ytd_var>=0 %}text-success{% else %}text-danger{% endif %}">
              Var: {{ '+' if ytd_var>=0 else '' }}${{ '{:,.2f}'.format(ytd_var) }}
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="p-3 bg-white border kpi">
            <div class="text-muted small">Last Month Variance</div>
            <div class="fs-5 {% if last_var>=0 %}text-success{% else %}text-danger{% endif %}">
              {{ '+' if last_var>=0 else '' }}${{ '{:,.2f}'.format(last_var) }}
            </div>
            <div class="small text-muted">Actual ${{ '{:,.2f}'.format(last_actual) }} vs Target ${{ '{:,.2f}'.format(last_target) }}</div>
          </div>
        </div>
      </div>

      <hr class="my-4">

      <!-- Chart + Downloads -->
      <div class="row g-4">
        <div class="col-lg-7">
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Sales vs Target Chart">
          <div class="mt-2 d-flex gap-2 flex-wrap">
            <a class="btn btn-sm btn-outline-secondary"
               href="data:image/png;base64,{{ chart_base64 }}"
               download="kpi_sales_vs_target{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.png">
               ⬇️ Download PNG
            </a>
            <a class="btn btn-sm btn-outline-secondary"
               href="data:text/csv;base64,{{ csv_b64 }}"
               download="kpi_sales_vs_target{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.csv">
               ⬇️ Download CSV
            </a>
          </div>
        </div>

        <div class="col-lg-5">
          <h5 class="mb-3">Recent 12 Months</h5>
          <div class="table-responsive">
            <table class="table table-sm table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th>Month</th>
                  <th class="text-end">Actual</th>
                  <th class="text-end">Target</th>
                  <th class="text-end">Variance</th>
                </tr>
              </thead>
              <tbody>
                {{ table_rows|safe }}
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    # Jinja
    regions=regions,
    selected_region=selected_region,
    target_mode=target_mode,
    uplift_pct=uplift_pct,
    sma_win=sma_win,
    target_note=target_note,
    hit_rate=hit_rate,
    ytd_actual=ytd_actual,
    ytd_target=ytd_target,
    ytd_var=ytd_var,
    last_var=last_var,
    last_actual=last_actual,
    last_target=last_target,
    chart_base64=chart_base64,
    csv_b64=csv_b64,
    table_rows=table_rows)

@app.route("/manager-dashboard")
@login_required
@roles_required("manager")
def manager_dashboard():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # ---------- helpers ----------
    def fig_to_base64():
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        out = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return out

    def safe_div(numer, denom):
        return float(numer / denom) if denom and denom != 0 else np.nan

    # ---------- data (ROI) ----------
    roi_error = None
    total_sales = total_mkt = 0.0
    roi_overall = np.nan
    best_region = None
    worst_region = None
    chart_b64 = None
    top5 = []

    try:
        df = pd.read_csv(DATA_CSV)

        # Ensure required columns exist
        for col in ["Sales", "Marketing Spend"]:
            if col not in df.columns:
                raise KeyError(f"Missing expected column: '{col}' in superstore_extended.csv")

        # Coerce types
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)
        df["Marketing Spend"] = pd.to_numeric(df["Marketing Spend"], errors="coerce").fillna(0.0)
        if "Region" not in df.columns:
            df["Region"] = "All"

        # Overall KPIs
        total_sales = float(df["Sales"].sum())
        total_mkt   = float(df["Marketing Spend"].sum())
        roi_overall = safe_div(total_sales - total_mkt, total_mkt) * 100.0

        # ROI by Region
        reg = (
            df.groupby("Region", dropna=False)
              .agg(Total_Sales=("Sales","sum"), Total_Marketing=("Marketing Spend","sum"))
              .reset_index()
        )
        reg["ROI (%)"] = reg.apply(
            lambda r: safe_div(r["Total_Sales"] - r["Total_Marketing"], r["Total_Marketing"]) * 100.0,
            axis=1
        )

        reg_nonan = reg.dropna(subset=["ROI (%)"]).copy()

        # Best / Worst region
        if not reg_nonan.empty:
            best_row = reg_nonan.sort_values("ROI (%)", ascending=False).iloc[0]
            worst_row = reg_nonan.sort_values("ROI (%)", ascending=True).iloc[0]
            best_region = {"Region": str(best_row["Region"]), "ROI": float(best_row["ROI (%)"])}
            worst_region = {"Region": str(worst_row["Region"]), "ROI": float(worst_row["ROI (%)"])}

            # Mini chart
            plt.figure(figsize=(7.5, 4))
            plt.bar(reg_nonan["Region"], reg_nonan["ROI (%)"])
            plt.title("Marketing ROI by Region")
            plt.ylabel("ROI (%)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            chart_b64 = fig_to_base64()

            # Top 5 ROI regions
            top5 = (
                reg_nonan.sort_values("ROI (%)", ascending=False)
                         .head(5)
                         .assign(**{
                             "Total_Sales": lambda d: d["Total_Sales"].round(2),
                             "Total_Marketing": lambda d: d["Total_Marketing"].round(2),
                             "ROI (%)": lambda d: d["ROI (%)"].round(2),
                         })
                         .to_dict(orient="records")
            )
    except Exception as e:
        roi_error = str(e)

    # ---------- UI ----------
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Manager Dashboard — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .btn { border-radius: 10px; }
    .lead-small { font-size:.98rem; color:#6c757d; }
    .kpi { padding:14px; border-radius:12px; background:#fff; border:1px solid #eee; }
    .kpi h6 { margin:0; color:#6c757d; }
    .kpi .val { font-size:1.25rem; font-weight:700; }
    .table-sm th, .table-sm td { padding: .45rem .6rem; }
  </style>
</head>
<body>

<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
    <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
  </div>
</nav>

<div class="container">
  <h2 class="mb-2">👤 Manager Dashboard</h2>
  <p class="lead-small mb-4">Decision-support tools for profitability, pricing, KPIs, and inventory risk.</p>

  <!-- Tabs -->
  <ul class="nav nav-tabs" id="mgrTabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="dashboards-tab" data-bs-toggle="tab" data-bs-target="#dashboards" type="button" role="tab" aria-controls="dashboards" aria-selected="true">
        <i class="bi bi-layout-text-window-reverse me-1"></i> Dashboards
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="pricing-tab" data-bs-toggle="tab" data-bs-target="#pricing" type="button" role="tab" aria-controls="pricing" aria-selected="false">
        <i class="bi bi-tags me-1"></i> Pricing & Discount
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="inventory-tab" data-bs-toggle="tab" data-bs-target="#inventory" type="button" role="tab" aria-controls="inventory" aria-selected="false">
        <i class="bi bi-box-seam me-1"></i> Inventory
      </button>
    </li>
  </ul>

  <div class="tab-content pt-3">
    <!-- Dashboards -->
    <div class="tab-pane fade show active" id="dashboards" role="tabpanel" aria-labelledby="dashboards-tab">
      <div class="row g-4">
        <!-- NEW: Marketing ROI Quick Card -->
        <div class="col-lg-12">
          <div class="card h-100">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center mb-2">
                <h5 class="section-title mb-0"><i class="bi bi-cash-coin"></i> Marketing ROI (Quick View)</h5>
                <div>
                  <a class="btn btn-sm btn-outline-primary" href="/roi">
                    <i class="bi bi-cash-coin me-1"></i> Full ROI Analysis
                  </a>
                </div>
              </div>

              {% if roi_error %}
                <div class="alert alert-danger mb-0"><b>ROI Error:</b> {{ roi_error }}</div>
              {% else %}
                <!-- KPIs -->
                <div class="row g-3 mb-3">
                  <div class="col-12 col-md-4">
                    <div class="kpi">
                      <h6>Total Sales</h6>
                      <div class="val">${{ "{:,.2f}".format(total_sales) }}</div>
                    </div>
                  </div>
                  <div class="col-12 col-md-4">
                    <div class="kpi">
                      <h6>Total Marketing Spend</h6>
                      <div class="val">${{ "{:,.2f}".format(total_mkt) }}</div>
                    </div>
                  </div>
                  <div class="col-12 col-md-4">
                    <div class="kpi">
                      <h6>Overall ROI</h6>
                      <div class="val">
                        {% if roi_overall == roi_overall %}
                          {{ "{:,.2f}".format(roi_overall) }}%
                        {% else %} N/A {% endif %}
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Best / Worst badges -->
                <div class="d-flex flex-wrap gap-3 mb-3">
                  {% if best_region %}
                    <span class="badge text-bg-success p-2">
                      <i class="bi bi-emoji-smile me-1"></i>
                      Best: {{ best_region.Region }} — {{ "{:,.2f}".format(best_region.ROI) }}%
                    </span>
                  {% endif %}
                  {% if worst_region %}
                    <span class="badge text-bg-danger p-2">
                      <i class="bi bi-emoji-frown me-1"></i>
                      Worst: {{ worst_region.Region }} — {{ "{:,.2f}".format(worst_region.ROI) }}%
                    </span>
                  {% endif %}
                </div>

                <!-- Chart + Table -->
                <div class="row g-3">
                  <div class="col-12 col-lg-7">
                    {% if chart_b64 %}
                      <img src="data:image/png;base64,{{ chart_b64 }}" class="img-fluid border rounded" alt="ROI by Region">
                    {% else %}
                      <div class="alert alert-warning mb-0">Not enough data to render ROI chart.</div>
                    {% endif %}
                  </div>
                  <div class="col-12 col-lg-5">
                    {% if top5 %}
                      <div class="table-responsive">
                        <table class="table table-sm table-striped table-bordered bg-white">
                          <thead class="table-light">
                            <tr>
                              <th>Region</th>
                              <th>Total Sales</th>
                              <th>Total Marketing</th>
                              <th>ROI (%)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {% for row in top5 %}
                              <tr>
                                <td>{{ row["Region"] }}</td>
                                <td>${{ "{:,.2f}".format(row["Total_Sales"]) }}</td>
                                <td>${{ "{:,.2f}".format(row["Total_Marketing"]) }}</td>
                                <td>{{ "{:,.2f}".format(row["ROI (%)"]) }}</td>
                              </tr>
                            {% endfor %}
                          </tbody>
                        </table>
                      </div>
                    {% else %}
                      <div class="alert alert-info mb-0">No ROI rows to display.</div>
                    {% endif %}
                  </div>
                </div>
              {% endif %}

              <div class="text-end mt-3">
                <a class="btn btn-outline-primary" href="/roi">
                  Go to Detailed ROI <i class="bi bi-arrow-right-short"></i>
                </a>
              </div>
            </div>
          </div>
        </div>

        <!-- Your existing cards -->
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body d-flex flex-column">
              <h5 class="section-title"><i class="bi bi-cash-coin"></i> Profitability (Estimated)</h5>
              <p class="text-muted mb-4">Estimate profit by applying a global margin% over Sales. Filter by region and export charts.</p>
              <div class="mt-auto">
                <a href="/profitability" class="btn btn-primary">
                  <i class="bi bi-graph-up-arrow me-1"></i> Open Profitability
                </a>
              </div>
            </div>
          </div>
        </div>
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body d-flex flex-column">
              <h5 class="section-title"><i class="bi bi-bullseye"></i> KPI Tracker (Sales vs Target)</h5>
              <p class="text-muted mb-4">Track monthly Sales vs manager-set targets; highlight gaps and momentum.</p>
              <div class="mt-auto">
                <a href="/kpi-tracker" class="btn btn-primary">
                  <i class="bi bi-speedometer2 me-1"></i> Open KPI Tracker
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>      
    </div>

    <!-- Pricing & Discount -->
    <div class="tab-pane fade" id="pricing" role="tabpanel" aria-labelledby="pricing-tab">
      <div class="row g-4">
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-body d-flex flex-column">
              <h5 class="section-title"><i class="bi bi-tag"></i> Discount Impact</h5>
              <p class="text-muted mb-4">Visualize relationship between Discount and Sales. Filter by region and sub-category; export PNG.</p>
              <div class="mt-auto">
                <a href="/discount-impact" class="btn btn-primary">
                  <i class="bi bi-scatter-chart me-1"></i> Open Discount Impact
                </a>
              </div>
            </div>
          </div>
        </div>
        <!-- (Optional) Add “Price Elasticity (beta)” later -->
      </div>
    </div>

    <!-- Inventory -->
    <div class="tab-pane fade" id="inventory" role="tabpanel" aria-labelledby="inventory-tab">
      <div class="row g-4">
        <div class="col-lg-12">
          <div class="card h-100">
            <div class="card-body d-flex flex-column">
              <h5 class="section-title"><i class="bi bi-box-seam"></i> Stock Alert Simulation</h5>
              <p class="text-muted mb-4">Monte Carlo inventory simulation with service level & lead time; visualize stockout risk and export PNG/CSV.</p>
              <div class="mt-auto">
                <a href="/stock-alert" class="btn btn-primary">
                  <i class="bi bi-lightning-charge me-1"></i> Open Simulation
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>      
    </div>
  </div>

  <footer class="mt-5 text-center small">
    Superstore Forecasting &amp; Analytics Portal © 2025 · <span class="text-muted">Manager Suite</span>
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    roi_error=roi_error,
    total_sales=total_sales,
    total_mkt=total_mkt,
    roi_overall=roi_overall,
    best_region=best_region,
    worst_region=worst_region,
    top5=top5,
    chart_b64=chart_b64)



# --- helper you can keep near the top of your file (reused by route) ---
def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


@app.route("/discount-impact", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")
def discount_impact():
    # ---- imports kept INSIDE so names are always defined ----
    import base64, numpy as np, pandas as pd, matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    from io import BytesIO, StringIO
    from flask import render_template_string, request, url_for
    from flask_login import current_user

    # ---- tiny helpers ----
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        out = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return out

    def df_to_csv_b64(xdf: pd.DataFrame):
        out = StringIO()
        xdf.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    def safe_url(endpoint, default="home"):
        try:
            return url_for(endpoint)
        except Exception:
            return url_for(default)

    # ---- load ----
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='{url_for('home')}'>Back</a>"

    col_discount, col_sales, col_region, col_subcat = "Discount", "Sales", "Region", "Sub-Category"
    for c in (col_discount, col_sales, col_region, col_subcat):
        if c not in df.columns:
            return (f"<p style='color:red'><b>Missing column '{c}' in CSV.</b></p>"
                    f"<a href='{safe_url('data_diagnostics')}'>See Data Diagnostics</a>")

    # ---- clean ----
    df[col_discount] = pd.to_numeric(df[col_discount], errors="coerce")
    df[col_sales]    = pd.to_numeric(df[col_sales], errors="coerce")
    df = df.dropna(subset=[col_discount, col_sales])

    regions = ["All"] + sorted(df[col_region].dropna().unique().tolist())
    subcats = ["All"] + sorted(df[col_subcat].dropna().unique().tolist())

    # ---- role flags ----
    role = str(getattr(current_user, "role", "manager")).lower()
    is_manager = role == "manager"
    is_analyst = role == "analyst"

    # ---- selections ----
    selected_region = request.form.get("region", "All")
    selected_subcat = request.form.get("subcategory", "All")
    trendline_sel   = request.form.get("trendline", "auto") if is_analyst else "auto"
    bins_mode_sel   = request.form.get("bins_mode", "auto") if is_analyst else "none"
    do_sma          = (request.form.get("sma") == "on") if is_analyst else False

    # ---- filter ----
    df_plot = df.copy()
    if selected_region != "All":
        df_plot = df_plot[df_plot[col_region] == selected_region]
    if selected_subcat != "All":
        df_plot = df_plot[df_plot[col_subcat] == selected_subcat]
    if df_plot.empty:
        return f"<p>No data after filters.</p><a href='{url_for('home')}'>Back</a>"

    x = pd.to_numeric(df_plot[col_discount], errors="coerce").astype(float).values
    y = pd.to_numeric(df_plot[col_sales],    errors="coerce").astype(float).values
    n = int(len(df_plot))
    avg_discount = float(np.nanmean(x)) if n else float("nan")

    # ---------- ANALYST KPIs: correlation + linear regression ----------
    if n > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = float("nan")
    r2 = (r**2) if np.isfinite(r) else float("nan")

    slope = intercept = float("nan")
    if n >= 2 and np.unique(x[~np.isnan(x)]).size >= 2:
        try:
            z = np.polyfit(x, y, 1)  # y = slope*x + intercept
            slope, intercept = float(z[0]), float(z[1])
        except Exception:
            pass
    reg_eq = f"Sales = {intercept:,.2f} + {slope:,.2f}·Discount" if np.isfinite(slope) and np.isfinite(intercept) else "—"
    abs_r = abs(r) if np.isfinite(r) else 0.0

    # ---------- MANAGER sweet spot summary ----------
    try:
        df_plot["_bin_m"] = pd.qcut(df_plot[col_discount], q=5, duplicates="drop")
    except Exception:
        df_plot["_bin_m"] = pd.qcut(df_plot[col_discount], q=4, duplicates="drop")

    band_df = (df_plot.groupby("_bin_m", observed=True)
               .agg(AvgDiscount=(col_discount, "mean"),
                    AvgSales=(col_sales, "mean"),
                    Orders=(col_sales, "size"))
               .reset_index()
               .sort_values("AvgDiscount", kind="stable"))

    min_orders = max(30, int(0.03 * n))
    cand = band_df[band_df["Orders"] >= min_orders]
    sweet = cand.sort_values("AvgSales", ascending=False).head(1) if not cand.empty else band_df.head(1)

    if not sweet.empty:
        sweet_band     = str(sweet.iloc[0]["_bin_m"])
        sweet_avg_disc = float(sweet.iloc[0]["AvgDiscount"])
        sweet_avg_sale = float(sweet.iloc[0]["AvgSales"])
        sweet_orders   = int(sweet.iloc[0]["Orders"])
        coverage_pct   = 100.0 * sweet_orders / n
    else:
        sweet_band, sweet_avg_disc, sweet_avg_sale, sweet_orders, coverage_pct = "—", np.nan, np.nan, 0, 0.0

    if abs_r < 0.10:
        verdict, tone = "Weak effect", "secondary"
        msg = "Discounts have little impact here. Focus on product mix, seasonality and campaigns."
    elif abs_r < 0.25:
        verdict, tone = "Moderate effect", "warning"
        msg = "Discounts influence sales somewhat. Use targeted discounts near the sweet spot."
    else:
        verdict, tone = "Strong effect", "success"
        msg = "Discounts strongly move sales. Manage discount levels carefully to protect margins."

    top_band = band_df.tail(1)
    high_disc_avg = float(top_band["AvgDiscount"].iloc[0]) if not top_band.empty else np.nan
    margin_risk = (high_disc_avg >= 0.35)

    # ---------- charts ----------
    title_bits = []
    if selected_region != "All": title_bits.append(selected_region)
    if selected_subcat != "All": title_bits.append(selected_subcat)
    title_tag = " — ".join(title_bits)

    plt.figure(figsize=(10, 5))
    y_plot = np.clip(y, None, np.nanpercentile(y, 99.5)) if n >= 20 else y
    plt.scatter(x, y_plot, alpha=0.35, label="Orders")
    plt.title("Discount vs Sales" + (f" — {title_tag}" if title_tag else ""))
    plt.xlabel("Discount"); plt.ylabel("Sales"); plt.tight_layout()
    base_chart_b64 = fig_to_b64()

    trendline_eff = trendline_sel
    if trendline_eff == "auto":
        trendline_eff = "none" if (not np.isfinite(r) or abs_r < 0.20 or n < 100) else "linear"

    analyst_chart_b64 = base_chart_b64
    if is_analyst and trendline_eff in ("linear", "quadratic"):
        try:
            deg = 1 if trendline_eff == "linear" else 2
            if np.unique(x[~np.isnan(x)]).size > deg:
                xv = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                z = np.polyfit(x, y, deg); p = np.poly1d(z)
                plt.figure(figsize=(10, 5))
                plt.scatter(x, y_plot, alpha=0.35, label="Orders")
                plt.plot(xv, p(xv), linestyle="--", label=f"{trendline_eff.title()} Trend")
                plt.title("Discount vs Sales" + (f" — {title_tag}" if title_tag else ""))
                plt.xlabel("Discount"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
                analyst_chart_b64 = fig_to_b64()
        except Exception:
            pass

    # ---------- bands (Analyst) ----------
    table_rows = ""
    bin_export_csv_b64 = df_to_csv_b64(pd.DataFrame(
        columns=["DiscountBand","AvgDiscount","AvgSales","Orders","AvgSales_SMA3"])
    )  # safe empty by default
    bins_label = "None"

    if is_analyst:
        bins_mode_eff = bins_mode_sel
        if bins_mode_eff == "auto":
            bins_mode_eff = "quantile" if do_sma else "none"

        if bins_mode_eff in ("quantile", "fixed"):
            if bins_mode_eff == "fixed":
                d = x[np.isfinite(x)]
                dmax = float(np.nanmax(d)) if d.size else 0.0
                if dmax <= 1.0:
                    top = max(0.30, float(np.ceil((dmax + 1e-6) / 0.05) * 0.05))
                    edges = np.arange(0.0, top + 1e-7, 0.05)
                else:
                    top = max(25, int(np.ceil(dmax / 5.0) * 5))
                    edges = np.arange(0, top + 1e-7, 5)
                if np.unique(edges).size < 3:
                    bins_mode_eff = "quantile"

            if bins_mode_eff == "quantile":
                try:
                    df_plot["_bin_a"] = pd.qcut(df_plot[col_discount], q=5, duplicates="drop")
                except Exception:
                    df_plot["_bin_a"] = pd.qcut(df_plot[col_discount], q=4, duplicates="drop")
                bins_label = "Quantile (Q1–Q5)"
            else:
                df_plot["_bin_a"] = pd.cut(df_plot[col_discount], bins=edges, include_lowest=True)
                bins_label = "Fixed (heuristic)"

            bin_summary = (df_plot.groupby("_bin_a", observed=True)
                           .agg(AvgDiscount=(col_discount, "mean"),
                                AvgSales=(col_sales, "mean"),
                                Orders=(col_sales, "size"))
                           .reset_index()
                           .sort_values("AvgDiscount", kind="stable"))
            if do_sma and len(bin_summary) >= 3:
                bin_summary["AvgSales_SMA3"] = bin_summary["AvgSales"].rolling(3, min_periods=3).mean()
            else:
                bin_summary["AvgSales_SMA3"] = np.nan

            rows = []
            for _, rr in bin_summary.iterrows():
                rows.append(f"""
                  <tr>
                    <td>{rr["_bin_a"]}</td>
                    <td class="text-end">{float(rr["AvgDiscount"]):,.4f}</td>
                    <td class="text-end">${float(rr["AvgSales"]):,.2f}</td>
                    <td class="text-end">{int(rr["Orders"])}</td>
                    <td class="text-end">{(f"${float(rr['AvgSales_SMA3']):,.2f}" if pd.notna(rr["AvgSales_SMA3"]) else "-")}</td>
                  </tr>""")
            table_rows = "\n".join(rows)
            bin_export_csv_b64 = df_to_csv_b64(bin_summary.rename(columns={"_bin_a":"DiscountBand"}))

    # ---------- exports ----------
    export_cols = [col_discount, col_sales, col_region, col_subcat]
    export_csv_b64 = df_to_csv_b64(
        df_plot[export_cols].rename(columns={
            col_discount: "Discount", col_sales: "Sales",
            col_region: "Region", col_subcat: "Sub-Category"
        })
    )

    mgr_summary_df = pd.DataFrame([{
        "Verdict": verdict, "SweetBand": sweet_band,
        "Sweet_AvgDiscount": sweet_avg_disc, "Sweet_AvgSales": sweet_avg_sale,
        "Sweet_Orders": sweet_orders, "Coverage_%": coverage_pct, "n_orders": n
    }])
    mgr_summary_b64 = df_to_csv_b64(mgr_summary_df)

    back_to_analyst = safe_url('analyst_dashboard')
    back_to_manager = safe_url('manager_dashboard')

    # ---------- render ----------
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Discount Impact — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .kpi { border-radius: var(--card-radius); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="{{ url_for('home') }}"><i class="bi bi-tag"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="{{ url_for('logout') }}"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="{{ url_for('login') }}"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-tag"></i> Discount Impact Analysis</h3>
      <p class="text-muted mb-3">Managers see a summary; Analysts get full controls, KPIs, regression, and bands.</p>

      <!-- Controls -->
      <form method="post" action="{{ url_for('discount_impact') }}" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}<option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>{% endfor %}
          </select>
        </div>
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Sub-Category</label>
          <select name="subcategory" class="form-select">
            {% for s in subcats %}<option value="{{ s }}" {% if s == selected_subcat %}selected{% endif %}>{{ s }}</option>{% endfor %}
          </select>
        </div>

        {% if is_analyst %}
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Trendline</label>
          <select name="trendline" class="form-select">
            <option value="auto" {% if trendline_sel=='auto' %}selected{% endif %}>Auto</option>
            <option value="none" {% if trendline_sel=='none' %}selected{% endif %}>None</option>
            <option value="linear" {% if trendline_sel=='linear' %}selected{% endif %}>Linear</option>
            <option value="quadratic" {% if trendline_sel=='quadratic' %}selected{% endif %}>Quadratic</option>
          </select>
        </div>
        <div class="col-sm-6 col-md-3">
          <label class="form-label">Bins</label>
          <select name="bins_mode" class="form-select">
            <option value="auto" {% if bins_mode_sel=='auto' %}selected{% endif %}>Auto</option>
            <option value="none" {% if bins_mode_sel=='none' %}selected{% endif %}>None</option>
            <option value="quantile" {% if bins_mode_sel=='quantile' %}selected{% endif %}>Quantile (Q1–Q5)</option>
            <option value="fixed" {% if bins_mode_sel=='fixed' %}selected{% endif %}>Fixed (heuristic)</option>
          </select>
        </div>
        <div class="col-sm-6 col-md-3">
          <div class="form-check mt-4">
            <input class="form-check-input" type="checkbox" name="sma" id="sma" {% if do_sma %}checked{% endif %}>
            <label class="form-check-label" for="sma">SMA(3) on bands</label>
          </div>
        </div>
        {% endif %}

        <div class="col-12 d-flex gap-2">
          <button class="btn btn-primary" type="submit"><i class="bi bi-arrow-repeat"></i> Apply</button>
          {% if is_analyst %}
            <a href="{{ back_to_analyst }}" class="btn btn-outline-secondary">Back to Dashboard</a>
          {% elif is_manager %}
            <a href="{{ back_to_manager }}" class="btn btn-outline-secondary">Back to Dashboard</a>
          {% else %}
            <a href="{{ url_for('home') }}" class="btn btn-outline-secondary">Back</a>
          {% endif %}
        </div>
      </form>

      {% if is_analyst %}
      <!-- Analyst KPIs + Regression -->
      <div class="row g-3 mt-3">
        <div class="col-md-2"><div class="p-3 bg-white border kpi">
          <div class="text-muted small">Correlation (r)</div>
          <div class="fs-4 fw-semibold">{{ r|round(3) if r==r else '—' }}</div>
          <div class="small text-muted">Strength &amp; direction</div>
        </div></div>
        <div class="col-md-2"><div class="p-3 bg-white border kpi">
          <div class="text-muted small">R²</div>
          <div class="fs-4 fw-semibold">{{ r2|round(3) if r2==r2 else '—' }}</div>
          <div class="small text-muted">Explained variance</div>
        </div></div>
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="text-muted small">Linear Regression</div>
          <div class="mono small">{{ reg_eq }}</div>
          <div class="small text-muted">Intercept &amp; slope</div>
        </div></div>
        <div class="col-md-2"><div class="p-3 bg-white border kpi">
          <div class="text-muted small">n (orders)</div>
          <div class="fs-4 fw-semibold">{{ n }}</div>
          <div class="small text-muted">Filtered sample</div>
        </div></div>
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="text-muted small">Avg Discount</div>
          <div class="fs-4 fw-semibold">{{ avg_discount|round(4) if avg_discount==avg_discount else '—' }}</div>
          <div class="small text-muted">Across filtered rows</div>
        </div></div>
      </div>
      <hr class="my-3">
      {% endif %}

      {% if is_manager %}
      <!-- Manager summary -->
      <div class="row g-3">
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="small text-muted">Effect of Discount</div>
          <span class="badge text-bg-{{ tone }} mt-1">{{ verdict }}</span>
          <div class="small text-muted mt-2">Based on current selection</div>
        </div></div>
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="small text-muted">Sweet Spot Band</div>
          <div class="fw-semibold">{{ sweet_band }}</div>
          <div class="small text-muted mt-1">Avg Disc: {{ sweet_avg_disc|round(3) if sweet_avg_disc==sweet_avg_disc else '—' }}</div>
        </div></div>
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="small text-muted">Avg Sales in Sweet Spot</div>
          <div class="fs-5">${{ sweet_avg_sale|round(2) if sweet_avg_sale==sweet_avg_sale else '—' }}</div>
        </div></div>
        <div class="col-md-3"><div class="p-3 bg-white border kpi">
          <div class="small text-muted">Orders Coverage</div>
          <div class="fs-5">{{ coverage_pct|round(1) }}%</div>
          <div class="small text-muted">({{ sweet_orders }} of {{ n }})</div>
        </div></div>
      </div>
      {% if margin_risk %}
      <div class="alert alert-warning mt-3"><i class="bi bi-exclamation-triangle me-1"></i>
        Very high discounts are common in the top band — review margin impact.
      </div>
      {% endif %}
      <p class="mt-3 mb-2">{{ msg }}</p>
      <img class="img-fluid border rounded" src="data:image/png;base64,{{ base_chart_b64 }}" alt="Discount vs Sales">
      <div class="mt-2 d-flex gap-2 flex-wrap">
        <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ base_chart_b64 }}" download="discount_vs_sales_manager.png">⬇️ Download PNG</a>
        <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ mgr_summary_b64 }}" download="discount_manager_summary.csv">⬇️ Download Summary</a>
        <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ export_csv_b64 }}" download="discount_impact_filtered.csv">⬇️ Download Data</a>
      </div>

      {% else %}
      <!-- Analyst chart + bands -->
      <div class="row g-4">
        <div class="col-lg-7">
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ analyst_chart_b64 }}" alt="Discount vs Sales">
          <div class="mt-2 d-flex gap-2 flex-wrap">
            <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ analyst_chart_b64 }}" download="discount_vs_sales.png">⬇️ Download PNG</a>
            <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ export_csv_b64 }}" download="discount_impact_filtered.csv">⬇️ Download Filtered CSV</a>
            {% if table_rows %}
            <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ bin_export_csv_b64 }}" download="discount_bands_summary.csv">⬇️ Download Bands CSV</a>
            {% endif %}
          </div>
        </div>
        <div class="col-lg-5">
          {% if table_rows %}
          <h5 class="mb-3">Discount Bands ({{ bins_label }})</h5>
          <div class="table-responsive">
            <table class="table table-sm table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th>Band</th><th class="text-end">Avg Discount</th><th class="text-end">Avg Sales</th>
                  <th class="text-end">Orders</th><th class="text-end">SMA(3)</th>
                </tr>
              </thead>
              <tbody>{{ table_rows|safe }}</tbody>
            </table>
          </div>
          {% endif %}
        </div>
      </div>
      {% endif %}

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">Superstore Forecasting &amp; Analytics Portal © 2025</footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    regions=regions, subcats=subcats,
    selected_region=selected_region, selected_subcat=selected_subcat,
    is_manager=is_manager, is_analyst=is_analyst,
    trendline_sel=trendline_sel, bins_mode_sel=bins_mode_sel, do_sma=do_sma,
    r=r, r2=r2, slope=slope, intercept=intercept, reg_eq=reg_eq, n=n, avg_discount=avg_discount,
    verdict=verdict, tone=tone, msg=msg,
    sweet_band=sweet_band, sweet_avg_disc=sweet_avg_disc,
    sweet_avg_sale=sweet_avg_sale, sweet_orders=sweet_orders,
    coverage_pct=coverage_pct, margin_risk=margin_risk,
    base_chart_b64=base_chart_b64, analyst_chart_b64=analyst_chart_b64,
    export_csv_b64=export_csv_b64, mgr_summary_b64=mgr_summary_b64,
    table_rows=table_rows, bin_export_csv_b64=bin_export_csv_b64, bins_label=bins_label,
    back_to_analyst=back_to_analyst, back_to_manager=back_to_manager
    )


@app.route("/analyst-dashboard", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")  # analysts get full access like managers
def analyst_dashboard():
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    import base64
    from io import BytesIO, StringIO

    # --------------------------- helpers ---------------------------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def df_to_csv_b64(xdf: pd.DataFrame):
        out = StringIO()
        xdf.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    def safe_div(numer, denom):
        return float(numer / denom) if denom and denom != 0 else np.nan

    # --------------------------- ROI quick card (same as manager) ---------------------------
    roi_error = None
    total_sales = total_mkt = 0.0
    roi_overall = np.nan
    best_region = worst_region = None
    chart_b64 = None
    top5 = []

    try:
        df = pd.read_csv(DATA_CSV)


        # Required columns for ROI and DI
        needed = ["Sales", "Marketing Spend", "Discount", "Region", "Sub-Category"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected column(s): {', '.join(missing)} in superstore_extended.csv")

        # Coerce numeric
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)
        df["Marketing Spend"] = pd.to_numeric(df["Marketing Spend"], errors="coerce").fillna(0.0)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
        if "Region" not in df.columns:
            df["Region"] = "All"

        # ROI KPIs
        total_sales = float(df["Sales"].sum())
        total_mkt   = float(df["Marketing Spend"].sum())
        roi_overall = safe_div(total_sales - total_mkt, total_mkt) * 100.0

        reg = (df.groupby("Region", dropna=False)
                 .agg(Total_Sales=("Sales","sum"),
                      Total_Marketing=("Marketing Spend","sum"))
                 .reset_index())
        reg["ROI (%)"] = reg.apply(
            lambda r: safe_div(r["Total_Sales"] - r["Total_Marketing"], r["Total_Marketing"]) * 100.0,
            axis=1
        )
        reg_nonan = reg.dropna(subset=["ROI (%)"]).copy()
        if not reg_nonan.empty:
            best_row = reg_nonan.sort_values("ROI (%)", ascending=False).iloc[0]
            worst_row = reg_nonan.sort_values("ROI (%)", ascending=True).iloc[0]
            best_region = {"Region": str(best_row["Region"]), "ROI": float(best_row["ROI (%)"])}
            worst_region = {"Region": str(worst_row["Region"]), "ROI": float(worst_row["ROI (%)"])}

            # small bar chart
            plt.figure(figsize=(7.5, 4))
            plt.bar(reg_nonan["Region"], reg_nonan["ROI (%)"])
            plt.title("Marketing ROI by Region"); plt.ylabel("ROI (%)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            chart_b64 = fig_to_b64()

            top5 = (reg_nonan.sort_values("ROI (%)", ascending=False)
                           .head(5)
                           .assign(**{
                               "Total_Sales": lambda d: d["Total_Sales"].round(2),
                               "Total_Marketing": lambda d: d["Total_Marketing"].round(2),
                               "ROI (%)": lambda d: d["ROI (%)"].round(2),
                           })
                           .to_dict(orient="records"))

        # --------------------------- Discount Impact (embedded, analyst controls) ---------------------------
        # UI lists
        regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
        subcats = ["All"] + sorted(df["Sub-Category"].dropna().unique().tolist())

        # form selections (analyst has full control on this dashboard)
        di_region = request.form.get("di_region", "All")
        di_subcat = request.form.get("di_subcat", "All")
        di_trend  = request.form.get("di_trend", "auto")       # auto|none|linear|quadratic
        di_bins   = request.form.get("di_bins", "auto")        # auto|none|quantile|fixed
        di_sma    = (request.form.get("di_sma") == "on")       # checkbox

        # filter view
        df_di = df.copy()
        if di_region != "All":
            df_di = df_di[df_di["Region"] == di_region]
        if di_subcat != "All":
            df_di = df_di[df_di["Sub-Category"] == di_subcat]
        if df_di.empty:
            # keep panel rendering but with a message
            di_msg = "No data after filters."
            di_chart_b64 = None
            di_table_rows = ""
            di_bin_export_b64 = df_to_csv_b64(pd.DataFrame(columns=["DiscountBand","AvgDiscount","AvgSales","Orders","AvgSales_SMA3"]))
        else:
            di_msg = None

            # stats
            x = pd.to_numeric(df_di["Discount"], errors="coerce").astype(float).values
            y = pd.to_numeric(df_di["Sales"], errors="coerce").astype(float).values
            n_di = int(len(df_di))
            if n_di > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                di_r = float(np.corrcoef(x, y)[0, 1])
            else:
                di_r = float("nan")
            di_r2 = di_r**2 if np.isfinite(di_r) else float("nan")

            di_slope = float("nan")
            if n_di >= 2 and np.unique(x[~np.isnan(x)]).size >= 2:
                try:
                    di_slope = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    pass

            # chart + trendline
            plt.figure(figsize=(10, 5))
            y_plot = np.clip(y, None, np.nanpercentile(y, 99.5)) if n_di >= 20 else y
            plt.scatter(x, y_plot, alpha=0.35, label="Orders")
            title_bits = []
            if di_region != "All": title_bits.append(di_region)
            if di_subcat != "All": title_bits.append(di_subcat)
            ttl = "Discount vs Sales" + (" — " + " — ".join(title_bits) if title_bits else "")
            plt.title(ttl); plt.xlabel("Discount"); plt.ylabel("Sales")

            # trendline auto logic (same as your page)
            trend_eff = di_trend
            abs_r = abs(di_r) if np.isfinite(di_r) else 0.0
            if trend_eff == "auto":
                trend_eff = "none" if (not np.isfinite(di_r) or abs_r < 0.20 or n_di < 100) else "linear"

            if trend_eff in ("linear","quadratic") and n_di >= (2 if trend_eff=="linear" else 3) and np.unique(x[~np.isnan(x)]).size > (1 if trend_eff=="linear" else 2):
                try:
                    deg = 1 if trend_eff=="linear" else 2
                    z = np.polyfit(x, y, deg); p = np.poly1d(z)
                    xv = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    plt.plot(xv, p(xv), linestyle="--", label=f"{trend_eff.title()} Trend")
                    plt.legend()
                except Exception:
                    pass
            plt.tight_layout()
            di_chart_b64 = fig_to_b64()

            # bands table
            di_table_rows = ""
            di_bin_export_b64 = df_to_csv_b64(pd.DataFrame(columns=["DiscountBand","AvgDiscount","AvgSales","Orders","AvgSales_SMA3"]))
            bins_eff = di_bins
            if bins_eff == "auto":
                bins_eff = "quantile" if di_sma else "none"

            if bins_eff in ("quantile","fixed"):
                if bins_eff == "fixed":
                    d = x[np.isfinite(x)]
                    dmax = float(np.nanmax(d)) if d.size else 0.0
                    if dmax <= 1.0:
                        top = max(0.30, float(np.ceil((dmax + 1e-6)/0.05)*0.05))
                        edges = np.arange(0.0, top+1e-7, 0.05)
                    else:
                        top = max(25, int(np.ceil(dmax/5.0)*5))
                        edges = np.arange(0, top+1e-7, 5)
                    if np.unique(edges).size < 3:
                        bins_eff = "quantile"
                if bins_eff == "quantile":
                    try:
                        df_di["_bin_a"] = pd.qcut(df_di["Discount"], q=5, duplicates="drop")
                    except Exception:
                        df_di["_bin_a"] = pd.qcut(df_di["Discount"], q=4, duplicates="drop")
                else:
                    df_di["_bin_a"] = pd.cut(df_di["Discount"], bins=edges, include_lowest=True)

                bin_summary = (df_di.groupby("_bin_a", observed=True)
                                   .agg(AvgDiscount=("Discount", "mean"),
                                        AvgSales=("Sales", "mean"),
                                        Orders=("Sales", "size"))
                                   .reset_index()
                                   .sort_values("AvgDiscount", kind="stable"))
                if di_sma and len(bin_summary) >= 3:
                    bin_summary["AvgSales_SMA3"] = bin_summary["AvgSales"].rolling(3, min_periods=3).mean()
                else:
                    bin_summary["AvgSales_SMA3"] = np.nan

                rows = []
                for _, rr in bin_summary.iterrows():
                    rows.append(f"""
                      <tr>
                        <td>{rr['_bin_a']}</td>
                        <td class="text-end">{float(rr['AvgDiscount']):,.4f}</td>
                        <td class="text-end">${float(rr['AvgSales']):,.2f}</td>
                        <td class="text-end">{int(rr['Orders'])}</td>
                        <td class="text-end">{(f"${float(rr['AvgSales_SMA3']):,.2f}" if pd.notna(rr['AvgSales_SMA3']) else "-")}</td>
                      </tr>""")
                di_table_rows = "\n".join(rows)
                di_bin_export_b64 = df_to_csv_b64(bin_summary.rename(columns={"_bin_a":"DiscountBand"}))

            # filtered export
            di_export_b64 = df_to_csv_b64(
                df_di[["Discount","Sales","Region","Sub-Category"]]
            )

    except Exception as e:
        # If anything above fails catastrophically, surface it in ROI area.
        roi_error = str(e)
        # Minimal DI defaults to render the dashboard anyway
        regions = ["All"]; subcats = ["All"]
        di_region = di_subcat = "All"
        di_trend = "auto"; di_bins = "auto"; di_sma = False
        di_msg = "Error loading Discount Impact panel."
        di_chart_b64 = None
        di_table_rows = ""
        di_bin_export_b64 = df_to_csv_b64(pd.DataFrame(columns=["DiscountBand","AvgDiscount","AvgSales","Orders","AvgSales_SMA3"]))
        di_export_b64 = df_to_csv_b64(pd.DataFrame(columns=["Discount","Sales","Region","Sub-Category"]))

    # --------------------------- UI ---------------------------
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Analyst Dashboard — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .btn { border-radius: 10px; }
    .lead-small { font-size:.98rem; color:#6c757d; }
    .kpi { padding:14px; border-radius:12px; background:#fff; border:1px solid #eee; }
    .kpi h6 { margin:0; color:#6c757d; }
    .kpi .val { font-size:1.25rem; font-weight:700; }
    .table-sm th, .table-sm td { padding: .45rem .6rem; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    <span class="badge text-bg-success text-capitalize">analyst</span>
    <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
  </div>
</nav>

<div class="container">
  <h2 class="mb-2">🧪 Analyst Dashboard</h2>
  <p class="lead-small mb-4">Same toolkit as Manager — profitability, KPIs, pricing/discount, and inventory risk. Discount Impact is embedded with full controls.</p>

  <!-- Tabs (same as manager) -->
  <ul class="nav nav-tabs" id="anTabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="dashboards-tab" data-bs-toggle="tab" data-bs-target="#dashboards" type="button" role="tab" aria-controls="dashboards" aria-selected="true">
        <i class="bi bi-layout-text-window-reverse me-1"></i> Dashboards
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="pricing-tab" data-bs-toggle="tab" data-bs-target="#pricing" type="button" role="tab" aria-controls="pricing" aria-selected="false">
        <i class="bi bi-tags me-1"></i> Pricing & Discount
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="inventory-tab" data-bs-toggle="tab" data-bs-target="#inventory" type="button" role="tab" aria-controls="inventory" aria-selected="false">
        <i class="bi bi-box-seam me-1"></i> Inventory
      </button>
    </li>
  </ul>

  <div class="tab-content pt-3">

    <!-- Dashboards -->
    <div class="tab-pane fade show active" id="dashboards" role="tabpanel" aria-labelledby="dashboards-tab">
      <div class="row g-4">

        <!-- Marketing ROI Quick Card -->
        <div class="col-lg-12">
          <div class="card h-100">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center mb-2">
                <h5 class="section-title mb-0"><i class="bi bi-cash-coin"></i> Marketing ROI (Quick View)</h5>
                <div>
                  <a class="btn btn-sm btn-outline-primary" href="/roi"><i class="bi bi-cash-coin me-1"></i> Full ROI Analysis</a>
                </div>
              </div>

              {% if roi_error %}
                <div class="alert alert-danger mb-0"><b>ROI Error:</b> {{ roi_error }}</div>
              {% else %}
                <div class="row g-3 mb-3">
                  <div class="col-12 col-md-4"><div class="kpi"><h6>Total Sales</h6><div class="val">${{ "{:,.2f}".format(total_sales) }}</div></div></div>
                  <div class="col-12 col-md-4"><div class="kpi"><h6>Total Marketing Spend</h6><div class="val">${{ "{:,.2f}".format(total_mkt) }}</div></div></div>
                  <div class="col-12 col-md-4"><div class="kpi"><h6>Overall ROI</h6><div class="val">{% if roi_overall == roi_overall %}{{ "{:,.2f}".format(roi_overall) }}%{% else %}N/A{% endif %}</div></div></div>
                </div>

                <div class="d-flex flex-wrap gap-3 mb-3">
                  {% if best_region %}<span class="badge text-bg-success p-2"><i class="bi bi-emoji-smile me-1"></i>Best: {{ best_region.Region }} — {{ "{:,.2f}".format(best_region.ROI) }}%</span>{% endif %}
                  {% if worst_region %}<span class="badge text-bg-danger p-2"><i class="bi bi-emoji-frown me-1"></i>Worst: {{ worst_region.Region }} — {{ "{:,.2f}".format(worst_region.ROI) }}%</span>{% endif %}
                </div>

                <div class="row g-3">
                  <div class="col-12 col-lg-7">
                    {% if chart_b64 %}
                      <img src="data:image/png;base64,{{ chart_b64 }}" class="img-fluid border rounded" alt="ROI by Region">
                    {% else %}
                      <div class="alert alert-warning mb-0">Not enough data to render ROI chart.</div>
                    {% endif %}
                  </div>
                  <div class="col-12 col-lg-5">
                    {% if top5 %}
                      <div class="table-responsive">
                        <table class="table table-sm table-striped table-bordered bg-white">
                          <thead class="table-light"><tr><th>Region</th><th>Total Sales</th><th>Total Marketing</th><th>ROI (%)</th></tr></thead>
                          <tbody>
                            {% for row in top5 %}
                              <tr><td>{{ row["Region"] }}</td><td>${{ "{:,.2f}".format(row["Total_Sales"]) }}</td><td>${{ "{:,.2f}".format(row["Total_Marketing"]) }}</td><td>{{ "{:,.2f}".format(row["ROI (%)"]) }}</td></tr>
                            {% endfor %}
                          </tbody>
                        </table>
                      </div>
                    {% else %}
                      <div class="alert alert-info mb-0">No ROI rows to display.</div>
                    {% endif %}
                  </div>
                </div>
              {% endif %}

              <div class="text-end mt-3"><a class="btn btn-outline-primary" href="/roi">Go to Detailed ROI <i class="bi bi-arrow-right-short"></i></a></div>
            </div>
          </div>
        </div>

        <!-- Profitability + KPI Tracker cards (same as manager) -->
        <div class="col-lg-6"><div class="card h-100"><div class="card-body d-flex flex-column">
          <h5 class="section-title"><i class="bi bi-cash-coin"></i> Profitability (Estimated)</h5>
          <p class="text-muted mb-4">Estimate profit by applying a global margin% over Sales. Filter by region and export charts.</p>
          <div class="mt-auto"><a href="/profitability" class="btn btn-primary"><i class="bi bi-graph-up-arrow me-1"></i> Open Profitability</a></div>
        </div></div></div>

        <div class="col-lg-6"><div class="card h-100"><div class="card-body d-flex flex-column">
          <h5 class="section-title"><i class="bi bi-bullseye"></i> KPI Tracker (Sales vs Target)</h5>
          <p class="text-muted mb-4">Track monthly Sales vs manager-set targets; highlight gaps and momentum.</p>
          <div class="mt-auto"><a href="/kpi-tracker" class="btn btn-primary"><i class="bi bi-speedometer2 me-1"></i> Open KPI Tracker</a></div>
        </div></div></div>

      </div>
    </div>

    <!-- Pricing & Discount (embedded Discount Impact with full controls) -->
    <div class="tab-pane fade" id="pricing" role="tabpanel" aria-labelledby="pricing-tab">
      <div class="card">
        <div class="card-body">
          <h5 class="section-title mb-2"><i class="bi bi-tag"></i> Discount Impact (Analyst Controls)</h5>
          <p class="text-muted mb-3">Explore how discount levels relate to sales. Adjust region, sub-category, trendline, bins, and optional SMA for the binned series. Download the filtered data and band summaries.</p>

          <form method="post" class="row gy-3 gx-3 align-items-end">
            <div class="col-sm-6 col-md-3">
              <label class="form-label">Region</label>
              <select name="di_region" class="form-select">
                {% for r in regions %}<option value="{{ r }}" {% if r==di_region %}selected{% endif %}>{{ r }}</option>{% endfor %}
              </select>
            </div>
            <div class="col-sm-6 col-md-3">
              <label class="form-label">Sub-Category</label>
              <select name="di_subcat" class="form-select">
                {% for s in subcats %}<option value="{{ s }}" {% if s==di_subcat %}selected{% endif %}>{{ s }}</option>{% endfor %}
              </select>
            </div>
            <div class="col-sm-6 col-md-2">
              <label class="form-label">Trendline</label>
              <select name="di_trend" class="form-select">
                <option value="auto" {% if di_trend=='auto' %}selected{% endif %}>Auto</option>
                <option value="none" {% if di_trend=='none' %}selected{% endif %}>None</option>
                <option value="linear" {% if di_trend=='linear' %}selected{% endif %}>Linear</option>
                <option value="quadratic" {% if di_trend=='quadratic' %}selected{% endif %}>Quadratic</option>
              </select>
            </div>
            <div class="col-sm-6 col-md-2">
              <label class="form-label">Bins</label>
              <select name="di_bins" class="form-select">
                <option value="auto" {% if di_bins=='auto' %}selected{% endif %}>Auto</option>
                <option value="none" {% if di_bins=='none' %}selected{% endif %}>None</option>
                <option value="quantile" {% if di_bins=='quantile' %}selected{% endif %}>Quantile</option>
                <option value="fixed" {% if di_bins=='fixed' %}selected{% endif %}>Fixed</option>
              </select>
            </div>
            <div class="col-sm-6 col-md-2">
              <div class="form-check mt-4">
                <input class="form-check-input" type="checkbox" name="di_sma" id="di_sma" {% if di_sma %}checked{% endif %}>
                <label class="form-check-label" for="di_sma">SMA(3) on bands</label>
              </div>
            </div>

            <div class="col-12">
              <button class="btn btn-primary" type="submit"><i class="bi bi-arrow-repeat"></i> Apply</button>
            </div>
          </form>

          {% if di_msg %}
            <div class="alert alert-warning mt-3 mb-0">{{ di_msg }}</div>
          {% else %}
            <div class="row g-4 mt-1">
              <div class="col-lg-7">
                {% if di_chart_b64 %}
                  <img class="img-fluid border rounded" src="data:image/png;base64,{{ di_chart_b64 }}" alt="Discount vs Sales">
                {% endif %}
                <div class="mt-2 d-flex gap-2 flex-wrap">
                  {% if di_chart_b64 %}
                  <a class="btn btn-sm btn-outline-secondary" href="data:image/png;base64,{{ di_chart_b64 }}" download="discount_vs_sales_analyst.png">⬇️ Download PNG</a>
                  {% endif %}
                  {% if di_export_b64 %}
                  <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ di_export_b64 }}" download="discount_filtered.csv">⬇️ Download Filtered CSV</a>
                  {% endif %}
                  {% if di_table_rows %}
                  <a class="btn btn-sm btn-outline-secondary" href="data:text/csv;base64,{{ di_bin_export_b64 }}" download="discount_bands_summary.csv">⬇️ Download Bands CSV</a>
                  {% endif %}
                </div>
              </div>

              <div class="col-lg-5">
                {% if di_table_rows %}
                <h6 class="mb-2">Discount Bands</h6>
                <div class="table-responsive">
                  <table class="table table-sm table-hover align-middle">
                    <thead class="table-light">
                      <tr><th>Band</th><th class="text-end">Avg Discount</th><th class="text-end">Avg Sales</th><th class="text-end">Orders</th><th class="text-end">SMA(3)</th></tr>
                    </thead>
                    <tbody>{{ di_table_rows|safe }}</tbody>
                  </table>
                </div>
                {% endif %}
              </div>
            </div>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Inventory -->
    <div class="tab-pane fade" id="inventory" role="tabpanel" aria-labelledby="inventory-tab">
      <div class="row g-4">
        <div class="col-lg-12">
          <div class="card h-100"><div class="card-body d-flex flex-column">
            <h5 class="section-title"><i class="bi bi-box-seam"></i> Stock Alert Simulation</h5>
            <p class="text-muted mb-4">Monte Carlo inventory simulation with service level & lead time; visualize stockout risk and export PNG/CSV.</p>
            <div class="mt-auto"><a href="/stock-alert" class="btn btn-primary"><i class="bi bi-lightning-charge me-1"></i> Open Simulation</a></div>
          </div></div>
        </div>
      </div>      
    </div>

  </div>

  <footer class="mt-5 text-center small">
    Superstore Forecasting &amp; Analytics Portal © 2025 · <span class="text-muted">Analyst Suite</span>
  </footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    # ROI card vars
    roi_error=roi_error,
    total_sales=total_sales, total_mkt=total_mkt, roi_overall=roi_overall,
    best_region=best_region, worst_region=worst_region, top5=top5, chart_b64=chart_b64,
    # DI panel vars
    regions=locals().get("regions", ["All"]),
    subcats=locals().get("subcats", ["All"]),
    di_region=locals().get("di_region", "All"),
    di_subcat=locals().get("di_subcat", "All"),
    di_trend=locals().get("di_trend", "auto"),
    di_bins=locals().get("di_bins", "auto"),
    di_sma=locals().get("di_sma", False),
    di_msg=locals().get("di_msg", None),
    di_chart_b64=locals().get("di_chart_b64", None),
    di_table_rows=locals().get("di_table_rows", ""),
    di_bin_export_b64=locals().get("di_bin_export_b64", df_to_csv_b64(pd.DataFrame())),
    di_export_b64=locals().get("di_export_b64", df_to_csv_b64(pd.DataFrame())),
    )


@app.route("/stock-alert", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")
def stock_alert():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    # ---------- Helpers ----------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def clamp(val, lo, hi):
        try:
            return max(lo, min(hi, val))
        except Exception:
            return lo

    # ---------- Load data ----------
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back</a>"

    have_subcat = "Sub-Category" in df.columns
    have_region = "Region" in df.columns
    subcats = ["All"] + (sorted(df["Sub-Category"].dropna().unique()) if have_subcat else [])
    regions = ["All"] + (sorted(df["Region"].dropna().unique()) if have_region else [])

    # ---------- Defaults ----------
    params = {
        "subcategory": "All",
        "region": "All",
        "initial_stock": 2000.0,
        "lead_time_days": 7,
        "review_period_days": 1,
        "service_level": 0.95,
        "order_up_to_multiplier": 1.5,
        "runs": 200,
        "horizon_days": 90
    }

    chart_base64 = None
    csv_b64 = None
    result = None
    err = None
    info_note = "Uses historical Sales as a proxy for demand."

    # ---------- Read form ----------
    if request.method == "POST":
        def fnum(name, cast=float, default=None):
            raw = request.form.get(name, None)
            try:
                return cast(raw) if raw not in (None, "") else default
            except Exception:
                return default

        params["subcategory"] = request.form.get("subcategory", "All")
        params["region"] = request.form.get("region", "All")

        params["initial_stock"] = fnum("initial_stock", float, 2000.0)
        params["lead_time_days"] = int(clamp(fnum("lead_time_days", int, 7), 0, 60))
        params["review_period_days"] = int(clamp(fnum("review_period_days", int, 1), 1, 30))
        params["service_level"] = float(clamp(fnum("service_level", float, 0.95), 0.5, 0.999))
        params["order_up_to_multiplier"] = float(clamp(fnum("order_up_to_multiplier", float, 1.5), 1.1, 3.0))
        params["runs"] = int(clamp(fnum("runs", int, 200), 50, 1000))
        params["horizon_days"] = int(clamp(fnum("horizon_days", int, 90), 30, 365))

        # ---------- Build demand & simulate ----------
        try:
            # NOTE: expects your helpers to exist
            # build_daily_demand(df, subcat=None|'name', region=None|'name') -> daily demand Series with DatetimeIndex
            # simulate_inventory_mc(...) -> dict with 'percentiles', 'paths', and KPI scalars
            demand = build_daily_demand(
                df,
                subcat=None if params["subcategory"] in ("", "All") else params["subcategory"],
                region=None if params["region"] in ("", "All") else params["region"],
            )

            if demand is None or getattr(demand, "empty", False):
                err = "No historical demand found for the chosen filters."
            else:
                result = simulate_inventory_mc(
                    demand_series=demand,
                    initial_stock=params["initial_stock"],
                    lead_time_days=params["lead_time_days"],
                    review_period_days=params["review_period_days"],
                    service_level=params["service_level"],
                    order_up_to_multiplier=params["order_up_to_multiplier"],
                    runs=params["runs"],
                    horizon_days=params["horizon_days"],
                )

                # Plot percentile fan + one sample path
                pct = result["percentiles"]
                plt.figure(figsize=(11, 6))
                plt.fill_between(pct["date"], pct["p10"], pct["p90"], alpha=0.2, label="p10–p90")
                plt.plot(pct["date"], pct["p50"], label="Median on-hand (p50)")
                last_path = result["paths"][-1]
                plt.plot(last_path["date"], last_path["on_hand"], linestyle="--", label="Sample run")

                ttl = ["Inventory Simulation"]
                tag_region = None if params["region"] in ("", "All") else params["region"]
                tag_subcat = None if params["subcategory"] in ("", "All") else params["subcategory"]
                for t in (tag_subcat, tag_region):
                    if t: ttl.append(t)
                plt.title(" — ".join(ttl))
                plt.ylabel("On-hand (units)")
                plt.xlabel("Date")
                plt.legend()
                plt.tight_layout()
                chart_base64 = fig_to_b64()
                plt.close()

                # CSV of last run (on_hand, order_qty, backorder)
                csv_b64 = base64.b64encode(last_path.to_csv(index=False).encode("utf-8")).decode("utf-8")

        except Exception as e:
            err = str(e)

    # ---------- Render UI ----------
    # Convert dict to simple object for dot access in Jinja
    P = type("P", (), params)

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Inventory Stock Alert — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .kpi { border-radius: var(--card-radius); }
    .form-text { margin-top: .25rem; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-box-seam"></i> Inventory Stock Alert — Simulation</h3>
      <p class="text-muted mb-3">{{ info_note }}</p>

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Sub-Category</label>
          <select name="subcategory" class="form-select">
            {% for s in subcats %}
              <option value="{{ s }}" {% if P.subcategory == s %}selected{% endif %}>{{ s }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-4">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{ r }}" {% if P.region == r %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-4">
          <label class="form-label">Initial Stock</label>
          <input type="number" step="0.01" name="initial_stock" value="{{ P.initial_stock }}" class="form-control">
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Lead Time (days)</label>
          <input type="number" min="0" max="60" name="lead_time_days" value="{{ P.lead_time_days }}" class="form-control">
          <div class="form-text">Supplier delivery delay.</div>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Review Period (days)</label>
          <input type="number" min="1" max="30" name="review_period_days" value="{{ P.review_period_days }}" class="form-control">
          <div class="form-text">How often you place orders.</div>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Service Level</label>
          <input type="number" step="0.001" min="0.5" max="0.999" name="service_level" value="{{ P.service_level }}" class="form-control">
          <div class="form-text">Fill-rate target (e.g., 0.95 = 95%).</div>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Order-up-to × ROP</label>
          <input type="number" step="0.1" min="1.1" max="3" name="order_up_to_multiplier" value="{{ P.order_up_to_multiplier }}" class="form-control">
          <div class="form-text">Safety span above Reorder Point.</div>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Monte Carlo Runs</label>
          <input type="number" min="50" max="1000" name="runs" value="{{ P.runs }}" class="form-control">
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Horizon (days)</label>
          <input type="number" min="30" max="365" name="horizon_days" value="{{ P.horizon_days }}" class="form-control">
        </div>

        <div class="col-12">
          <button class="btn btn-primary" type="submit"><i class="bi bi-play-fill"></i> Run</button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      {% if err %}
        <div class="alert alert-warning mt-3">{{ err }}</div>
      {% endif %}

      {% if result %}
        <hr class="my-4">

        <!-- KPI Cards -->
        <div class="row g-3">
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Stockout Probability</div>
              <div class="fs-4 fw-semibold">{{ '%.1f'|format(result.stockout_prob*100) }}%</div>
              <div class="small text-muted">Across {{ P.runs }} runs / {{ P.horizon_days }} days</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Average On-hand</div>
              <div class="fs-4 fw-semibold">{{ '{:,.2f}'.format(result.avg_on_hand) }}</div>
              <div class="small text-muted">Mean inventory level</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Average Backorders</div>
              <div class="fs-4 fw-semibold">{{ '{:,.2f}'.format(result.avg_backorders) }}</div>
              <div class="small text-muted">Unfilled demand</div>
            </div>
          </div>
        </div>

        <!-- Policy Summary -->
        <div class="mt-4">
          <h5 class="mb-2">Policy & Demand Summary</h5>
          <div class="row">
            <div class="col-md-6">
              <ul class="mb-0">
                <li><b>Mean daily demand:</b> {{ '%.2f'|format(result.mean_demand) }}</li>
                <li><b>Std daily demand:</b> {{ '%.2f'|format(result.std_demand) }}</li>
              </ul>
            </div>
            <div class="col-md-6">
              <ul class="mb-0">
                <li><b>Reorder Point (ROP):</b> {{ '%.2f'|format(result.rop) }}</li>
                <li><b>Order-up-to level:</b> {{ '%.2f'|format(result.order_up_to) }}</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- Chart + Downloads -->
        <div class="mt-4">
          <img src="data:image/png;base64,{{ chart_base64 }}" class="img-fluid border rounded" alt="Inventory Simulation">
          <div class="mt-2 d-flex gap-2 flex-wrap">
            <a class="btn btn-sm btn-outline-secondary"
               href="data:image/png;base64,{{ chart_base64 }}"
               download="inventory_simulation{% if P.subcategory not in ('', 'All') %}_{{ P.subcategory|lower }}{% endif %}{% if P.region not in ('', 'All') %}_{{ P.region|lower }}{% endif %}.png">
               ⬇️ Download PNG
            </a>
            {% if csv_b64 %}
              <a class="btn btn-sm btn-outline-secondary"
                 href="data:text/csv;base64,{{ csv_b64 }}"
                 download="inventory_simulation_last_run.csv">
                 ⬇️ Download CSV (last run)
              </a>
            {% endif %}
          </div>
        </div>

      {% endif %}
    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    subcats=subcats,
    regions=regions,
    P=P,
    result=result,
    err=err,
    info_note=info_note,
    chart_base64=chart_base64,
    csv_b64=csv_b64)



@app.route("/interactive-analysis", methods=["GET", "POST"])
@login_required
@roles_required("analyst", "manager")
def interactive_analysis():
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO, StringIO
    import base64

    # ---------- Helpers ----------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def df_to_csv_b64(xdf):
        out = StringIO()
        xdf.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    # ---------- Load & validate ----------
    try:
        df = pd.read_csv(DATA_CSV)

    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back</a>"

    need = {"Order Date", "Sales", "Sub-Category"}
    if not need.issubset(df.columns):
        missing = ", ".join(sorted(need - set(df.columns)))
        return f"<p style='color:red'><b>Missing columns:</b> {missing}</p><a href='/'>Back</a>"

    have_region = "Region" in df.columns

    # Clean
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales", "Sub-Category"])

    subcategories = sorted(df["Sub-Category"].dropna().unique().tolist())
    regions = ["All"] + (sorted(df["Region"].dropna().unique().tolist()) if have_region else ["All"])

    # ---------- Defaults / form ----------
    selected_subcat = subcategories[0] if subcategories else ""
    selected_region = "All"
    sma_win = 0     # 0, 3, 6, 12
    show_yoy = False

    chart_base64 = None
    csv_b64 = None
    table_rows = ""
    no_data_message = ""

    if request.method == "POST":
        selected_subcat = request.form.get("subcategory", selected_subcat)
        selected_region = request.form.get("region", "All")
        try:
            sma_win = int(request.form.get("sma", "0"))
            sma_win = sma_win if sma_win in (0, 3, 6, 12) else 0
        except:
            sma_win = 0
        show_yoy = request.form.get("yoy") == "on"

    # ---------- Filter ----------
    q = df[df["Sub-Category"] == selected_subcat].copy()
    if have_region and selected_region != "All":
        q = q[q["Region"] == selected_region]

    if q.empty:
        no_data_message = f"No sales data found for '{selected_subcat}'" + (f" in {selected_region}." if have_region and selected_region != "All" else ".")
    else:
        # Monthly aggregate
        q["Month"] = q["Order Date"].dt.to_period("M").dt.to_timestamp()
        monthly = (q.groupby("Month", as_index=False)["Sales"].sum()
                     .sort_values("Month"))
        monthly.rename(columns={"Sales": "Actual"}, inplace=True)

        # Optional YoY series (previous year alignment)
        yoy = None
        if show_yoy and len(monthly) >= 13:
            prev = monthly.copy()
            prev["Month"] = prev["Month"] + pd.offsets.DateOffset(years=1)
            yoy = monthly[["Month"]].merge(prev.rename(columns={"Actual":"YoY (prev year)"}), on="Month", how="left")

        # Optional SMA
        monthly["SMA"] = None
        if sma_win > 0 and len(monthly) >= sma_win:
            monthly["SMA"] = monthly["Actual"].rolling(window=sma_win, min_periods=sma_win).mean()

        # KPIs
        total_sales = float(monthly["Actual"].sum())
        last_row = monthly.iloc[-1]
        last_month_val = float(last_row["Actual"])
        best_idx = monthly["Actual"].idxmax()
        best_month = monthly.loc[best_idx, "Month"]
        best_value = float(monthly.loc[best_idx, "Actual"])

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(monthly["Month"], monthly["Actual"], marker="o", label="Monthly Sales")
        if sma_win > 0 and monthly["SMA"].notna().any():
            plt.plot(monthly["Month"], monthly["SMA"], linestyle="--", label=f"{sma_win}-Month SMA")
        if yoy is not None and yoy["YoY (prev year)"].notna().any():
            plt.plot(yoy["Month"], yoy["YoY (prev year)"], linestyle=":", label="YoY (prev year)")
        title_region = f" — {selected_region}" if have_region and selected_region != "All" else ""
        plt.title(f"Monthly Sales Trend — {selected_subcat}{title_region}")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        chart_base64 = fig_to_b64()
        plt.close()

        # Recent 12 months table
        recent = monthly.tail(12).copy()
        recent["MonthLabel"] = recent["Month"].dt.strftime("%Y-%m")
        for _, r in recent.iterrows():
            sma_txt = ("$" + f"{r['SMA']:,.2f}") if pd.notna(r["SMA"]) else "-"
            table_rows += f"""
              <tr>
                <td>{r['MonthLabel']}</td>
                <td class="text-end">${r['Actual']:,.2f}</td>
                <td class="text-end">{sma_txt}</td>
              </tr>
            """

        # CSV export
        export_df = monthly[["Month", "Actual", "SMA"]].copy()
        export_df["Month"] = export_df["Month"].dt.strftime("%Y-%m")
        csv_b64 = df_to_csv_b64(export_df)

    # ---------- Render ----------
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Analysis — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .table thead th { white-space: nowrap; }
    .kpi { border-radius: var(--card-radius); }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-intersect"></i> Interactive Sales Analysis</h3>
      <p class="text-muted mb-3">Select a sub-category (and optional region) to view monthly trends, YoY overlay, and smoothing.</p>

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Sub-Category</label>
          <select name="subcategory" class="form-select" required>
            {% for subcat in subcategories %}
              <option value="{{ subcat }}" {% if subcat == selected_subcat %}selected{% endif %}>{{ subcat }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Smoothing (SMA)</label>
          <select name="sma" class="form-select">
            <option value="0"  {% if sma_win == 0  %}selected{% endif %}>None</option>
            <option value="3"  {% if sma_win == 3  %}selected{% endif %}>3 months</option>
            <option value="6"  {% if sma_win == 6  %}selected{% endif %}>6 months</option>
            <option value="12" {% if sma_win == 12 %}selected{% endif %}>12 months</option>
          </select>
        </div>

        <div class="col-sm-6 col-md-2">
          <div class="form-check mt-4">
            <input class="form-check-input" type="checkbox" name="yoy" id="yoy" {% if show_yoy %}checked{% endif %}>
            <label class="form-check-label" for="yoy">Show YoY</label>
          </div>
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-primary">
            <i class="bi bi-arrow-repeat"></i> Update
          </button>
          <a href="/" class="btn btn-link">Back to Home</a>
        </div>
      </form>

      {% if no_data_message %}
        <div class="alert alert-warning mt-3">{{ no_data_message }}</div>
      {% endif %}

      {% if chart_base64 %}
        <!-- KPI Cards -->
        <div class="row g-3 mt-3">
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Total Sales</div>
              <div class="fs-4 fw-semibold">
                ${{ '{:,.2f}'.format(total_sales) }}
              </div>
              <div class="small text-muted">{{ selected_subcat }}{% if selected_region != 'All' %} — {{ selected_region }}{% endif %}</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Last Month</div>
              <div class="fs-5">
                ${{ '{:,.2f}'.format(last_month_val) }}
              </div>
              <div class="small text-muted">Most recent period</div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 bg-white border kpi">
              <div class="text-muted small">Best Month</div>
              <div class="fs-6">
                {{ best_month.strftime('%Y-%m') }} — <b>${{ '{:,.2f}'.format(best_value) }}</b>
              </div>
              <div class="small text-muted">Peak within range</div>
            </div>
          </div>
        </div>

        <hr class="my-4">

        <!-- Chart + Downloads -->
        <div class="row g-4">
          <div class="col-lg-7">
            <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Monthly Sales Trend">
            <div class="mt-2 d-flex gap-2 flex-wrap">
              <a class="btn btn-sm btn-outline-secondary"
                 href="data:image/png;base64,{{ chart_base64 }}"
                 download="trend_{{ selected_subcat|replace(' ', '_')|lower }}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.png">
                 ⬇️ Download PNG
              </a>
              {% if csv_b64 %}
                <a class="btn btn-sm btn-outline-secondary"
                   href="data:text/csv;base64,{{ csv_b64 }}"
                   download="monthly_sales_{{ selected_subcat|replace(' ', '_')|lower }}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.csv">
                   ⬇️ Download CSV
                </a>
              {% endif %}
            </div>
          </div>

          <div class="col-lg-5">
            <h5 class="mb-3">Recent 12 Months</h5>
            <div class="table-responsive">
              <table class="table table-sm table-hover align-middle">
                <thead class="table-light">
                  <tr>
                    <th>Month</th>
                    <th class="text-end">Sales</th>
                    <th class="text-end">SMA</th>
                  </tr>
                </thead>
                <tbody>
                  {{ table_rows|safe }}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      {% endif %}

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """,
    # Jinja vars
    subcategories=subcategories,
    regions=regions,
    selected_subcat=selected_subcat,
    selected_region=selected_region,
    sma_win=sma_win,
    show_yoy=show_yoy,
    chart_base64=chart_base64,
    csv_b64=csv_b64,
    table_rows=table_rows,
    no_data_message=no_data_message,
    # KPIs (only used if chart exists)
    total_sales=locals().get("total_sales", 0.0),
    last_month_val=locals().get("last_month_val", 0.0),
    best_month=locals().get("best_month", pd.NaT),
    best_value=locals().get("best_value", 0.0),
)


@app.route("/compare-models", methods=["GET", "POST"])
@login_required
@roles_required("manager", "analyst")
def compare_models():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO, StringIO
    import base64

    # ---------- Helpers ----------
    def fig_to_b64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def df_to_csv_b64(xdf):
        out = StringIO()
        xdf.to_csv(out, index=False)
        return base64.b64encode(out.getvalue().encode("utf-8")).decode("utf-8")

    def clamp(v, lo, hi, cast=int):
        try:
            return max(lo, min(hi, cast(v)))
        except Exception:
            return lo

    # ---------- Load & validate ----------
    try:
        df = pd.read_csv(DATA_CSV)
    except Exception as e:
        return f"<p style='color:red'><b>Error reading superstore_extended.csv:</b> {e}</p><a href='/'>Back to Home</a>"

    required_cols = {"Order Date", "Sub-Category", "Sales"}
    if not required_cols.issubset(df.columns):
        return (
            "<p style='color:red'><b>Required columns missing. "
            "Need 'Order Date', 'Sub-Category', 'Sales'.</b></p>"
            "<a href='/'>Back to Home</a>"
        )

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    have_region = "Region" in df.columns
    subcategories = sorted(df["Sub-Category"].dropna().unique().tolist())
    regions = ["All"] + (sorted(df["Region"].dropna().unique().tolist()) if have_region else ["All"])

    # ---------- Defaults / form ----------
    selected_subcat = subcategories[0] if subcategories else ""
    selected_region = "All"
    forecast_months = 6   # 1..24
    backtest_months = 6   # 3..18
    model_error_note = ""
    chart_base64 = None
    csv_b64 = None
    table_rows = ""
    no_data_message = ""
    kpis = {"arima": {"MAE": None, "RMSE": None, "MAPE": None},
            "prophet": {"MAE": None, "RMSE": None, "MAPE": None}}

    if request.method == "POST":
        selected_subcat = request.form.get("subcategory", selected_subcat)
        selected_region = request.form.get("region", "All")
        forecast_months = clamp(request.form.get("months", 6), 1, 24, int)
        backtest_months = clamp(request.form.get("backtest", 6), 3, 18, int)

    # ---------- Filter & monthly aggregate ----------
    q = df[df["Sub-Category"] == selected_subcat].copy()
    if have_region and selected_region != "All":
        q = q[q["Region"] == selected_region]

    if q.empty:
        no_data_message = f"No data for sub-category '{selected_subcat}'" + (f" in {selected_region}." if have_region and selected_region != "All" else ".")
        return render_template_string(PAGE_TMPL,
            subcategories=subcategories, regions=regions,
            selected_subcat=selected_subcat, selected_region=selected_region,
            forecast_months=forecast_months, backtest_months=backtest_months,
            model_error_note=model_error_note, chart_base64=chart_base64,
            csv_b64=csv_b64, table_rows=table_rows, no_data_message=no_data_message,
            kpis=kpis
        )

    q["Month"] = q["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (q.groupby("Month", as_index=False)["Sales"].sum()
                 .sort_values("Month"))
    if len(monthly) < 6:
        no_data_message = f"Not enough history for '{selected_subcat}' to compare models."
        return render_template_string(PAGE_TMPL,
            subcategories=subcategories, regions=regions,
            selected_subcat=selected_subcat, selected_region=selected_region,
            forecast_months=forecast_months, backtest_months=backtest_months,
            model_error_note=model_error_note, chart_base64=chart_base64,
            csv_b64=csv_b64, table_rows=table_rows, no_data_message=no_data_message,
            kpis=kpis
        )

    # Build series
    y = (monthly.set_index("Month")["Sales"].astype(float).asfreq("MS"))
    history_end = y.index.max()

    # ---------- Backtest split ----------
    bt = min(backtest_months, max(3, len(y) // 4))  # keep it sane
    train = y.iloc[:-bt] if len(y) > bt else y.iloc[:0]
    test = y.iloc[-bt:] if len(y) > bt else y.iloc[:0]

    # ---------- Run Prophet (optional) ----------
    prophet_pred_future = None
    prophet_bt_pred = None
    prophet_ok = False
    try:
        from prophet import Prophet  # may not be installed
        prophet_ok = True
    except Exception:
        model_error_note = "Prophet not installed; showing ARIMA/SARIMAX only."

    if prophet_ok and len(train) >= 3:
        try:
            # Train Prophet on train
            prophet_train = train.reset_index().rename(columns={"Month": "ds", "Sales": "y"})
            m_bt = Prophet()
            m_bt.fit(prophet_train)
            future_bt = pd.DataFrame({"ds": test.index})
            fc_bt = m_bt.predict(future_bt)
            prophet_bt_pred = pd.Series(fc_bt["yhat"].values, index=test.index)

            # Fit on full and forecast horizon
            prophet_full = y.reset_index().rename(columns={"Month": "ds", "Sales": "y"})
            m = Prophet()
            m.fit(prophet_full)
            future = m.make_future_dataframe(periods=forecast_months, freq="MS")
            fc = m.predict(future)
            prophet_pred = fc.set_index("ds")["yhat"]
            prophet_pred_future = prophet_pred.iloc[-forecast_months:]
        except Exception as e:
            model_error_note = f"Prophet issue: {e}"
            prophet_ok = False

    # ---------- Run ARIMA/SARIMAX ----------
    arima_bt_pred = None
    arima_future_mean = None
    arima_ci_lower = None
    arima_ci_upper = None
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Log-transform to stabilize variance
        train_log = np.log1p(train) if len(train) else train
        full_log = np.log1p(y)

        # Heuristic seasonal order
        use_seasonal = len(train_log) >= 24
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12) if use_seasonal else (0, 0, 0, 0)
        trend = "c" if use_seasonal else "t"

        # Backtest fit
        if len(train_log) >= 3:
            model_bt = SARIMAX(
                train_log, order=order, seasonal_order=seasonal_order, trend=trend,
                enforce_stationarity=False, enforce_invertibility=False
            )
            res_bt = model_bt.fit(disp=False)
            fc_bt = res_bt.get_forecast(steps=len(test))
            arima_bt_pred = np.expm1(fc_bt.predicted_mean)
        # Full fit & future
        model_full = SARIMAX(
            full_log, order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res_full = model_full.fit(disp=False)
        fc = res_full.get_forecast(steps=forecast_months)
        arima_future_mean = np.expm1(fc.predicted_mean)
        ci = fc.conf_int()
        arima_ci_lower = np.expm1(ci.iloc[:, 0])
        arima_ci_upper = np.expm1(ci.iloc[:, 1])
    except Exception as e:
        model_error_note = f"ARIMA/SARIMAX issue: {e}"

    # ---------- Compute backtest KPIs ----------
    def metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype="float64")
        y_pred = np.asarray(y_pred, dtype="float64")
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        # avoid div-by-zero for mape
        denom = np.where(y_true == 0, np.nan, y_true)
        mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
        return mae, rmse, mape

    if len(test) and arima_bt_pred is not None and len(arima_bt_pred) == len(test):
        kpis["arima"]["MAE"], kpis["arima"]["RMSE"], kpis["arima"]["MAPE"] = metrics(test.values, arima_bt_pred.values)

    if prophet_ok and prophet_bt_pred is not None and len(prophet_bt_pred) == len(test):
        kpis["prophet"]["MAE"], kpis["prophet"]["RMSE"], kpis["prophet"]["MAPE"] = metrics(test.values, prophet_bt_pred.values)

    # ---------- Build combined export (history + future) ----------
    # Historical segment
    hist_df = monthly[["Month", "Sales"]].rename(columns={"Month": "Date", "Sales": "Actual"})
    # Backtest predictions
    if len(test):
        hist_df = hist_df.merge(
            pd.DataFrame({"Date": test.index, "ARIMA_bt": arima_bt_pred.values if arima_bt_pred is not None and len(arima_bt_pred)==len(test) else np.nan}),
            on="Date", how="left"
        )
        if prophet_ok:
            hist_df = hist_df.merge(
                pd.DataFrame({"Date": test.index, "Prophet_bt": prophet_bt_pred.values if prophet_bt_pred is not None and len(prophet_bt_pred)==len(test) else np.nan}),
                on="Date", how="left"
            )
    # Future segment
    future_dates = pd.date_range(history_end + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
    future_df = pd.DataFrame({"Date": future_dates})
    if arima_future_mean is not None:
        future_df["ARIMA_fc"] = arima_future_mean.values
        if arima_ci_lower is not None and arima_ci_upper is not None:
            future_df["ARIMA_low"] = arima_ci_lower.values
            future_df["ARIMA_high"] = arima_ci_upper.values
    if prophet_ok and prophet_pred_future is not None:
        future_df["Prophet_fc"] = prophet_pred_future.values

    export_df = pd.concat([hist_df, future_df], ignore_index=True)
    # CSV
    csv_b64 = df_to_csv_b64(export_df)

    # ---------- Plot ----------
    plt.figure(figsize=(10, 5))
    # Historical
    plt.plot(hist_df["Date"], hist_df["Actual"], label="Historical Sales")
    # Prophet future
    if prophet_ok and prophet_pred_future is not None:
        plt.plot(future_dates, prophet_pred_future.values, linestyle="--", label="Prophet Forecast")
    # ARIMA future
    if arima_future_mean is not None:
        plt.plot(future_dates, arima_future_mean.values, linestyle="--", label="ARIMA/SARIMAX Forecast")
        try:
            if ("ARIMA_low" in future_df.columns) and ("ARIMA_high" in future_df.columns):
                plt.fill_between(future_dates, future_df["ARIMA_low"], future_df["ARIMA_high"], alpha=0.15, label="ARIMA 95% CI")
        except Exception:
            pass

    title_region = f" — {selected_region}" if have_region and selected_region != "All" else ""
    plt.title(f"ARIMA vs Prophet — {selected_subcat}{title_region}")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    chart_base64 = fig_to_b64()
    plt.close()

    # ---------- Build KPI table rows ----------
    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return f"{v:,.2f}%" if pct else f"{v:,.2f}"

    table_rows = f"""
      <tr><td>ARIMA/SARIMAX</td><td class="text-end">{fmt(kpis['arima']['MAE'])}</td><td class="text-end">{fmt(kpis['arima']['RMSE'])}</td><td class="text-end">{fmt(kpis['arima']['MAPE'], pct=True)}</td></tr>
      <tr><td>Prophet</td><td class="text-end">{fmt(kpis['prophet']['MAE'])}</td><td class="text-end">{fmt(kpis['prophet']['RMSE'])}</td><td class="text-end">{fmt(kpis['prophet']['MAPE'], pct=True)}</td></tr>
    """

    # ---------- Render ----------
    return render_template_string(PAGE_TMPL,
        subcategories=subcategories, regions=regions,
        selected_subcat=selected_subcat, selected_region=selected_region,
        forecast_months=forecast_months, backtest_months=backtest_months,
        model_error_note=model_error_note, chart_base64=chart_base64,
        csv_b64=csv_b64, table_rows=table_rows, no_data_message=no_data_message,
        kpis=kpis
    )


# ---------- Page Template (Bootstrap UI) ----------
PAGE_TMPL = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Compare Forecast Models — Sales Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap + Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root { --card-radius: 14px; }
    body { padding: 24px; background:#f7f8fa; }
    .card { border-radius: var(--card-radius); box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .section-title { display:flex; align-items:center; gap:.5rem; }
    .section-title .bi { opacity:.85; }
    .table thead th { white-space: nowrap; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary rounded mb-4 px-3">
  <a class="navbar-brand fw-semibold" href="/"><i class="bi bi-graph-up-arrow me-1"></i> Sales Insights</a>
  <div class="ms-auto d-flex align-items-center gap-2">
    {% if current_user.is_authenticated %}
      <span class="badge text-bg-primary text-capitalize">{{ current_user.role }}</span>
      <a class="btn btn-sm btn-outline-secondary" href="/logout"><i class="bi bi-box-arrow-right"></i> Logout</a>
    {% else %}
      <a class="btn btn-sm btn-primary" href="/login"><i class="bi bi-box-arrow-in-right"></i> Login</a>
    {% endif %}
  </div>
</nav>

<div class="container">
  <div class="card">
    <div class="card-body">
      <h3 class="section-title mb-2"><i class="bi bi-activity"></i> Compare Forecast Models</h3>
      <p class="text-muted mb-3">ARIMA/SARIMAX vs Prophet (if installed). Backtest on recent months, then forecast into the future.</p>

      <!-- Controls -->
      <form method="post" class="row gy-3 gx-3 align-items-end">
        <div class="col-sm-6 col-md-4">
          <label class="form-label">Sub-Category</label>
          <select name="subcategory" class="form-select" required>
            {% for subcat in subcategories %}
              <option value="{{ subcat }}" {% if subcat == selected_subcat %}selected{% endif %}>{{ subcat }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-3">
          <label class="form-label">Region</label>
          <select name="region" class="form-select">
            {% for r in regions %}
              <option value="{{ r }}" {% if r == selected_region %}selected{% endif %}>{{ r }}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-sm-6 col-md-2">
          <label class="form-label">Forecast (months)</label>
          <input type="number" name="months" value="{{ forecast_months }}" min="1" max="24" class="form-control">
        </div>

        <div class="col-sm-6 col-md-2">
          <label class="form-label">Backtest (months)</label>
          <input type="number" name="backtest" value="{{ backtest_months }}" min="3" max="18" class="form-control">
        </div>

        <div class="col-sm-6 col-md-1">
          <button type="submit" class="btn btn-primary w-100"><i class="bi bi-arrow-repeat"></i></button>
        </div>
      </form>

      {% if model_error_note %}
        <div class="alert alert-warning mt-3">{{ model_error_note }}</div>
      {% endif %}

      {% if no_data_message %}
        <div class="alert alert-warning mt-3">{{ no_data_message }}</div>
      {% endif %}

      {% if chart_base64 %}
        <hr class="my-4">

        <!-- KPI Table -->
        <h5 class="mb-2">Backtest KPIs</h5>
        <div class="table-responsive">
          <table class="table table-sm table-hover align-middle">
            <thead class="table-light">
              <tr>
                <th>Model</th>
                <th class="text-end">MAE</th>
                <th class="text-end">RMSE</th>
                <th class="text-end">MAPE</th>
              </tr>
            </thead>
            <tbody>
              {{ table_rows|safe }}
            </tbody>
          </table>
        </div>

        <!-- Chart + Downloads -->
        <div class="row g-4 mt-1">
          <div class="col-lg-8">
            <img class="img-fluid border rounded" src="data:image/png;base64,{{ chart_base64 }}" alt="Forecast Comparison">
            <div class="mt-2 d-flex gap-2 flex-wrap">
              <a class="btn btn-sm btn-outline-secondary"
                 href="data:image/png;base64,{{ chart_base64 }}"
                 download="compare_arima_prophet{% if selected_subcat %}_{{ selected_subcat|replace(' ', '_')|lower }}{% endif %}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.png">
                 ⬇️ Download PNG
              </a>
              {% if csv_b64 %}
                <a class="btn btn-sm btn-outline-secondary"
                   href="data:text/csv;base64,{{ csv_b64 }}"
                   download="forecast_comparison{% if selected_subcat %}_{{ selected_subcat|replace(' ', '_')|lower }}{% endif %}{% if selected_region!='All' %}_{{ selected_region|lower }}{% endif %}.csv">
                   ⬇️ Download CSV
                </a>
              {% endif %}
            </div>
          </div>
        </div>
      {% endif %}

    </div>
  </div>

  <footer class="mt-4 text-center small text-muted">
    Superstore Forecasting &amp; Analytics Portal © 2025
  </footer>
</div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""




def build_top_subcat_chart_base64(df, n=10):
    """
    Robustly compute and plot top-N sub-categories by total Sales.
    Returns base64 PNG or None if no data.
    """
    # Validate columns
    required_cols = {"Sub-Category", "Sales"}
    if not required_cols.issubset(df.columns):
        print(f"[top-products] Missing required columns. Found: {df.columns}")
        return None

    # Coerce Sales to numeric
    df = df.copy()
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])

    if df.empty:
        print("[top-products] DataFrame empty after cleaning.")
        return None

    # Aggregate
    totals = (
        df.groupby("Sub-Category", as_index=False)["Sales"]
          .sum()
          .sort_values("Sales", ascending=False)
          .head(n)
    )
    if totals.empty:
        print("[top-products] No totals computed.")
        return None

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(totals["Sub-Category"], totals["Sales"])
    plt.title(f"Top {len(totals)} Sub-Categories by Total Sales")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return out


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
