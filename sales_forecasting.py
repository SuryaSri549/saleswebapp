import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import os

# Show current working directory
print("📁 Current working directory:", os.getcwd())

# Load cleaned Superstore data
df = pd.read_csv("cleaned_superstore.csv")

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# -------------------------------
# PART 1: Monthly Sales Forecasting
# -------------------------------
df['Month'] = df['Order Date'].dt.to_period('M')
monthly_grouped = df.groupby('Month')['Sales'].sum().reset_index()
monthly_grouped['Month'] = monthly_grouped['Month'].astype(str)
monthly_grouped.to_csv("monthly_sales.csv", index=False)
print("✅ Saved monthly grouped sales to monthly_sales.csv")

# Resample monthly for time series modeling
df.set_index('Order Date', inplace=True)
monthly_sales = df['Sales'].resample('M').sum().reset_index()
monthly_sales['MonthIndex'] = range(1, len(monthly_sales) + 1)

# Train monthly forecasting model
X_monthly = monthly_sales['MonthIndex'].values.reshape(-1, 1)
y_monthly = monthly_sales['Sales'].values

model_monthly = LinearRegression()
model_monthly.fit(X_monthly, y_monthly)

# Save monthly model
with open("sales_forecast_model_monthly.pkl", "wb") as f:
    pickle.dump(model_monthly, f)
print("✅ Trained monthly model saved as sales_forecast_model_monthly.pkl")

# Save data for forecasting
monthly_sales[['Order Date', 'Sales']].to_csv("sales_forecast_data_monthly.csv", index=False)
print("✅ Saved forecast data to sales_forecast_data_monthly.csv")

# -------------------------------
# PART 2: Yearly Sales Forecasting
# -------------------------------
df.reset_index(inplace=True)  # Reset index after resampling
df['Year'] = df['Order Date'].dt.year
yearly_grouped = df.groupby('Year')['Sales'].sum().reset_index()
yearly_grouped['YearIndex'] = range(1, len(yearly_grouped) + 1)

# Train yearly forecasting model
X_yearly = yearly_grouped['YearIndex'].values.reshape(-1, 1)
y_yearly = yearly_grouped['Sales'].values

model_yearly = LinearRegression()
model_yearly.fit(X_yearly, y_yearly)

# Save yearly model
with open("sales_forecast_model_yearly.pkl", "wb") as f:
    pickle.dump(model_yearly, f)
print("✅ Trained yearly model saved as sales_forecast_model_yearly.pkl")

# Save data for yearly forecasting
yearly_grouped[['Year', 'Sales']].to_csv("sales_forecast_data_yearly.csv", index=False)
print("✅ Saved yearly sales data to sales_forecast_data_yearly.csv")

# -------------------------------
# OPTIONAL: Plot Comparison
# -------------------------------
plt.figure(figsize=(12, 5))

# Monthly
plt.subplot(1, 2, 1)
plt.plot(monthly_sales['Order Date'], y_monthly, label='Actual Monthly Sales')
plt.plot(monthly_sales['Order Date'], model_monthly.predict(X_monthly), linestyle='--', label='Trend')
plt.title("📅 Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)

# Yearly
plt.subplot(1, 2, 2)
plt.plot(yearly_grouped['Year'], y_yearly, marker='o', label='Actual Yearly Sales')
plt.plot(yearly_grouped['Year'], model_yearly.predict(X_yearly), linestyle='--', label='Trend')
plt.title("📆 Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.legend()

plt.tight_layout()
plt.show()
