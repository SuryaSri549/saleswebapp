import pandas as pd

# Load cleaned data
df = pd.read_csv("cleaned_superstore.csv")

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Calculate Recency: Days since last purchase (assuming today is last date in data)
snapshot_date = df['Order Date'].max()

# Group by Customer ID
customer_metrics = df.groupby('Customer ID').agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Order ID': 'nunique',                                   # Frequency
    'Sales': 'sum'                                           # Monetary value (Total Sales)
}).reset_index()
 
# Rename columns
customer_metrics.rename(columns={
    'Order Date': 'Recency',
    'Order ID': 'Frequency',
    'Sales': 'Monetary'
}, inplace=True)

# Display first few rows
print(customer_metrics.head())

# Save metrics to CSV (optional)
customer_metrics.to_csv("customer_metrics.csv", index=False)
print("✅ Customer metrics saved to customer_metrics.csv")
