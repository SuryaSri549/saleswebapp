import pandas as pd

# Load old dataset
df_old = pd.read_csv("superstore_extended.csv", parse_dates=["Order Date"])

# Load new dataset
df_new = pd.read_csv("superstore_2019_2022.csv", parse_dates=["order_date"])

# Rename columns in new dataset to match old schema
df_new.rename(columns={
    'order_id': 'Order ID',
    'order_date': 'Order Date',
    'ship_date': 'Ship Date',
    'customer': 'Customer Name',
    'segment': 'Segment',
    'region': 'Region',
    'subcategory': 'Sub-Category',
    'sales': 'Sales',
    'discount': 'Discount'
}, inplace=True)

# Optional: Add missing columns with placeholder values if required
for col in ['CustomerID', 'Marketing Spend']:
    if col not in df_new.columns:
        df_new[col] = "Unknown" if col == 'CustomerID' else 0.0

# Select only matching columns between the two DataFrames
common_columns = df_old.columns.intersection(df_new.columns)
df_old = df_old[common_columns]
df_new = df_new[common_columns]

# Combine datasets
merged_df = pd.concat([df_old, df_new], ignore_index=True)

# Save to new CSV
merged_df.to_csv("superstore_merged_2013_2022.csv", index=False)
print("✅ Merged dataset saved as superstore_merged_2013_2022.csv")
