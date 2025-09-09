import pandas as pd

# Load the dataset
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')


# Show first few rows to understand the data
print(df.head())

# Show column names and data types
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Select important columns
columns_to_keep = [
    'Order ID', 'Order Date', 'Sales', 'Quantity', 'Profit', 'Discount',
    'Customer ID', 'Customer Name', 'Segment', 'Region', 'State', 'City',
    'Product ID', 'Category', 'Sub-Category'
]

df_selected = df[columns_to_keep]

# Show first few rows of cleaned data
print(df_selected.head())

# Save to a new CSV (optional)
df_selected.to_csv("cleaned_superstore.csv", index=False)
print("✅ Cleaned data saved as cleaned_superstore.csv")
