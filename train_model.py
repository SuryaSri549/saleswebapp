import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load and clean the data
df = pd.read_csv("superstore_extended.csv")

# Select only necessary columns
df = df[['Sales', 'Discount', 'Region', 'Marketing Spend']]  # Assuming marketing spend is included

# Drop any missing values
df = df.dropna()

# One-hot encode Region
region_dummies = pd.get_dummies(df['Region'])

# Combine features
X = pd.concat([df[['Marketing Spend', 'Discount']], region_dummies], axis=1)
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("sales_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved with: Marketing Spend, Discount, Region")
