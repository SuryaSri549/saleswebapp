import pandas as pd

# Load the dataset
df = pd.read_csv("superstore_extended.csv")

# Ask for user input (region)
region_input = input("Enter Region (Central, East, South, West): ").strip().title()

# Validate region
valid_regions = df['Region'].unique().tolist()
if region_input not in valid_regions:
    print("❌ Invalid region. Please enter one of:", ', '.join(valid_regions))
    exit()

# Filter data by region
region_data = df[df['Region'] == region_input]

# Group by sub-category and sum sales
recommendations = region_data.groupby("Sub-Category")["Sales"].sum().reset_index()

# Sort and take top 3
top_products = recommendations.sort_values(by="Sales", ascending=False).head(3)

# Display results
print(f"\n🎯 Top 3 Product Recommendations for Region: {region_input}")
print(top_products)
