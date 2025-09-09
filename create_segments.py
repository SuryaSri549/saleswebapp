import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load customer metrics
df = pd.read_csv("customer_metrics.csv")

# Select features for clustering
features = df[['Recency', 'Frequency', 'Monetary']]

# Scale features (important for KMeans)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Create KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Map numeric segments to descriptive names (optional)
segment_labels = {
    0: "Loyal",
    1: "At Risk",
    2: "New"
}

# Assign labels (you can adjust based on analysis)
df['Segment Label'] = df['Segment'].map(segment_labels)

# Save segmented customers
df.to_csv("segmented_customers.csv", index=False)

print(df[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Segment Label']].head())
print("✅ Segmented customers saved to segmented_customers.csv")
