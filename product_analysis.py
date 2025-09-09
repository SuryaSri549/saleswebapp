import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_region_chart_base64(region, df):
    subcat_sales = df[df['Region'] == region] \
        .groupby('Sub-Category')['Sales'].sum() \
        .sort_values(ascending=False).head(5)

    plt.figure(figsize=(10, 5))
    subcat_sales.plot(kind='bar', color='skyblue')
    plt.title(f"Top 5 Sub-Categories in {region}")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return chart_base64


# ✅ Add this: Monthly Sales Trend
def get_monthly_sales_chart(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()

    plt.figure(figsize=(12, 5))
    monthly_sales.plot(kind='line', marker='o')
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return chart_base64


# ✅ Add this: Top Products Chart
def get_top_products_chart(df):
    top_products = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    top_products.plot(kind='bar', color='orange')
    plt.title("Top 10 Sub-Categories by Sales")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()
    
def generate_filtered_chart(filtered_df, month, _):
    subcat_sales = filtered_df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    monthly_sales.plot(kind='line', marker='o', color='blue')
    plt.title(f"Monthly Sales Trend for '{selected_subcat}'")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return chart_base64



    
