import pandas as pd
import numpy as np

def extend_dataset_realistic(
    input_csv="superstore_extended.csv",
    output_csv="superstore_extended.csv",  # overwrite or change filename
    start_year=2019,
    end_year=2025,
    base_sales_mean=200,
    annual_growth_rate=0.08,
    anomaly_chance=0.05
):
    # Load original data
    df = pd.read_csv(input_csv)
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Extract original data before 2019 (unchanged)
    df_real = df[df['Order Date'].dt.year < start_year].copy()

    # For multipliers and seasonality, use full original dataset
    monthly_avg = df.groupby(df['Order Date'].dt.month)['Sales'].mean()
    monthly_seasonality = monthly_avg / monthly_avg.mean()

    region_mult = df.groupby('Region')['Sales'].mean() / df['Sales'].mean()
    subcat_mult = df.groupby('Sub-Category')['Sales'].mean() / df['Sales'].mean()
    segment_mult = df.groupby('Segment')['Sales'].mean() / df['Sales'].mean()

    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')
    regions = df['Region'].unique()
    subcategories = df['Sub-Category'].unique()
    segments = df['Segment'].unique()

    np.random.seed(42)

    synthetic_rows = []

    for date in dates:
        year_diff = date.year - (start_year - 1)  # years since 2018

        seasonal_marketing_factor = 1.2 if date.month in [11, 12] else 1.0

        for region in regions:
            growth_var = annual_growth_rate + np.random.uniform(-0.02, 0.02)
            r_mult = region_mult.get(region, 1.0)

            for subcat in subcategories:
                s_mult = subcat_mult.get(subcat, 1.0)

                for segment in segments:
                    seg_mult = segment_mult.get(segment, 1.0)

                    sales_base = base_sales_mean * r_mult * s_mult * seg_mult
                    sales_growth = (1 + growth_var) ** year_diff
                    month_factor = monthly_seasonality.get(date.month, 1.0)
                    expected_sales = sales_base * sales_growth * month_factor

                    sales = np.random.normal(loc=expected_sales, scale=expected_sales * 0.15)
                    sales = max(round(sales, 2), 0)

                    if np.random.rand() < anomaly_chance:
                        anomaly_factor = np.random.uniform(-0.3, 0.5)
                        sales = sales * (1 + anomaly_factor)
                        sales = max(round(sales, 2), 0)

                    marketing_spend = sales * np.random.uniform(0.12, 0.28) * seasonal_marketing_factor
                    marketing_spend = round(marketing_spend, 2)

                    discount_mean = max(0.05, min(0.3, 0.25 - (expected_sales / 10000)))
                    discount = np.random.normal(loc=discount_mean, scale=0.05)
                    discount = round(np.clip(discount, 0.05, 0.35), 2)

                    cust_num = np.random.randint(10000, 99999)
                    customer_id = f"CUST-{region[:2].upper()}{segment[:2].upper()}-{cust_num}"
                    customer_name = f"Customer {cust_num}"

                    synthetic_rows.append({
                        'Order Date': date,
                        'Region': region,
                        'Sub-Category': subcat,
                        'Segment': segment,
                        'Sales': sales,
                        'Marketing Spend': marketing_spend,
                        'Discount': discount,
                        'CustomerID': customer_id,
                        'CustomerName': customer_name
                    })

    synthetic_df = pd.DataFrame(synthetic_rows)

    # Combine original (pre-2019) and synthetic (2019-2025)
    combined_df = pd.concat([df_real, synthetic_df], ignore_index=True)

    combined_df.to_csv(output_csv, index=False)
    print(f"✅ Dataset extended realistically from {start_year} to {end_year} and saved to {output_csv}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    extend_dataset_realistic()
