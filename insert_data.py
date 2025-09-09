import pandas as pd
import sqlite3

# Load your cleaned data
df = pd.read_csv("cleaned_superstore.csv")

# Connect to database
conn = sqlite3.connect("superstore.db")
cursor = conn.cursor()

# Insert Customers
customers = df[['Customer ID', 'Customer Name', 'Segment', 'Region', 'State', 'City']].drop_duplicates()
for _, row in customers.iterrows():
    cursor.execute("""
        INSERT OR IGNORE INTO Customers (CustomerID, CustomerName, Segment, Region, State, City)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (row['Customer ID'], row['Customer Name'], row['Segment'], row['Region'], row['State'], row['City']))

# Insert Products
products = df[['Product ID', 'Category', 'Sub-Category']].drop_duplicates()
for _, row in products.iterrows():
    cursor.execute("""
        INSERT OR IGNORE INTO Products (ProductID, Category, SubCategory)
        VALUES (?, ?, ?)
    """, (row['Product ID'], row['Category'], row['Sub-Category']))

# Insert Orders
orders = df[['Order ID', 'Order Date', 'Sales', 'Quantity', 'Discount', 'Profit', 'Customer ID', 'Product ID']].drop_duplicates()
for _, row in orders.iterrows():
    cursor.execute("""
        INSERT OR IGNORE INTO Orders (OrderID, OrderDate, Sales, Quantity, Discount, Profit, CustomerID, ProductID)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (row['Order ID'], row['Order Date'], row['Sales'], row['Quantity'], row['Discount'], row['Profit'], row['Customer ID'], row['Product ID']))

# Save and close
conn.commit()
conn.close()

print("✅ Data inserted into database!")
