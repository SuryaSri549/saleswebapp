import sqlite3

# Create (or connect to) database file
conn = sqlite3.connect("superstore.db")
cursor = conn.cursor()

# Create Customers table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Customers (
    CustomerID TEXT PRIMARY KEY,
    CustomerName TEXT,
    Segment TEXT,
    Region TEXT,
    State TEXT,
    City TEXT
)
""")

# Create Products table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Products (
    ProductID TEXT PRIMARY KEY,
    Category TEXT,
    SubCategory TEXT
)
""")

# Create Orders table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Orders (
    OrderID TEXT PRIMARY KEY,
    OrderDate TEXT,
    Sales REAL,
    Quantity INTEGER,
    Discount REAL,
    Profit REAL,
    CustomerID TEXT,
    ProductID TEXT,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
)
""")

# Save changes and close
conn.commit()
conn.close()

print("✅ Database schema created: superstore.db")
