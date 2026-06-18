import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generate realistic sales dataset
n = 1000
regions = ['North', 'South', 'East', 'West']
categories = ['Electronics', 'Clothing', 'Furniture', 'Groceries', 'Sports']
months = pd.date_range('2023-01-01', '2023-12-31', periods=n)

data = {
    'Date': months,
    'Region': np.random.choice(regions, n, p=[0.3, 0.25, 0.25, 0.2]),
    'Category': np.random.choice(categories, n, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    'Sales': np.random.randint(500, 15000, n),
    'Profit': np.random.randint(50, 3000, n),
    'Units_Sold': np.random.randint(1, 100, n),
    'Customer_Age': np.random.randint(18, 65, n),
    'Customer_Gender': np.random.choice(['Male', 'Female'], n),
    'Discount': np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
}

df = pd.DataFrame(data)
df['Month'] = df['Date'].dt.month_name()
df['Quarter'] = df['Date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
df['Profit_Margin'] = ((df['Profit'] / df['Sales']) * 100).round(2)
df.to_csv('sales_data.csv', index=False)

# ---- EDA ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sales Performance Dashboard - 2023', fontsize=16, fontweight='bold', y=1.01)

# 1. Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
axes[0,0].bar(region_sales.index, region_sales.values, color=['#2E86AB','#A23B72','#F18F01','#C73E1D'])
axes[0,0].set_title('Total Sales by Region')
axes[0,0].set_ylabel('Total Sales (₹)')
for i, v in enumerate(region_sales.values):
    axes[0,0].text(i, v + 50000, f'₹{v:,.0f}', ha='center', fontsize=9)

# 2. Sales by Category
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
colors_cat = ['#264653','#2a9d8f','#e9c46a','#f4a261','#e76f51']
axes[0,1].barh(cat_sales.index, cat_sales.values, color=colors_cat)
axes[0,1].set_title('Sales by Product Category')
axes[0,1].set_xlabel('Total Sales (₹)')

# 3. Monthly Revenue Trend
monthly = df.groupby(df['Date'].dt.month)['Sales'].sum()
axes[0,2].plot(monthly.index, monthly.values, marker='o', color='#2E86AB', linewidth=2)
axes[0,2].fill_between(monthly.index, monthly.values, alpha=0.2, color='#2E86AB')
axes[0,2].set_title('Monthly Revenue Trend')
axes[0,2].set_xlabel('Month')
axes[0,2].set_ylabel('Sales (₹)')
axes[0,2].set_xticks(range(1,13))

# 4. Profit Margin by Category
margin = df.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=False)
bars = axes[1,0].bar(margin.index, margin.values, color='#2a9d8f')
axes[1,0].set_title('Avg Profit Margin by Category (%)')
axes[1,0].set_ylabel('Profit Margin (%)')
axes[1,0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, margin.values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}%', ha='center', fontsize=9)

# 5. Quarterly Sales Comparison
q_sales = df.groupby(['Quarter','Region'])['Sales'].sum().unstack()
q_sales.plot(kind='bar', ax=axes[1,1], colormap='Set2')
axes[1,1].set_title('Quarterly Sales by Region')
axes[1,1].set_xlabel('Quarter')
axes[1,1].set_ylabel('Sales (₹)')
axes[1,1].tick_params(axis='x', rotation=0)
axes[1,1].legend(title='Region', fontsize=8)

# 6. Discount vs Profit Margin
disc_margin = df.groupby('Discount')['Profit_Margin'].mean()
axes[1,2].bar([f'{int(d*100)}%' for d in disc_margin.index], disc_margin.values, color='#e76f51')
axes[1,2].set_title('Discount Rate vs Avg Profit Margin')
axes[1,2].set_xlabel('Discount %')
axes[1,2].set_ylabel('Avg Profit Margin (%)')

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Summary stats ----
print("=== KEY INSIGHTS ===")
print(f"Total Revenue: ₹{df['Sales'].sum():,.0f}")
print(f"Total Profit: ₹{df['Profit'].sum():,.0f}")
print(f"Avg Profit Margin: {df['Profit_Margin'].mean():.1f}%")
print(f"Best Region: {region_sales.index[0]} (₹{region_sales.iloc[0]:,.0f})")
print(f"Best Category: {cat_sales.index[0]}")
print(f"Peak Month: {monthly.idxmax()}")
print(f"Total Orders: {len(df)}")
