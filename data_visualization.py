import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Inventory Data Visualization")

# Load the DataFrame
try:
    df = pd.read_csv('/content/drive/MyDrive/Grocery_Inventory new v1.csv')
except FileNotFoundError:
    st.error("DataFrame not found. Please ensure 'Grocery_Inventory new v1.csv' is in the correct path.")
    st.stop()

st.write("Explore various visualizations of the grocery inventory data.")

# Visualization 1: Distribution of Product Categories
st.subheader("Distribution of Product Categories")
category_counts = df['Catagory'].value_counts()
fig1, ax1 = plt.subplots(figsize=(10, 6))
category_counts.plot(kind='bar', ax=ax1)
ax1.set_title("Number of Products per Category")
ax1.set_xlabel("Category")
ax1.set_ylabel("Number of Products")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

# Visualization 2: Sales Volume by Status
st.subheader("Sales Volume by Status")
# Ensure Sales_Volume is numeric, handling potential errors
df['Sales_Volume'] = pd.to_numeric(df['Sales_Volume'], errors='coerce')
sales_by_status = df.groupby('Status')['Sales_Volume'].sum().sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sales_by_status.plot(kind='bar', ax=ax2, color=sns.color_palette("viridis", len(sales_by_status)))
ax2.set_title("Total Sales Volume by Inventory Status")
ax2.set_xlabel("Status")
ax2.set_ylabel("Total Sales Volume")
plt.xticks(rotation=0, ha='center')
st.pyplot(fig2)

# Visualization 3: Inventory Turnover Rate Distribution (using Streamlit's native chart)
st.subheader("Inventory Turnover Rate Distribution")
# Ensure Inventory_Turnover_Rate is numeric, handling potential errors
df['Inventory_Turnover_Rate'] = pd.to_numeric(df['Inventory_Turnover_Rate'], errors='coerce')
# Drop rows with NaN in Inventory_Turnover_Rate after coercion
df_cleaned = df.dropna(subset=['Inventory_Turnover_Rate'])

st.bar_chart(df_cleaned['Inventory_Turnover_Rate'].value_counts().sort_index())
st.write("This chart shows the frequency of different inventory turnover rates.")
