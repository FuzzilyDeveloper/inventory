import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import json

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the app data (unique values, numerical means, training columns)
with open('app_data.json', 'r') as f:
    app_data = json.load(f)

unique_values = app_data['unique_values']
numerical_means = app_data['numerical_means']
train_cols = app_data['train_cols']


st.title('Inventory Prediction System')

st.header('Enter Product Details:')

# Numerical features
reorder_level = st.number_input('Reorder Level', min_value=0, value=int(numerical_means['Reorder_Level']), key='reorder_level_input')
reorder_quantity = st.number_input('Reorder Quantity', min_value=0, value=int(numerical_means['Reorder_Quantity']), key='reorder_quantity_input')
unit_price = st.number_input('Unit Price', min_value=0.0, value=numerical_means['Unit_Price'], key='unit_price_input')
sales_volume = st.number_input('Sales Volume', min_value=0, value=int(numerical_means['Sales_Volume']), key='sales_volume_input')
inventory_turnover_rate = st.number_input('Inventory Turnover Rate', min_value=0, value=int(numerical_means['Inventory_Turnover_Rate']), key='inventory_turnover_rate_input')
percentage = st.number_input('Percentage', min_value=-0.1, max_value=1.0, value=numerical_means['percentage'], key='percentage_input')


# Date components (using current date as default)
current_date = pd.to_datetime('today')
date_received_year = st.number_input('Date Received Year', min_value=2024, max_value=2030, value=current_date.year, key='date_received_year_input')
date_received_month = st.number_input('Date Received Month', min_value=1, max_value=12, value=current_date.month, key='date_received_month_input')
date_received_dayofweek = st.number_input('Date Received Day of Week', min_value=0, max_value=6, value=current_date.dayofweek, key='date_received_dayofweek_input')

last_order_date_year = st.number_input('Last Order Date Year', min_value=2024, max_value=2030, value=current_date.year, key='last_order_date_year_input')
last_order_date_month = st.number_input('Last Order Date Month', min_value=1, max_value=12, value=current_date.month, key='last_order_date_month_input')
last_order_date_dayofweek = st.number_input('Last Order Date Day of Week', min_value=0, max_value=6, value=current_date.dayofweek, key='last_order_date_dayofweek_input')

expiration_date_year = st.number_input('Expiration Date Year', min_value=2024, max_value=2030, value=current_date.year, key='expiration_date_year_input')
expiration_date_month = st.number_input('Expiration Date Month', min_value=1, max_value=12, value=current_date.month, key='expiration_date_month_input')
expiration_date_dayofweek = st.number_input('Expiration Date Day of Week', min_value=0, max_value=6, value=current_date.dayofweek, key='expiration_date_dayofweek_input')


# Duration features (will be calculated based on user input dates)


# Categorical features (using unique values from app_data)
catagory = st.selectbox('Catagory', unique_values['Catagory'], key='catagory_input')
supplier_name = st.selectbox('Supplier Name', unique_values['Supplier_Name'], key='supplier_name_input')
warehouse_location = st.selectbox('Warehouse Location', unique_values['Warehouse_Location'], key='warehouse_location_input')
status = st.selectbox('Status', unique_values['Status'], key='status_input')

# Ratio features
inventory_reorder_ratio = st.number_input('Inventory Reorder Ratio', min_value=0.0, value=numerical_means['inventory_reorder_ratio'], key='inventory_reorder_ratio_input')
sales_stock_ratio = st.number_input('Sales Stock Ratio', min_value=0.0, value=numerical_means['sales_stock_ratio'], key='sales_stock_ratio_input')


# Prediction button
predict_button = st.button('Predict Inventory')

if predict_button:
    # Create datetime objects from user inputs for duration calculation
    # Using day 1 as an approximation for duration calculation from year and month
    date_received_dt = pd.to_datetime(f"{date_received_year}-{date_received_month}-01")
    last_order_date_dt = pd.to_datetime(f"{last_order_date_year}-{last_order_date_month}-01")
    expiration_date_dt = pd.to_datetime(f"{expiration_date_year}-{expiration_date_month}-01")

    # Recalculate duration features based on actual date columns (approximated)
    days_to_expiration = (expiration_date_dt - date_received_dt).days
    days_since_last_order = (pd.to_datetime('today') - last_order_date_dt).days

    # Create a dictionary with user inputs
    user_input = {
        'Reorder_Level': reorder_level,
        'Reorder_Quantity': reorder_quantity,
        'Unit_Price': unit_price,
        'Sales_Volume': sales_volume,
        'Inventory_Turnover_Rate': inventory_turnover_rate,
        'percentage': percentage,
        'Date_Received_year': date_received_year,
        'Date_Received_month': date_received_month,
        'Date_Received_dayofweek': date_received_dayofweek,
        'Last_Order_Date_year': last_order_date_year,
        'Last_Order_Date_month': last_order_date_month,
        'Last_Order_Date_dayofweek': last_order_date_dayofweek,
        'Expiration_Date_year': expiration_date_year,
        'Expiration_Date_month': expiration_date_month,
        'Expiration_Date_dayofweek': expiration_date_dayofweek,
        'days_to_expiration': days_to_expiration,
        'days_since_last_order': days_since_last_order,
        'Catagory': catagory,
        'Supplier_Name': supplier_name,
        'Warehouse_Location': warehouse_location,
        'Status': status,
        'inventory_reorder_ratio': inventory_reorder_ratio,
        'sales_stock_ratio': sales_stock_ratio
    }

    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Apply one-hot encoding
    categorical_cols = ['Catagory', 'Supplier_Name', 'Warehouse_Location', 'Status']
    user_df = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)

    # Align columns with the training data - crucial step
    # Use the train_cols loaded from app_data.json for reindexing
    user_df = user_df.reindex(columns=train_cols, fill_value=0)

    # Make prediction
    prediction = model.predict(user_df)
    st.subheader('Predicted Stock Quantity:')
    st.write(f"{prediction[0]:.2f}")

import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if predict_button:
    # Create a dictionary with user inputs
    user_input = {
        'Reorder_Level': reorder_level,
        'Reorder_Quantity': reorder_quantity,
        'Unit_Price': unit_price,
        'Sales_Volume': sales_volume,
        'Inventory_Turnover_Rate': inventory_turnover_rate,
        'Date_Received_year': date_received_year,
        'Date_Received_month': date_received_month,
        'Date_Received_dayofweek': date_received_dayofweek,
        'Last_Order_Date_year': last_order_date_year,
        'Last_Order_Date_month': last_order_date_month,
        'Last_Order_Date_dayofweek': last_order_date_dayofweek,
        'Expiration_Date_year': expiration_date_year,
        'Expiration_Date_month': expiration_date_month,
        'Expiration_Date_dayofweek': expiration_date_dayofweek,
        'days_to_expiration': days_to_expiration,
        'days_since_last_order': days_since_last_order,
        'Catagory': catagory,
        'Supplier_Name': supplier_name,
        'Warehouse_Location': warehouse_location,
        'Status': status,
        'inventory_reorder_ratio': inventory_reorder_ratio,
        'sales_stock_ratio': sales_stock_ratio
    }

    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Recreate original date columns for duration calculation
    user_df['Date_Received'] = pd.to_datetime(f"{user_df['Date_Received_year']}-{user_df['Date_Received_month']}-01") # Approx day
    user_df['Last_Order_Date'] = pd.to_datetime(f"{user_df['Last_Order_Date_year']}-{user_df['Last_Order_Date_month']}-01") # Approx day
    user_df['Expiration_Date'] = pd.to_datetime(f"{user_df['Expiration_Date_year']}-{user_df['Expiration_Date_month']}-01") # Approx day

    # Recalculate duration features based on actual date columns (approximated)
    # user_df['days_to_expiration'] = (user_df['Expiration_Date'] - user_df['Date_Received']).dt.days
    # user_df['days_since_last_order'] = (pd.to_datetime('today') - user_df['Last_Order_Date']).dt.days


    # Apply one-hot encoding
    categorical_cols = ['Catagory', 'Supplier_Name', 'Warehouse_Location', 'Status']
    user_df = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)

    # Align columns with the training data - crucial step
    # Get the list of columns from the training data
    train_cols = user_df.columns
    user_df = user_df.reindex(columns=train_cols, fill_value=0)

    # Drop the temporary date columns created for duration calculation
    user_df = user_df.drop(['Date_Received', 'Last_Order_Date', 'Expiration_Date'], axis=1)

    # Display the preprocessed data (optional, for debugging)
    # st.write("Preprocessed User Input:")
    # st.write(user_df)

    # Make prediction (This part will be in the next step)
    # prediction = model.predict(user_df)
    # st.subheader('Predicted Stock Quantity:')
    # st.write(prediction[0])

    # Make prediction
    prediction = model.predict(user_df)
    st.subheader('Predicted Stock Quantity:')
    st.write(f"{prediction[0]:.2f}")
