import streamlit as st
import inventory_prediction
import data_visualization
import contact_information

st.set_page_config(
    page_title="Grocery Inventory App",
    layout="wide"
)

st.sidebar.title("Navigation")

pages = {
    "Inventory Prediction": inventory_prediction,
    "Data Visualization": data_visualization,
    "Contact Information": contact_information
}

selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Execute the selected page's code
if selected_page == "Inventory Prediction":
    inventory_prediction.st.title("Inventory Prediction") # Re-add title as each page script runs fully
    inventory_prediction.st.write("This page will contain the inventory prediction model.")
    # Rerun the content of the page script
    with open("inventory_prediction.py", "r") as f:
        code = f.read()
    exec(code)
elif selected_page == "Data Visualization":
    data_visualization.st.title("Inventory Data Visualization") # Re-add title
    data_visualization.st.write("Explore various visualizations of the grocery inventory data.")
    # Rerun the content of the page script
    with open("data_visualization.py", "r") as f:
        code = f.read()
    exec(code)
elif selected_page == "Contact Information":
    contact_information.st.title("Contact Information") # Re-add title
    contact_information.st.write("Please feel free to reach out with any questions or feedback regarding the Grocery Inventory Management application.")
    # Rerun the content of the page script
    with open("contact_information.py", "r") as f:
        code = f.read()
    exec(code)
