import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import numpy as np
from datetime import datetime

# Set page title
st.title('Yield Optimization Dashboard')

# Function to clean and validate data
def clean_and_validate_data(df, yield_column, manufacture_date_column):
    # Ensure that the yield column is numeric
    df[yield_column] = pd.to_numeric(df[yield_column], errors='coerce')
    
    # Convert integer timestamp to datetime
    df[manufacture_date_column] = pd.to_datetime(df[manufacture_date_column], unit='s', errors='coerce')

    # Drop rows where yield or manufacture date is NaN
    df_cleaned = df.dropna(subset=[yield_column, manufacture_date_column])
    
    return df_cleaned

# Function to prepare features for machine learning
def prepare_features(df, features, yield_column, manufacture_date_column):
    # Convert manufacture date to numeric timestamp (to be used as a feature)
    df[manufacture_date_column] = df[manufacture_date_column].apply(lambda x: x.timestamp())

    # Prepare the cleaned dataset for machine learning (excluding yield_value)
    X = df[features + [manufacture_date_column]]  # Features + manufacture date
    y = df[yield_column]  # Target
    
    return X, y

# Upload CSV file
st.sidebar.title("Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Automatically set yield column and manufacture date column
    yield_column = 'yield_value'
    manufacture_date_column = 'manufacture_date'

    # Clean and validate data
    df_cleaned = clean_and_validate_data(df, yield_column, manufacture_date_column)
    
    # If the dataset is empty after cleansing, show an error
    if df_cleaned.empty:
        st.error("No valid data available after cleaning. Please check the dataset and try again.")
        st.stop()

    # Display the cleaned dataset
    st.write("### Cleaned Dataset Preview")
    st.dataframe(df_cleaned.head())

    # --- Data Overview ---
    st.write("### Data Overview")
    st.write(f"Total records: {df_cleaned.shape[0]}")
    st.write(f"Total features: {df_cleaned.shape[1]}")

    # Filter numeric features for machine learning, excluding yield_value
    numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != yield_column]  # Exclude yield_value
    selected_features = st.sidebar.multiselect("Select Features for Correlation Matrix", numeric_columns, default=["quantity"] + numeric_columns[:9])

    # --- Yield Distribution Visualization ---
    st.write("### Yield Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_cleaned[yield_column], kde=True, ax=ax)
    ax.set_title('Yield Distribution')
    st.pyplot(fig)

    # --- Correlation Matrix ---
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,8))
    corr_matrix = df_cleaned[selected_features + [yield_column]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # --- Machine Learning Model ---
    st.write("### Machine Learning: Yield Prediction")
    
    # Prepare features for machine learning, including manufacture date
    X, y = prepare_features
