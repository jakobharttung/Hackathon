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

    # Prepare the cleaned dataset for machine learning
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

    # Filter numeric features for machine learning
    numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    selected_features = st.sidebar.multiselect("Select Features for Correlation Matrix", numeric_columns, default=numeric_columns[:10])

    # --- Visualization ---
    st.write("### Yield Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_cleaned[yield_column], kde=True, ax=ax)
    ax.set_title('Yield Distribution')
    st.pyplot(fig)

    # --- Variability in Yield using Scatter Plot with Day of the Month ---
    st.write("### Variability in Yield")
    fig, ax = plt.subplots()
    df_cleaned['day'] = df_cleaned[manufacture_date_column].dt.day  # Extract day of the month
    ax.scatter(df_cleaned['day'], df_cleaned[yield_column])

    # Set custom y-axis limits
    lower_bound = df_cleaned[yield_column].quantile(0.05)  # 5th percentile
    upper_bound = df_cleaned[yield_column].quantile(0.95)  # 95th percentile
    ax.set_ylim([lower_bound, upper_bound])  # Set y-axis limits

    # Limit x-axis to January (days 1 to 31)
    ax.set_xlim([1, 31])
    ax.set_xticks(range(1, 32))

    ax.set_xlabel('Day of the Month (January)')
    ax.set_ylabel('Yield')
    ax.set_title('Yield Variability Over Days of January')
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
    X, y = prepare_features(df_cleaned, selected_features, yield_column, manufacture_date_column)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model Performance:")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    # --- Feature Importance ---
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({'Feature': selected_features + [manufacture_date_column], 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # --- SHAP values for interpretability ---
    st.write("### SHAP Analysis (Top 5 Features)")
    
    try:
        # Ensure SHAP values are calculated for strictly numeric input
        X_test = X_test.astype(np.float64)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap.initjs()
        shap.summary_plot(shap_values, X_test, feature_names=selected_features + [manufacture_date_column], max_display=5, plot_type="bar")
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"Error generating SHAP values: {e}")

    # --- Parameter Optimization ---
    st.write("### Suggested Parameter Ranges for Yield Improvement")
    
    top_features = feature_importance['Feature'].head(5).values
    st.write(f"Top 5 parameters driving yield: {', '.join(top_features)}")

    st.write("Based on SHAP analysis, adjusting these parameters could lead to yield improvements. Here is a detailed analysis of optimal ranges and expected yield improvements:")

    for feature in top_features:
        st.write(f"**{feature}:** Adjust the range between X and Y (replace X and Y with specific values from analysis) to potentially increase yield by Z%.")

else:
    st.write("Please upload a dataset to begin analysis.")
