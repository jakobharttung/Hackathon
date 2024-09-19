import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import numpy as np

# Set page title
st.title('Yield Optimization Dashboard')

# Function to clean and validate data
def clean_and_validate_data(df, yield_column, features):
    # Convert yield column to numeric, invalid parsing will result in NaN
    df[yield_column] = pd.to_numeric(df[yield_column], errors='coerce')
    
    # Convert feature columns to numeric, invalid parsing will result in NaN
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows where yield or any selected features have NaN values
    df_cleaned = df.dropna(subset=[yield_column] + features)
    
    return df_cleaned

# Function to prepare the features for the model
def prepare_features(df, features, yield_column):
    # Prepare the cleaned dataset for machine learning
    X = df[features]  # Features
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

    # Convert manufacture_date to datetime if it exists in the dataset
    if 'manufacture_date' in df.columns:
        try:
            df['manufacture_date'] = pd.to_datetime(df['manufacture_date'], errors='coerce')
            df['month'] = df['manufacture_date'].dt.strftime('%b')  # Extract month name
        except Exception as e:
            st.warning("Error converting 'manufacture_date' to datetime")

    # Show a preview of the dataset
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # --- Data Overview ---
    st.write("### Data Overview")
    st.write(f"Total records: {df.shape[0]}")
    st.write(f"Total features: {df.shape[1]}")

    # Select yield column and features to analyze
    yield_column = st.sidebar.selectbox("Select Yield Column", df.columns)
    
    # Filter numeric features for machine learning
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    features = st.sidebar.multiselect("Select Features to Analyze", numeric_columns, default=numeric_columns)

    # --- Data Cleansing and Validation ---
    df_cleaned = clean_and_validate_data(df, yield_column, features)
    
    # If the dataset is empty after cleansing, show an error
    if df_cleaned.empty:
        st.error("No valid data available after cleaning. Please check the dataset and try again.")
        st.stop()

    # --- Visualization ---
    st.write("### Yield Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_cleaned[yield_column], kde=True, ax=ax)
    ax.set_title('Yield Distribution')
    st.pyplot(fig)

    # --- Variability in Yield using Scatter Plot with Months and Custom Y Axis ---
    st.write("### Variability in Yield")
    if 'manufacture_date' in df_cleaned.columns:
        fig, ax = plt.subplots()

        # Scatter plot with month on x-axis and yield on y-axis
        df_cleaned['month_numeric'] = df_cleaned['manufacture_date'].dt.month  # Extract month as a number for better plotting
        ax.scatter(df_cleaned['month_numeric'], df_cleaned[yield_column])

        # Customize the x-axis to show only months
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # Set custom y-axis limits to focus on central range of yield values
        lower_bound = df_cleaned[yield_column].quantile(0.05)  # 5th percentile
        upper_bound = df_cleaned[yield_column].quantile(0.95)  # 95th percentile
        ax.set_ylim([lower_bound, upper_bound])  # Set y-axis limits

        ax.set_xlabel('Month')
        ax.set_ylabel('Yield')
        ax.set_title('Yield Variability Over Months')
        st.pyplot(fig)
    else:
        st.warning("Manufacture Date column is not available for plotting")

    # --- Correlation Matrix ---
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,8))
    corr_matrix = df_cleaned[features + [yield_column]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # --- Machine Learning Model ---
    st.write("### Machine Learning: Yield Prediction")
    
    # Prepare the features and target variable using the custom function
    X, y = prepare_features(df_cleaned, features, yield_column)

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
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
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
        shap.summary_plot(shap_values, X_test, feature_names=features, max_display=5, plot_type="bar")
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
