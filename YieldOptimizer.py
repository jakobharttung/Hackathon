import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

# Set page title
st.title('Yield Optimization Dashboard')

# Function to prepare the features for the model
def prepare_features(df, features, yield_column):
    # Ensure only numeric columns are passed for features
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coerce invalids to NaN
    df[yield_column] = pd.to_numeric(df[yield_column], errors='coerce')  # Ensure yield column is numeric
    
    # Drop rows with NaN values in the selected features and yield column
    df_cleaned = df.dropna(subset=features + [yield_column])
    
    X = df_cleaned[features]  # Features
    y = df_cleaned[yield_column]  # Target
    
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

    # --- Visualization ---
    st.write("### Yield Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[yield_column], kde=True, ax=ax)
    ax.set_title('Yield Distribution')
    st.pyplot(fig)

    # --- Variability in Yield using Scatter Plot with Months and Custom Y Axis ---
    st.write("### Variability in Yield")
    if 'manufacture_date' in df.columns:
        fig, ax = plt.subplots()

        # Scatter plot with month on x-axis and yield on y-axis
        df['month_numeric'] = df['manufacture_date'].dt.month  # Extract month as a number for better plotting
        ax.scatter(df['month_numeric'], df[yield_column])

        # Customize the x-axis to show only months
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # Set custom y-axis with about 10 sub-steps
        ax.set_yticks([round(i, 2) for i in plt.MaxNLocator(10).tick_values(df[yield_column].min(), df[yield_column].max())])
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Yield')
        ax.set_title('Yield Variability Over Months')
        st.pyplot(fig)
    else:
        st.warning("Manufacture Date column is not available for plotting")

    # --- Correlation Matrix ---
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,8))
    corr_matrix = df[features + [yield_column]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # --- Machine Learning Model ---
    st.write("### Machine Learning: Yield Prediction")
    
    # Prepare the features and target variable using the custom function
    X, y = prepare_features(df, features, yield_column)

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
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.initjs()
    shap.summary_plot(shap_values, X_test, feature_names=features, max_display=5, plot_type="bar")
    st.pyplot(bbox_inches='tight')

    # --- Parameter Optimization ---
    st.write("### Suggested Parameter Ranges for Yield Improvement")
    
    top_features = feature_importance['Feature'].head(5).values
    st.write(f"Top 5 parameters driving yield: {', '.join(top_features)}")

    st.write("Based on SHAP analysis, adjusting these parameters could lead to yield improvements. Here is a detailed analysis of optimal ranges and expected yield improvements:")

    for feature in top_features:
        st.write(f"**{feature}:** Adjust the range between X and Y (replace X and Y with specific values from analysis) to potentially increase yield by Z%.")

else:
    st.write("Please upload a dataset to begin analysis.")
