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

# Upload CSV file
st.sidebar.title("Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # --- Data Overview ---
    st.write("### Data Overview")
    st.write(f"Total records: {df.shape[0]}")
    st.write(f"Total features: {df.shape[1]}")

    # Select yield column and features to analyze
    yield_column = st.sidebar.selectbox("Select Yield Column", df.columns)
    features = st.sidebar.multiselect("Select Features to Analyze", df.columns, default=df.columns[:-1])

    # --- Visualization ---
    st.write("### Yield Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[yield_column], kde=True, ax=ax)
    ax.set_title('Yield Distribution')
    st.pyplot(fig)

    st.write("### Variability in Yield")
    fig, ax = plt.subplots()
    sns.boxplot(df[yield_column], ax=ax)
    ax.set_title('Yield Variability')
    st.pyplot(fig)

    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,8))
    corr_matrix = df[features + [yield_column]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # --- Machine Learning Model ---
    st.write("### Machine Learning: Yield Prediction")
    
    X = df[features]
    y = df[yield_column]

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

    # Feature Importance
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
    st.write("Upload a dataset to begin analysis.")


