# Streamlit + XGBoost + Model Save/Load + Auto Forecast

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# ------------------------------------------------
# 1️⃣ Streamlit Configuration
# ------------------------------------------------
st.set_page_config(page_title="PredictIQ: Sales Insight Engine", layout="wide")

st.title("🤖 PredictIQ: Sales Insight Engine")
st.markdown("Empowering sales teams with data-driven forecasts and actionable business intelligence.")

MODEL_FILE = "xgboost_sales_model.pkl"

# ------------------------------------------------
# 2️⃣ Upload Data
# ------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your sales dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # ------------------------------------------------
    # 3️⃣ Data Preprocessing
    # ------------------------------------------------
    st.markdown("### 🧹 Data Preprocessing")

    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
    df = df.dropna(subset=['ORDERDATE', 'SALES'])
    df = df.sort_values('ORDERDATE')

    # Extract temporal features
    df['Month'] = df['ORDERDATE'].dt.month
    df['Year'] = df['ORDERDATE'].dt.year
    df['Quarter'] = df['ORDERDATE'].dt.quarter

    # Lag and rolling averages
    df['SALES_lag_1'] = df['SALES'].shift(1)
    df['SALES_lag_2'] = df['SALES'].shift(2)
    df['SALES_roll_3'] = df['SALES'].rolling(window=3).mean()
    df['SALES_roll_6'] = df['SALES'].rolling(window=6).mean()
    df = df.dropna()

    # Encode categoricals
    cat_cols = ['PRODUCTLINE', 'DEALSIZE', 'COUNTRY', 'TERRITORY']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    features = [
        'QUANTITYORDERED', 'PRICEEACH', 'MSRP',
        'Month', 'Year', 'Quarter',
        'QTR_ID', 'MONTH_ID', 'YEAR_ID',
        'PRODUCTLINE', 'DEALSIZE', 'COUNTRY', 'TERRITORY',
        'SALES_lag_1', 'SALES_lag_2', 'SALES_roll_3', 'SALES_roll_6'
    ]
    features = [f for f in features if f in df.columns]
    target = 'SALES'

    X = df[features]
    y = df[target]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # ------------------------------------------------
    # 4️⃣ Load or Train Model
    # ------------------------------------------------
    if os.path.exists(MODEL_FILE):
        st.info("📦 Pre-trained model found — loading it for instant predictions.")
        model = joblib.load(MODEL_FILE)
    else:
        st.warning("⚙️ No saved model found — training new XGBoost model...")
        params = {
            "n_estimators": [300, 400],
            "learning_rate": [0.03, 0.05],
            "max_depth": [6, 8],
            "subsample": [0.8, 1.0]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        base_model = XGBRegressor(random_state=42)
        grid = GridSearchCV(base_model, params, cv=tscv, scoring="r2", n_jobs=-1)
        with st.spinner("Training model... This may take a minute"):
            grid.fit(X_train, y_train)
        model = grid.best_estimator_
        joblib.dump(model, MODEL_FILE)
        st.success(f"✅ Model trained and saved! (R² = {grid.best_score_:.3f})")

    # ------------------------------------------------
    # 5️⃣ Model Evaluation
    # ------------------------------------------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:,.2f}")
    col2.metric("R² Score", f"{r2:.4f}")

    results = pd.DataFrame({
        'Date': df['ORDERDATE'].iloc[split_index:],
        'Actual': y_test.values,
        'Predicted': y_pred
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['Date'], results['Actual'], label='Actual', color='blue')
    ax.plot(results['Date'], results['Predicted'], label='Predicted', color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------
    # 6️⃣ Future Forecast
    # ------------------------------------------------
    st.markdown("### 🔮 Auto Forecast (Next 3–6 Months)")

    horizon = st.slider("Forecast Months Ahead", 3, 6, 3)
    last_data = df.iloc[-1:].copy()
    forecast = []

    for i in range(horizon):
        next_month = (last_data['Month'].values[0] % 12) + 1
        next_year = last_data['Year'].values[0] + (1 if next_month == 1 else 0)
        next_quarter = (next_month - 1) // 3 + 1

        new_row = last_data.copy()
        new_row['Month'] = next_month
        new_row['Year'] = next_year
        new_row['Quarter'] = next_quarter

        new_row['SALES_lag_1'] = last_data['SALES'].values[0]
        new_row['SALES_lag_2'] = last_data['SALES_lag_1'].values[0]
        new_row['SALES_roll_3'] = np.mean([last_data['SALES'].values[0], last_data['SALES_lag_1'].values[0], last_data['SALES_lag_2'].values[0]])
        new_row['SALES_roll_6'] = new_row['SALES_roll_3']

        pred = model.predict(new_row[features])[0]
        new_row['SALES'] = pred
        forecast.append({'Month': next_month, 'Year': next_year, 'Predicted_Sales': pred})

        last_data = new_row.copy()

    forecast_df = pd.DataFrame(forecast)
    st.dataframe(forecast_df.style.format({"Predicted_Sales": "${:,.2f}"}))

    # Forecast Visualization
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(results['Date'], results['Predicted'], label='Past Predicted', color='orange')
    ax2.plot(
        pd.date_range(df['ORDERDATE'].iloc[-1], periods=horizon + 1, freq='M')[1:],
        forecast_df['Predicted_Sales'],
        label='Forecasted',
        color='green',
        marker='o'
    )
    ax2.set_title("Future Sales Forecast (Next Months)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Predicted Sales")
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Forecast Complete – Model Ready & Saved.")
else:
    st.info("📂 Please upload a sales dataset to start.")
