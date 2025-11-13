# 🤖 PredictIQ: Sales Insight Engine

PredictIQ is a **Streamlit‑based sales forecasting application** powered by **XGBoost**. It empowers sales teams and analysts to upload historical sales data, visualize trends, train predictive models, and generate **future sales forecasts** for 3–6 months ahead.

---

## 🚀 Features

- **CSV Upload:** Upload your sales dataset easily. Supports multiple CSV encodings (`UTF‑8`, `ISO‑8859‑1`, etc.).  
- **Data Preprocessing:** Cleans data, handles missing values, sorts by date, and extracts temporal features (Month, Year, Quarter).  
- **Feature Engineering:** Adds lag features, rolling averages, and encodes categorical columns.  
- **Model Training & Loading:**  
  - Automatically loads a pre‑trained XGBoost model if available.  
  - Trains a new XGBoost model if none exists, with early stopping for faster training.  
- **Model Evaluation:** Displays RMSE and R² score along with plots comparing actual vs predicted sales.  
- **Future Forecasting:** Generates predictions for the next 3–6 months with interactive visualizations.  
- **Visualization:** Line plots for historical sales, predictions, and forecasted values.

---

## 📂 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/predictiq-sales-insight.git
cd predictiq-sales-insight

## 💻 Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt

