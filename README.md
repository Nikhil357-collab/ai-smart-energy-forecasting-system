# ai-smart-energy-forecasting-system
AI-powered smart grid energy forecasting system using Machine Learning, Streamlit, and real-world PJM electricity demand datasets with real-time visualization and automated overload alerts.
# ⚡ AI Smart Energy Forecasting System

An AI-powered smart-grid forecasting platform that predicts future electricity demand using Machine Learning and real-world PJM energy datasets.

---

# 🚀 Features

- 🔋 Smart Grid Energy Forecasting
- 📈 Real-Time Prediction Dashboard
- 🧠 MLP Neural Network Model
- 📂 Dynamic CSV Upload
- 🚨 Automated Email Alert System
- 📊 Interactive Plotly Visualizations
- ⚡ Energy Demand Prediction in MW
- 🌍 Smart Energy Analytics

---

# 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Joblib
- Yagmail

---

# 📊 Dataset

Dataset Used:
- PJM Hourly Energy Consumption Dataset

The dataset contains historical smart-grid electricity demand data measured in Megawatts (MW).

---

# 🧠 Machine Learning Model

Model Used:
- MLP Regressor (Multi-Layer Perceptron)

Feature Engineering:
- Hour
- Day of Week
- Month
- Weekend Detection
- Lag Features
- Rolling Mean Features

---

# 📈 Model Performance

| Metric | Value |
|---|---|
| MAE | 353.64 |
| RMSE | 444.97 |
| R² Score | 0.9705 |

---

# ⚡ Real-World Impact

AI-based energy forecasting helps:
- Reduce power wastage
- Improve smart-grid stability
- Prevent overload conditions
- Optimize industrial energy usage
- Support smart city infrastructure

---

# 🚨 Alert System

The system automatically sends email alerts when predicted energy demand exceeds safe thresholds.

---

# ▶️ Run Project

## Install Dependencies

```bash
pip install -r requirements.txt
