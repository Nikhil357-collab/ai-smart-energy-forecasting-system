import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

data = pd.read_csv(
    "AI_Powerd_Energy/data/archive (6)/energy.csv",
    parse_dates=['Datetime']
)

# Set datetime index
data.set_index("Datetime", inplace=True)

# Rename energy column automatically
data.rename(
    columns={data.columns[0]: 'Energy'},
    inplace=True
)

# Resample hourly
data = data.resample('h').mean()

# Fill missing values
data = data.ffill()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

# Time Features
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek
data['month'] = data.index.month
data['weekend'] = data['day'].apply(
    lambda x: 1 if x >= 5 else 0
)

# Lag Features
data['lag_1'] = data['Energy'].shift(1)
data['lag_24'] = data['Energy'].shift(24)

# Rolling Mean
data['rolling_mean_24'] = data['Energy'].rolling(24).mean()

# Remove NaN rows
data.dropna(inplace=True)

# --------------------------------------------------
# FEATURES & TARGET

X = data[
    [
        'hour',
        'day',
        'month',
        'weekend',
        'lag_1',
        'lag_24',
        'rolling_mean_24'
    ]
]

y = data['Energy']

# --------------------------------------------------
# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# MODEL

model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=1000,
    random_state=42
)

# --------------------------------------------------
# TRAIN
# 

model.fit(X_train, y_train)

# --------------------------------------------------
# PREDICT

predictions = model.predict(X_test)

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

# --------------------------------------------------
# PRINT RESULTS

print("\n========= MODEL PERFORMANCE =========")

print(f"MAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2 Score : {r2:.4f}")

# --------------------------------------------------
# SAVE MODEL

joblib.dump(
    model,
    "AI_Powerd_Energy/models/energy_forecast_model.pkl"
)

print("\nModel saved successfully!")

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(15,6))

plt.plot(
    y_test.values[:500],
    label='Actual'
)

plt.plot(
    predictions[:500],
    label='Predicted'
)

plt.title("Actual vs Predicted Energy")

plt.xlabel("Samples")

plt.ylabel("Energy Consumption")

plt.legend()

plt.grid(True)

plt.show()