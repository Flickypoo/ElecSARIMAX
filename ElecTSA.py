# Electricity Consumption Forecasting Using Optimized SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings('ignore')


# Load the dataset
try:
    df = pd.read_csv('household_power_consumption.txt', 
                     sep=';', 
                     parse_dates={'Datetime': ['Date', 'Time']}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['nan','?'])
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: 'household_power_consumption.txt' not found in the working directory.")
    exit()

print("\nFirst 5 rows of the dataset:")
print(df.head())

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)

df.ffill(inplace=True)

df_hourly = df.resample('H').mean()

print("\nData Resampled to Hourly Frequency:")
print(df_hourly.head())

# EDA

plt.figure(figsize=(15,5))
plt.plot(df_hourly['Global_active_power'], color='blue')
plt.title('Global Active Power Over Time')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.show()

# Decompose the time series to observe trend, seasonality, and residuals
decomposition = seasonal_decompose(df_hourly['Global_active_power'], model='additive', period=24*30)  # Approx monthly seasonality

fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.show()


# Create time-based features
df_hourly['Hour'] = df_hourly.index.hour
df_hourly['DayOfWeek'] = df_hourly.index.dayofweek
df_hourly['Month'] = df_hourly.index.month

# Lag features: Previous 24 hours
for lag in range(1,25):
    df_hourly[f'lag_{lag}'] = df_hourly['Global_active_power'].shift(lag)

# Drop rows with NaN values resulting from lag features
df_hourly.dropna(inplace=True)

print("\nDataFrame with Lag Features:")
print(df_hourly.head())

# Stationarity Check and Differencing

def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '#Lags Used', '#Observations Used']
    for value, label in zip(result[:4], labels):
        print(f'{label}: {value}')
    if result[1] <= 0.05:
        print("=> The series is stationary.\n")
    else:
        print("=> The series is non-stationary.\n")

adf_test(df_hourly['Global_active_power'], 'Original Series')

y = df_hourly['Global_active_power']

X = df_hourly[['Hour', 'DayOfWeek', 'Month'] + [f'lag_{lag}' for lag in range(1,25)]]

# Spplit the data into training and testing sets (last 7 days as test)
train_size = int(len(df_hourly) - 24*7)  # 7 days of hourly data
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# Time Series Modeling and Forecasting using SARIMAX


order = (2, 1, 1)  # (p,d,q)
seasonal_order = (1, 0, 1, 24)  # (P,D,Q,s)

print("\nFitting SARIMAX model...")
model = SARIMAX(y_train, 
                exog=X_train, 
                order=order, 
                seasonal_order=seasonal_order, 
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model with progress indication
model_fit = model.fit(disp=False)
print("SARIMAX Model Fitted Successfully.")

print("\nSARIMAX Model Summary:")
print(model_fit.summary())

# Forecasting

print("\nForecasting...")
forecast = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
print("Forecasting Completed.")

# Model Evaluation

mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

#  Visualization of Results

plt.figure(figsize=(15,5))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Actual vs Forecasted Global Active Power')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

residuals = y_test - forecast
plt.figure(figsize=(15,5))
plt.plot(residuals, label='Residuals', color='purple')
plt.title('Residuals of SARIMAX Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(residuals, kde=True, bins=30, color='green')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

