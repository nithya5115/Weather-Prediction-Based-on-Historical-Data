import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv('historical_weather.csv') 
data['Date'] = pd.to_datetime(data['Date'])
data['Timestamp'] = data['Date'].astype(int) / 10**9 
X = data['Timestamp'].values.reshape(-1, 1)  
y = data['Temperature'].values  # Target variable (temperature)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
future_dates = pd.date_range(start=data['Date'].max(), periods=10, freq='D')
future_timestamps = future_dates.astype(int) / 10**9
future_predictions = model.predict(future_timestamps.values.reshape(-1, 1))
plt.scatter(data['Date'], data['Temperature'], label='Historical Data', color='blue')
plt.plot(future_dates, future_predictions, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
for date, temp in zip(future_dates, future_predictions):
    print(f'{date.date()}: {temp:.2f}°C')