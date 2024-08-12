import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Getting 20 years of Amazon stock data
data = yf.download('AMZN', start='2000-01-01', end='2023-10-29')

# Splitting the data into features and target variable
X = data['Close']
split_point = int(len(X) * 0.8)

train, test = X[:split_point], X[split_point:]

result = adfuller(train)
print("ADF:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Not Stationary")
else:
    print("Stationary")

# train_diff = train.diff().dropna()
# train_diff_2 = train_diff.diff().dropna()
# print(train_diff.head())
# train_diff_2.plot()

# plot_acf(train_diff, lags = 20)
# plt.show()

# plot_pacf(train_diff, lags=20)
# plt.show()



model = ARIMA(train, order=(1, 2, 1))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

print("RMSE:", rmse)
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label="actual")
plt.plot(test.index, predictions, label="predicted", color="red")
plt.legend()
plt.show()