import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import statsmodels.api as sm

# Getting 20 years of Amazon stock data
data = yf.download('AMZN', start='2020-01-01', end='2023-10-29')

# Feature engineering for classification
data['Price_Up'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Dropping NaN values
data.dropna(inplace=True)

# Splitting the data into features and target variable
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Price_Up']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)

# Creating the RandomForestClassifier model
clf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=15, min_samples_leaf=4, max_features="sqrt")

# Fitting the model
clf.fit(X_train, y_train)

# Making predictions
predictions = clf.predict(X_test)

# Calculating the accuracy
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Accuracy: {accuracy}")
print("RMSE:", rmse)