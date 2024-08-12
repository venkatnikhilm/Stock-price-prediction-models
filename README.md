# Comparative Analysis of Machine Learning Models for Stock Price Predictive Analytics

## Introduction

This proposal explores the complexities of predicting stock prices using machine learning (ML) models, given the non-linear and volatile nature of financial markets. The study aims to analyze and compare the performance of various ML models, including Artificial Neural Networks (ANN), Long Short-Term Memory (LSTM), Random Forest (RF), ARIMA, and Ensemble Support Vector Machine (ESVM), to determine the most accurate and robust models for stock price prediction.

## Methodology

- **Data Acquisition**: Using the yfinance library to collect 20 years of historical data for Amazon and Google stocks.
- **Implementation**: Models are implemented using PyTorch, Keras, and Scikit-Learn.
- **Training and Testing**: A rolling window approach is applied to mimic real-world trading conditions, with strategies to prevent overfitting.
- **Predictive Models**: Detailed descriptions of MLP, LSTM, RF, ARIMA, and SVR, focusing on their strengths and weaknesses in stock price prediction.

## Comparative Analysis

Models are evaluated based on accuracy metrics like RMSE, MAE, MAPE, and R-Squared. The analysis focuses on each model's performance, computational efficiency, and robustness.

## Results

- **RF and SVR**: Showed high accuracy in predicting Amazon and Google stock prices.
- **ARIMA**: Demonstrated lower accuracy compared to other models.

## Milestone Review and Team Contributions

All project milestones were successfully achieved. The team collaborated effectively, with each member contributing to different aspects of the project.

## Conclusion and Future Scope

This research underscores the importance of model selection in stock price predictive analytics. Future work could explore hybrid models, advanced feature engineering, and the ethical implications of AI in financial markets.
