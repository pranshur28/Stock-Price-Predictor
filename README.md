# Stock Price Predictor

A machine learning-based stock price prediction tool that can analyze and forecast prices for any stock or ETF.

## Features

- **Flexible Ticker Support**: Predict prices for any valid stock or ETF ticker symbol
- **Multiple Regression Models**: Implements and compares several regression approaches:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Support Vector Regression (SVR)
  - Random Forest Regression
- **Technical Indicators**: Automatically calculates relevant technical indicators:
  - Moving Averages (5, 20, 50 days)
  - Price Changes and Momentum
  - Volatility Metrics
  - Volume Analysis
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- **Model Evaluation**: Comprehensive metrics for model performance:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- **Visualizations**: Generates insightful charts:
  - Feature Importance Analysis
  - Model Comparison
  - Actual vs. Predicted Prices
  - Future Price Predictions

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - yfinance

You can install all required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### QQQ Price Predictor (Original)

To run the original QQQ price predictor:

```bash
python qqq_price_predictor.py
```

### Universal Stock Price Predictor

To predict prices for any stock or ETF:

```bash
python stock_price_predictor.py TICKER [options]
```

#### Arguments:

- `TICKER`: Stock ticker symbol (e.g., AAPL, MSFT, QQQ)

#### Options:

- `--start-date`: Start date for historical data (YYYY-MM-DD)
- `--end-date`: End date for historical data (YYYY-MM-DD)
- `--prediction-days`: Number of days to predict (default: 7)
- `--model`: Model to use for prediction (linear_regression, ridge, lasso, elastic_net, svr, random_forest, or best)

#### Examples:

Predict AAPL prices using the best model:
```bash
python stock_price_predictor.py AAPL
```

Predict MSFT prices for the next 14 days using Linear Regression:
```bash
python stock_price_predictor.py MSFT --prediction-days 14 --model linear_regression
```

Predict SPY prices with custom date range:
```bash
python stock_price_predictor.py SPY --start-date 2018-01-01 --end-date 2023-01-01
```

## Results

Results and visualizations are saved in the `results/[TICKER]` directory, including:
- Model comparison metrics
- Feature importance charts
- Prediction visualizations

## Limitations

- The predictions are based on historical data and technical indicators only
- Market sentiment, news events, and macroeconomic factors are not considered
- Future predictions become less reliable the further into the future they go
- Past performance is not indicative of future results

## Disclaimer

This tool is for educational purposes only. Financial markets are complex and unpredictable, and no prediction model can guarantee accurate results. Always consult with a financial advisor before making investment decisions.

