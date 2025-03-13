"""
QQQ Price Predictor

This script builds machine learning models to predict QQQ ETF prices using historical data.
It implements multiple regression approaches for prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

# Set random seeds for reproducibility
np.random.seed(42)

class QQQPricePredictor:
    def __init__(self, start_date=None, end_date=None, prediction_days=30):
        """
        Initialize the QQQ Price Predictor
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for historical data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for historical data in 'YYYY-MM-DD' format
        prediction_days : int, optional
            Number of days to use for feature creation (lookback period)
        """
        self.ticker = 'QQQ'
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            # Default to 5 years of historical data
            start_date_dt = datetime.now() - timedelta(days=5*365)
            self.start_date = start_date_dt.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.prediction_days = prediction_days
        self.data = None
        self.models = {}
        self.scalers = {}
        
        # Create directory for models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def fetch_data(self):
        """Fetch QQQ historical data from Yahoo Finance"""
        print(f"Fetching QQQ data from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if self.data.empty or len(self.data) < 100:
            raise ValueError("Not enough data fetched. Check your date range and internet connection.")
            
        # Handle multi-level columns if present
        if isinstance(self.data.columns, pd.MultiIndex):
            print("Multi-level columns detected. Flattening columns...")
            # Flatten multi-level columns
            self.data.columns = [col[0] if col[0] != '' else col[1] for col in self.data.columns]
            
        print(f"Successfully fetched {len(self.data)} days of QQQ data")
        print(f"Columns in the data: {self.data.columns.tolist()}")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data and create features"""
        if self.data is None:
            self.fetch_data()
            
        # Create a copy of the data
        df = self.data.copy()
        
        # Feature Engineering
        # Add technical indicators
        # 1. Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # 2. Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # 3. Volatility (standard deviation over a period)
        df['Volatility_5d'] = df['Close'].rolling(window=5).std()
        df['Volatility_20d'] = df['Close'].rolling(window=20).std()
        
        # 4. Trading volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # 5. Price momentum
        df['Momentum_5d'] = df['Close'] - df['Close'].shift(5)
        
        # 6. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 7. Moving Average Convergence Divergence (MACD)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Drop NaN values that result from the rolling calculations
        df.dropna(inplace=True)
        
        # Store the processed data
        self.processed_data = df
        
        print(f"Data preprocessing complete. Shape after preprocessing: {df.shape}")
        return df
    
    def prepare_traditional_ml_data(self, test_size=0.2):
        """Prepare data for traditional ML models like Random Forest"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
            
        df = self.processed_data.copy()
        
        # Features and target
        # Check if 'Adj Close' exists in the columns
        columns_to_drop = ['Close']
        if 'Adj Close' in df.columns:
            columns_to_drop.append('Adj Close')
            
        features = df.drop(columns_to_drop, axis=1)
        target = df['Close']
        
        # Scale the features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers['traditional'] = scaler
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=test_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_regression_models(self):
        """Train multiple regression models"""
        print("Training regression models...")
        X_train, X_test, y_train, y_test = self.prepare_traditional_ml_data()
        
        # Define regression models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name.replace('_', ' ').title()} Model Evaluation:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Save the model
            with open(f'models/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                
            # Store results
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred
            }
        
        # Plot feature importance for Random Forest
        if 'random_forest' in self.models:
            self._plot_feature_importance()
            
        # Compare models
        self._compare_models(results, y_test)
        
        return results
    
    def _plot_feature_importance(self):
        """Plot feature importance for Random Forest model"""
        if 'random_forest' not in self.models:
            print("Random Forest model not trained yet.")
            return
            
        # Get feature names
        columns_to_drop = ['Close']
        if 'Adj Close' in self.processed_data.columns:
            columns_to_drop.append('Adj Close')
            
        feature_names = self.processed_data.drop(columns_to_drop, axis=1).columns.tolist()
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.models['random_forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
    
    def _compare_models(self, results, y_test):
        """Compare different regression models"""
        # Create a comparison DataFrame
        metrics = {
            'Model': [],
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'R²': []
        }
        
        for name, result in results.items():
            metrics['Model'].append(name.replace('_', ' ').title())
            metrics['MSE'].append(result['mse'])
            metrics['RMSE'].append(result['rmse'])
            metrics['MAE'].append(result['mae'])
            metrics['R²'].append(result['r2'])
        
        comparison_df = pd.DataFrame(metrics)
        print("\nModel Comparison:")
        print(comparison_df)
        
        # Save the comparison
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Plot the comparison
        plt.figure(figsize=(14, 10))
        
        # Plot MSE
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='MSE', data=comparison_df)
        plt.title('Mean Squared Error (Lower is Better)')
        plt.xticks(rotation=45)
        
        # Plot RMSE
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='RMSE', data=comparison_df)
        plt.title('Root Mean Squared Error (Lower is Better)')
        plt.xticks(rotation=45)
        
        # Plot MAE
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='MAE', data=comparison_df)
        plt.title('Mean Absolute Error (Lower is Better)')
        plt.xticks(rotation=45)
        
        # Plot R²
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='R²', data=comparison_df)
        plt.title('R² Score (Higher is Better)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png')
        plt.close()
        
        # Plot predictions for all models
        test_dates = self.processed_data.index[-len(y_test):]
        
        plt.figure(figsize=(16, 8))
        plt.plot(test_dates, y_test, label='Actual Prices', linewidth=2)
        
        for name, result in results.items():
            plt.plot(test_dates, result['y_pred'], label=f'{name.replace("_", " ").title()} Predictions', alpha=0.7)
        
        plt.title('QQQ Price Prediction: Model Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/all_models_prediction.png')
        plt.close()
        
        return comparison_df
    
    def predict_next_days(self, days=7, model_name='random_forest'):
        """Predict prices for the next specified days using the selected model"""
        if model_name not in self.models:
            raise ValueError(f"{model_name} model not trained yet. Please train the model first.")
            
        # Get the latest data point
        columns_to_drop = ['Close']
        if 'Adj Close' in self.processed_data.columns:
            columns_to_drop.append('Adj Close')
            
        latest_data = self.processed_data.iloc[-1:].drop(columns_to_drop, axis=1)
        
        # Scale the data
        scaled_data = self.scalers['traditional'].transform(latest_data)
        
        # Make prediction for the next day
        next_day_prediction = self.models[model_name].predict(scaled_data)[0]
        
        # For multiple days ahead, we would need to simulate future features
        # This is a simplified approach
        predictions = [next_day_prediction]
        
        for _ in range(1, days):
            # This is a naive approach - in a real scenario, you'd need to update all features
            # For simplicity, we're just adding a small random change
            last_pred = predictions[-1]
            next_pred = last_pred * (1 + np.random.normal(0, 0.01))
            predictions.append(next_pred)
            
        return predictions
    
    def plot_predictions(self, actual, predictions, title='Price Predictions'):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual Prices')
        plt.plot(range(len(actual), len(actual) + len(predictions)), predictions, label='Predicted Prices', linestyle='--')
        plt.title(title)
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
        plt.close()

def main():
    """Main function to run the QQQ price predictor"""
    print("QQQ Price Predictor")
    print("=================")
    
    # Initialize the predictor
    predictor = QQQPricePredictor()
    
    # Fetch and preprocess data
    predictor.fetch_data()
    predictor.preprocess_data()
    
    # Train regression models
    results = predictor.train_regression_models()
    
    # Find the best model based on R²
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\nBest model based on R² score: {best_model.replace('_', ' ').title()}")
    
    # Make predictions for the next 7 days using the best model
    next_days = 7
    predictions = predictor.predict_next_days(days=next_days, model_name=best_model)
    
    # Get the last 30 actual prices for plotting
    last_prices = predictor.processed_data['Close'].values[-30:]
    
    # Plot predictions
    predictor.plot_predictions(last_prices, predictions, f'{best_model.replace("_", " ").title()} Predictions')
    
    # Print predictions
    print(f"\nPredictions for the next {next_days} days using {best_model.replace('_', ' ').title()}:")
    for i, pred in enumerate(predictions):
        print(f"Day {i+1}: ${pred:.2f}")
    
    print("\nAnalysis complete! Check the 'results' folder for visualizations.")

if __name__ == "__main__":
    main()
