"""
Stock Price Predictor GUI

A graphical user interface for the stock price predictor application.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
import threading
import sys
import io
from PIL import Image, ImageTk

# Import the StockPricePredictor class
from stock_price_predictor import StockPricePredictor

class RedirectText:
    """Redirect print statements to the GUI"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO()

    def write(self, string):
        self.buffer.write(string)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, self.buffer.getvalue())
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass

class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set the icon
        # self.root.iconbitmap("icon.ico")  # Uncomment and add an icon file if available
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the input frame
        self.create_input_frame()
        
        # Create the output frame
        self.create_output_frame()
        
        # Create the visualization frame
        self.create_visualization_frame()
        
        # Initialize variables
        self.predictor = None
        self.results = None
        self.prediction_thread = None
        
        # Add a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_input_frame(self):
        """Create the input frame with form elements"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        # Create a grid layout
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=2)
        input_frame.columnconfigure(2, weight=1)
        input_frame.columnconfigure(3, weight=2)
        
        # Ticker symbol
        ttk.Label(input_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ticker_var = tk.StringVar()
        ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var)
        ticker_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5, padx=5)
        
        # Popular tickers dropdown
        ttk.Label(input_frame, text="Popular Tickers:").grid(row=0, column=2, sticky=tk.W, pady=5)
        popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "QQQ", "SPY", "DIA"]
        self.popular_ticker_var = tk.StringVar()
        popular_ticker_combo = ttk.Combobox(input_frame, textvariable=self.popular_ticker_var, values=popular_tickers)
        popular_ticker_combo.grid(row=0, column=3, sticky=tk.W+tk.E, pady=5, padx=5)
        popular_ticker_combo.bind("<<ComboboxSelected>>", self.on_popular_ticker_selected)
        
        # Date range
        ttk.Label(input_frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W, pady=5)
        default_start = datetime.now() - timedelta(days=5*365)
        self.start_date_entry = DateEntry(input_frame, width=12, background='darkblue',
                                         foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd',
                                         year=default_start.year, month=default_start.month, day=default_start.day)
        self.start_date_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(input_frame, text="End Date:").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.end_date_entry = DateEntry(input_frame, width=12, background='darkblue',
                                       foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=1, column=3, sticky=tk.W, pady=5, padx=5)
        
        # Prediction days
        ttk.Label(input_frame, text="Prediction Days:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.prediction_days_var = tk.IntVar(value=7)
        prediction_days_spin = ttk.Spinbox(input_frame, from_=1, to=30, textvariable=self.prediction_days_var, width=5)
        prediction_days_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Model selection
        ttk.Label(input_frame, text="Model:").grid(row=2, column=2, sticky=tk.W, pady=5)
        models = ["best", "linear_regression", "ridge", "lasso", "elastic_net", "svr", "random_forest"]
        self.model_var = tk.StringVar(value="best")
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, values=models)
        model_combo.grid(row=2, column=3, sticky=tk.W+tk.E, pady=5, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.start_prediction)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_prediction, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_all)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
    def create_output_frame(self):
        """Create the output frame with text area for logs"""
        output_frame = ttk.LabelFrame(self.main_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a text widget for output
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, width=80, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.output_text, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # Redirect stdout to the text widget
        self.stdout_redirect = RedirectText(self.output_text)
        
    def create_visualization_frame(self):
        """Create the visualization frame with tabs for different plots"""
        viz_frame = ttk.LabelFrame(self.main_frame, text="Visualizations", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.predictions_tab = ttk.Frame(self.notebook)
        self.model_comparison_tab = ttk.Frame(self.notebook)
        self.feature_importance_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.predictions_tab, text="Predictions")
        self.notebook.add(self.model_comparison_tab, text="Model Comparison")
        self.notebook.add(self.feature_importance_tab, text="Feature Importance")
        
        # Initialize the figure canvases
        self.predictions_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.predictions_canvas = FigureCanvasTkAgg(self.predictions_figure, self.predictions_tab)
        self.predictions_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.model_comparison_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.model_comparison_canvas = FigureCanvasTkAgg(self.model_comparison_figure, self.model_comparison_tab)
        self.model_comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.feature_importance_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.feature_importance_canvas = FigureCanvasTkAgg(self.feature_importance_figure, self.feature_importance_tab)
        self.feature_importance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_popular_ticker_selected(self, event):
        """Handle selection from the popular tickers dropdown"""
        self.ticker_var.set(self.popular_ticker_var.get())
        
    def start_prediction(self):
        """Start the prediction process in a separate thread"""
        # Validate inputs
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return
        
        start_date = self.start_date_entry.get_date().strftime("%Y-%m-%d")
        end_date = self.end_date_entry.get_date().strftime("%Y-%m-%d")
        prediction_days = self.prediction_days_var.get()
        model = self.model_var.get()
        
        # Update UI state
        self.predict_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set(f"Predicting {ticker} prices...")
        
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        # Redirect stdout
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_redirect
        
        # Start prediction in a separate thread
        self.prediction_thread = threading.Thread(
            target=self.run_prediction,
            args=(ticker, start_date, end_date, prediction_days, model)
        )
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
    def run_prediction(self, ticker, start_date, end_date, prediction_days, model):
        """Run the prediction process"""
        try:
            # Initialize the predictor
            self.predictor = StockPricePredictor(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            # Fetch and preprocess data
            self.predictor.fetch_data()
            self.predictor.preprocess_data()
            
            # Train regression models
            self.results = self.predictor.train_regression_models()
            
            # Determine which model to use for prediction
            if model == 'best':
                # Find the best model based on R²
                best_model = max(self.results.items(), key=lambda x: x[1]['r2'])[0]
                print(f"\nBest model based on R² score: {best_model.replace('_', ' ').title()}")
                model_to_use = best_model
            else:
                if model in self.results:
                    model_to_use = model
                    print(f"\nUsing specified model: {model_to_use.replace('_', ' ').title()}")
                else:
                    print(f"Specified model '{model}' not found. Using best model instead.")
                    best_model = max(self.results.items(), key=lambda x: x[1]['r2'])[0]
                    model_to_use = best_model
                    print(f"Best model based on R² score: {model_to_use.replace('_', ' ').title()}")
            
            # Make predictions for the next days
            predictions = self.predictor.predict_next_days(days=prediction_days, model_name=model_to_use)
            
            # Print predictions
            print(f"\nPredictions for the next {prediction_days} days using {model_to_use.replace('_', ' ').title()}:")
            for i, pred in enumerate(predictions):
                print(f"Day {i+1}: ${pred:.2f}")
            
            print(f"\nAnalysis complete! Check the 'results/{ticker}' folder for visualizations.")
            
            # Update visualizations
            self.root.after(100, lambda: self.update_visualizations(ticker, model_to_use, predictions))
            
        except Exception as e:
            print(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            # Restore stdout
            sys.stdout = self.old_stdout
            
            # Update UI state
            self.root.after(0, self.reset_ui_state)
    
    def update_visualizations(self, ticker, model_name, predictions):
        """Update the visualization tabs with the prediction results"""
        try:
            # Update predictions tab
            self.predictions_figure.clear()
            ax = self.predictions_figure.add_subplot(111)
            
            # Get the last 30 actual prices
            last_prices = self.predictor.processed_data['Close'].values[-30:]
            dates = list(range(len(last_prices)))
            future_dates = list(range(len(last_prices), len(last_prices) + len(predictions)))
            
            ax.plot(dates, last_prices, label='Historical Prices', color='blue')
            ax.plot(future_dates, predictions, label=f'Predicted Prices ({model_name.replace("_", " ").title()})', 
                   color='red', linestyle='--', marker='o')
            
            ax.set_title(f'{ticker} Price Prediction')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True)
            
            self.predictions_figure.tight_layout()
            self.predictions_canvas.draw()
            
            # Update model comparison tab
            if os.path.exists(f'results/{ticker}/model_comparison.csv'):
                comparison_df = pd.read_csv(f'results/{ticker}/model_comparison.csv')
                
                self.model_comparison_figure.clear()
                
                # Create a 2x2 grid for the metrics
                gs = self.model_comparison_figure.add_gridspec(2, 2)
                
                # MSE plot
                ax1 = self.model_comparison_figure.add_subplot(gs[0, 0])
                ax1.bar(comparison_df['Model'], comparison_df['MSE'], color='skyblue')
                ax1.set_title('Mean Squared Error (Lower is Better)')
                ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                
                # RMSE plot
                ax2 = self.model_comparison_figure.add_subplot(gs[0, 1])
                ax2.bar(comparison_df['Model'], comparison_df['RMSE'], color='lightgreen')
                ax2.set_title('Root Mean Squared Error (Lower is Better)')
                ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                
                # MAE plot
                ax3 = self.model_comparison_figure.add_subplot(gs[1, 0])
                ax3.bar(comparison_df['Model'], comparison_df['MAE'], color='salmon')
                ax3.set_title('Mean Absolute Error (Lower is Better)')
                ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                
                # R² plot
                ax4 = self.model_comparison_figure.add_subplot(gs[1, 1])
                ax4.bar(comparison_df['Model'], comparison_df['R²'], color='mediumpurple')
                ax4.set_title('R² Score (Higher is Better)')
                ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                
                self.model_comparison_figure.tight_layout()
                self.model_comparison_canvas.draw()
            
            # Update feature importance tab
            if os.path.exists(f'results/{ticker}/feature_importance.png'):
                self.feature_importance_figure.clear()
                img = plt.imread(f'results/{ticker}/feature_importance.png')
                ax = self.feature_importance_figure.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                self.feature_importance_canvas.draw()
            
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
    
    def stop_prediction(self):
        """Stop the prediction process"""
        if self.prediction_thread and self.prediction_thread.is_alive():
            # We can't directly stop a thread, but we can set a flag
            # that the thread can check periodically
            self.status_var.set("Stopping prediction...")
            
            # For now, we'll just disable the stop button and wait for the thread to complete
            self.stop_button.config(state=tk.DISABLED)
    
    def reset_ui_state(self):
        """Reset the UI state after prediction"""
        self.predict_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Ready")
    
    def clear_all(self):
        """Clear all inputs and outputs"""
        self.ticker_var.set("")
        self.popular_ticker_var.set("")
        
        # Reset dates
        default_start = datetime.now() - timedelta(days=5*365)
        self.start_date_entry._set_text(default_start.strftime("%Y-%m-%d"))
        self.end_date_entry._set_text(datetime.now().strftime("%Y-%m-%d"))
        
        self.prediction_days_var.set(7)
        self.model_var.set("best")
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        
        # Clear visualizations
        self.predictions_figure.clear()
        self.predictions_canvas.draw()
        
        self.model_comparison_figure.clear()
        self.model_comparison_canvas.draw()
        
        self.feature_importance_figure.clear()
        self.feature_importance_canvas.draw()
        
        # Reset status
        self.status_var.set("Ready")

def main():
    """Main function to run the Stock Price Predictor GUI"""
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
