import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preperation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preperation', 'simple_trend'))) 

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, confusion_matrix
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from simple_trend import TargetCreatorSimple
from dataset_initiation import prepare_dataset

class EnsembleRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'xgboost': XGBRegressor(random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state, n_estimators=100),
            'catboost': CatBoostRegressor(random_state=random_state, verbose=0),
            'linear': LinearRegression()
        }
        self.fitted = False

    def fit(self, X_train, y_train):
        """Train all models in the ensemble"""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, X_test):
        """Make predictions by averaging model outputs"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = np.zeros((X_test.shape[0], len(self.models)))
        for idx, (name, model) in enumerate(self.models.items()):
            pred = model.predict(X_test)
            predictions[:, idx] = pred.ravel()
        
        # Average predictions
        return np.mean(predictions, axis=1)

    def evaluate(self, X_test, y_test_close, y_test_volume, df_test, save_path='models/ensemble_regression/'):
        """Evaluate the ensemble model for close and volume, and apply targeting"""
        # Predict close and volume
        y_pred_close = self.predict(X_test)
        y_pred_volume = self.predict(X_test)  # Assuming same features for simplicity
        
        # Calculate MSE for close and volume
        mse_close = mean_squared_error(y_test_close, y_pred_close)
        mse_volume = mean_squared_error(y_test_volume, y_pred_volume)
        
        # Create DataFrame with predicted close prices
        pred_df = df_test.copy()
        pred_df['close'] = y_pred_close
        
        # Apply TargetCreatorSimple to predicted close prices
        creator = TargetCreatorSimple(
            df=pred_df,
            price_type='close',
            date_col='datetime',
            pct_threshold=0.005,
            min_length=3,
            smoothing_factor=0.001
        )
        creator.detect_trends(flactuation_length=5, flactuation_threshold=0.01)
        creator.add_trend_columns(direction=['Upward', 'Downward'])
        pred_df = creator.get_df(reset_index=True)
        
        # Get actual combined_target from original test data
        actual_target = df_test['Upward']
        # # Get actual combined_target from original test data
        # actual_target = df_test['combined_target']
        
        # # Evaluate classification metrics
        # y_pred_target = pred_df['combined_target']
        # Evaluate classification metrics
        y_pred_target = pred_df['Upward']
        accuracy = accuracy_score(actual_target, y_pred_target)
        class_report = classification_report(actual_target, y_pred_target)
        conf_matrix = confusion_matrix(actual_target, y_pred_target)
        
        # Print results
        print("Ensemble Model Evaluation:")
        print(f"Close Price MSE: {mse_close:.4f}")
        print(f"Volume MSE: {mse_volume:.4f}")
        print(f"Combined Target Accuracy: {accuracy:.4f}")
        print("\nClassification Report for Combined Target:")
        print(class_report)
        print("\nConfusion Matrix for Combined Target:")
        print(conf_matrix)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Combined Target)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot actual vs predicted close prices
        plt.subplot(1, 2, 2)
        plt.plot(df_test['datetime'], y_test_close, color='blue', label='Actual Close', alpha=0.5)
        plt.plot(df_test['datetime'], y_pred_close, color='red', label='Predicted Close', alpha=0.5)
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/ensemble_results.png')
        print(f"Results plot saved to f'{save_path}/ensemble_results.png'")

# Main execution code
if __name__ == "__main__":
    result_folder =os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Ensembles', 'ensemble_regression')
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'sol_binance.csv'))

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime')


    # Detect trends using simple_trend
    df = prepare_dataset(
        df=df,
        aggregate=True,
        group_length=1,
        visualize_trends=False
    )
    
    # Prepare features and targets
    X = df.drop(columns=['datetime', 'close', 'volume', 'Upward', 'Downward', 'trend_target', 'trend_label', 'combined_target', 'date'])
    y_close = df['close']
    y_volume = df['volume']
    
    # Handle missing values
    X = X.fillna(X.mean())
    X = X.astype(float)
    
    # Train-test split
    X_train, X_test, y_close_train, y_close_test, y_volume_train, y_volume_test, df_train, df_test = train_test_split(
        X, y_close, y_volume, df, test_size=0.2, shuffle=False, random_state=42
    )

    # Initialize and train ensemble for close
    ensemble_close = EnsembleRegressor(random_state=42)
    ensemble_close.fit(X_train, y_close_train)
    
    # Initialize and train ensemble for volume
    ensemble_volume = EnsembleRegressor(random_state=42)
    ensemble_volume.fit(X_train, y_volume_train)

    # Create directory if not exists
    os.makedirs(result_folder, exist_ok=True)

    path_prefix='models/ensemble_regression/'
    # Create directory if it doesn't exist
    os.makedirs(path_prefix, exist_ok=True)

    # Save models
    joblib.dump(ensemble_close, os.path.join(result_folder, f'{path_prefix}ensemble_close_model.pkl'))
    joblib.dump(ensemble_volume, os.path.join(result_folder, f'{path_prefix}ensemble_volume_model.pkl'))
    print(f"Models saved to {path_prefix}")
    
    # Evaluate model
    ensemble_close.evaluate(X_test, y_close_test, y_volume_test, df_test)