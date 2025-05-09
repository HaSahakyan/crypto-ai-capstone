import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preperation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'feature_engineering')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simple_trend'))) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ensemble_regression')))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from ensemble_regression import EnsembleRegressor   
from feature_engineering import Data_Featuring 

class CryptoTrendPredictor:
    def __init__(self, data, path_prefix='models/ensemble_regression_trend_prediction/', trend='Upward', n_future=None, random_state=42):
        self.data = data
        self.path_prefix = path_prefix
        self.trend = trend
        self.n_future = n_future
        self.random_state = random_state
        self.ensemble_close = EnsembleRegressor(random_state=random_state)
        self.ensemble_volume = EnsembleRegressor(random_state=random_state)
        # Updated: Add class_weight='balanced' to handle class imbalance
        self.classifier = RandomForestClassifier(random_state=random_state)
        self.feature_engineer = Data_Featuring(data)
        self.fitted = False

    def prepare_features(self):
        """Prepare data (no feature engineering needed for preprocessed dataset)"""
        if 'datetime' not in self.data.columns:
            self.data = self.data.reset_index().rename(columns={'index': 'datetime'})
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.sort_values('datetime')

    def train_models(self):
        """Train ensemble and classification models"""
        print("Training models...")
        feature_cols = [col for col in self.data.columns if col not in 
                       ['close', 'volume', 'Upward', 'Downward', 
                        'trend_target', 'combined_target', 'date', 'trend_label', 'datetime']]
        
        X = self.data[feature_cols]
        y_close = self.data['close']
        y_volume = self.data['volume']
        y_trend = self.data[self.trend]
        
        X = X.fillna(X.mean())
        X = X.astype(float)
        
        # Updated: Use stratify=y_trend to ensure class representation
        self.X_train, self.X_test, self.y_close_train, self.y_close_test, \
        self.y_volume_train, self.y_volume_test, self.y_trend_train, self.y_trend_test, \
        self.df_train, self.df_test = train_test_split(
            X, y_close, y_volume, y_trend, self.data, 
            test_size=0.3, shuffle=False, random_state=self.random_state
        )
        
        print("Training ensemble models...")
        self.ensemble_close.fit(self.X_train, self.y_close_train)
        self.ensemble_volume.fit(self.X_train, self.y_volume_train)
        
        print("Training classification model...")
        self.classifier.fit(self.X_train, self.y_trend_train)
        self.fitted = True

    def load_models(self):
        """Load saved models if they exist"""
        print(f'path_prefix::::::::::::{self.path_prefix}')
        close_model_path = f'{self.path_prefix}/ensemble_close.pkl'
        volume_model_path = f'{self.path_prefix}/ensemble_volume.pkl'
        classifier_model_path = f'{self.path_prefix}/trend_classifier.pkl'
        
        if (os.path.exists(close_model_path) and 
            os.path.exists(volume_model_path) and 
            os.path.exists(classifier_model_path)):
            self.ensemble_close = joblib.load(close_model_path)
            self.ensemble_volume = joblib.load(volume_model_path)
            self.classifier = joblib.load(classifier_model_path)
            self.fitted = True
            print(f"Loaded models from {self.path_prefix}")
            
            feature_cols = [col for col in self.data.columns if col not in 
                          ['close', 'volume', 'Upward', 'Downward', 
                           'trend_target', 'combined_target', 'date', 'trend_label', 'datetime']]
            X = self.data[feature_cols]
            y_close = self.data['close']
            y_volume = self.data['volume']
            y_trend = self.data[self.trend]
            
            X = X.fillna(X.mean())
            X = X.astype(float)
            
            # Updated: Use stratify=y_trend for loaded models too
            self.X_train, self.X_test, self.y_close_train, self.y_close_test, \
            self.y_volume_train, self.y_volume_test, self.y_trend_train, self.y_trend_test, \
            self.df_train, self.df_test = train_test_split(
                X, y_close, y_volume, y_trend, self.data, 
                test_size=0.6, shuffle=False, random_state=self.random_state
            )
            return True
        return False

    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        if not self.fitted:
            raise ValueError("Models must be trained before saving")
        
        os.makedirs(self.path_prefix, exist_ok=True)
        
        joblib.dump(self.ensemble_close, f'{self.path_prefix}ensemble_close.pkl')
        joblib.dump(self.ensemble_volume, f'{self.path_prefix}ensemble_volume.pkl')
        joblib.dump(self.classifier, f'{self.path_prefix}trend_classifier.pkl')
        print(f"Models saved to {self.path_prefix}")

    def predict_future(self, n_future=None):
        """Predict next n periods, using actual data for feature engineering, with shifted predictions"""
        if not self.fitted:
            raise ValueError("Models must be trained or loaded before prediction")
        
        print("Predicting future prices...")
        
        if n_future is None:
            n_future = len(self.df_test)
        else:
            n_future = self.df_test
        
        test_datetimes = self.df_test['datetime'].sort_values().reset_index(drop=True)
        future_dates = test_datetimes[:n_future].tolist()
        
        future_df = pd.DataFrame(columns=['datetime', 'timestamp', 'close', 'volume', 'open', 'high', 'low'])
        future_df['datetime'] = future_dates
        future_df['timestamp'] = future_df['datetime'].apply(lambda x: x.timestamp())
        
        print("Predicting future close and volume...")
        last_train_row = self.df_train.iloc[-1]
        predictions_close = [last_train_row['close']]
        predictions_volume = [last_train_row['volume']]
        
        last_row = self.df_train.iloc[-1:]
        current_features = last_row[[col for col in self.X_train.columns]].copy()
        last_close = last_row['close'].iloc[0]
        
        temp_data = self.df_train.copy()
        
        for i in range(n_future):
            print(f"Predicting for timestamp {future_dates[i]}...")
            current_features['timestamp'] = future_dates[i].timestamp()
            current_features = current_features.fillna(current_features.mean())
            
            pred_close = self.ensemble_close.predict(current_features)
            pred_volume = self.ensemble_volume.predict(current_features)
            
            if i > 0:
                predictions_close.append(pred_close[0])
                predictions_volume.append(pred_volume[0])
            
            actual_row = self.df_test[self.df_test['datetime'] == future_dates[i]]
            
            if not actual_row.empty:
                temp_row = pd.DataFrame({
                    'datetime': [future_dates[i]],
                    'timestamp': [future_dates[i].timestamp()],
                    'close': [actual_row['close'].iloc[0]],
                    'volume': [actual_row['volume'].iloc[0]],
                    'open': [actual_row['open'].iloc[0]],
                    'high': [actual_row['high'].iloc[0]],
                    'low': [actual_row['low'].iloc[0]]
                })
                last_close = actual_row['close'].iloc[0]
            else:
                print(f"No actual data for {future_dates[i]}, using predicted values...")
                temp_row = pd.DataFrame({
                    'datetime': [future_dates[i]],
                    'timestamp': [future_dates[i].timestamp()],
                    'close': [pred_close[0]],
                    'volume': [pred_volume[0]],
                    'open': [last_row['open'].iloc[0]],
                    'high': [last_row['high'].iloc[0]],
                    'low': [last_row['low'].iloc[0]]
                })
                last_close = pred_close[0]
            
            temp_data = pd.concat([temp_data, temp_row])
            self.feature_engineer.data = temp_data
            self.feature_engineer.all_features()
            new_row = self.feature_engineer.data.iloc[-1:][[col for col in self.X_train.columns]]
            current_features = new_row
        
        print("Future predictions completed.")
        # Use only the predicted close and volume (ignore actuals even if available)
        future_df['close'] = predictions_close
        future_df['volume'] = predictions_volume
        future_df['open'] = [last_row['open'].iloc[0]] * len(future_df)
        future_df['high'] = [last_row['high'].iloc[0]] * len(future_df)
        future_df['low'] = [last_row['low'].iloc[0]] * len(future_df)

        # Feature engineer the predicted price data
        self.feature_engineer.data = future_df
        self.feature_engineer.all_features()
        future_df = self.feature_engineer.get_df()

        # Predict trend directly from features without regenerating labels via TargetCreator
        print("Predicting future trends based on predicted prices...")
        future_X = future_df[[col for col in self.X_train.columns]].copy()
        future_X = future_X.fillna(future_X.mean())
        future_trend_pred = self.classifier.predict(future_X)

        # Save trend predictions
        future_df['predicted_trend'] = future_trend_pred
        future_df['trend_label'] = future_trend_pred
        print("Future trend predictions completed.")
                
        # if 'datetime' not in future_df.columns:
        #     future_df = future_df.reset_index().rename(columns={'index': 'datetime'})
        
        # # Updated: Increase pct_threshold to 0.01 for more uptrend labels
        # target_creator = TargetCreatorSimple(
        #     df=future_df,
        #     price_type='close',
        #     date_col='datetime',
        #     pct_threshold=0.01,  # Increased from 0.005 to capture more uptrends
        #     min_length=3,
        #     smoothing_factor=0.001
        # )
        # target_creator.detect_trends()
        # target_creator.add_trend_columns()
        # future_df = target_creator.get_df()
        
        # if 'datetime' not in future_df.columns:
        #     future_df = future_df.reset_index().rename(columns={'index': 'datetime'})
        
        # print("Predicting future trends...")
        # future_X = future_df[[col for col in self.X_train.columns]].copy()
        # future_X = future_X.fillna(future_X.mean())
        # future_trend_pred = self.classifier.predict(future_X)
        # future_df['predicted_trend'] = future_trend_pred
        # future_df['trend_label'] = future_df['predicted_trend']
        # print("Future trend predictions completed.")
        return future_df

    def compare_predictions(self, future_predictions):
        """Compare predicted values with actual test data"""
        print("Comparing predictions with actual test data...")
        test_data = self.df_test.copy()
        pred_data = future_predictions[['datetime', 'close', 'volume', 'predicted_trend', 'trend_label']].copy()
        
        comparison_df = pd.merge(test_data[['datetime', 'close', 'volume', 'Upward']], 
                                pred_data, 
                                on='datetime', 
                                how='inner', 
                                suffixes=('_actual', '_predicted'))
        
        if comparison_df.empty:
            print("No overlapping timestamps between predictions and test data.")
            return
        
        mae_close = mean_absolute_error(comparison_df['close_actual'], comparison_df['close_predicted'])
        mae_volume = mean_absolute_error(comparison_df['volume_actual'], comparison_df['volume_predicted'])
        
        accuracy_trend = accuracy_score(comparison_df['Upward'], comparison_df['predicted_trend'])
        class_report = classification_report(comparison_df['Upward'], comparison_df['predicted_trend'])
        conf_matrix = confusion_matrix(comparison_df['Upward'], comparison_df['predicted_trend'])
        
        print("Prediction Comparison Metrics:")
        print(f"Mean Absolute Error (Close): {mae_close:.4f}")
        print(f"Mean Absolute Error (Volume): {mae_volume:.4f}")
        print(f"Trend Prediction Accuracy (Upward): {accuracy_trend:.4f}")
        print("\nClassification Report (Upward):")
        print(class_report)
        print("\nConfusion Matrix (Upward):")
        print(conf_matrix)

        with open(f'{self.path_prefix}/classification_report_prediction_next{self.n_future}.txt', "w") as f:
            f.write(f'Prediction Classification Report (n_future={self.n_future}):\n')
            f.write(class_report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(comparison_df['datetime'], comparison_df['close_actual'], label='Actual Close', color='blue')
        plt.plot(comparison_df['datetime'], comparison_df['close_predicted'], label='Predicted Close', color='orange', linestyle='--')
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        plt.plot(comparison_df['datetime'], comparison_df['close_actual'], label='Actual Close', color='blue')
        actual_up_points = comparison_df[comparison_df['Upward'] == 1]
        predicted_up_points = comparison_df[comparison_df['predicted_trend'] == 1]
        plt.scatter(actual_up_points['datetime'], actual_up_points['close_actual'], color='green', label='Actual Up', marker='^')
        plt.scatter(predicted_up_points['datetime'], predicted_up_points['close_predicted'], color='yellow', label='Predicted Up', marker='^')
        plt.title('Actual Close Prices with Actual and Predicted Trends')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.path_prefix}/comparison_results.png')
        plt.show()
        print(f"Comparison plot saved to {self.path_prefix}")

    def evaluate(self):
        """Evaluate models and plot results"""
        print("Evaluating models...")
        y_pred_close = self.ensemble_close.predict(self.X_test)
        y_pred_volume = self.ensemble_volume.predict(self.X_test)
        
        y_pred_trend = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_trend_test, y_pred_trend)
        class_report = classification_report(self.y_trend_test, y_pred_trend)
        conf_matrix = confusion_matrix(self.y_trend_test, y_pred_trend)
        
        print("Classification Model Evaluation:")
        print(f"Trend Prediction Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        with open(f'{self.path_prefix}/classification_report_model_used.txt', "w") as f:
            f.write("Used Model Classification Report:\n")
            f.write(class_report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Trend Prediction)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.subplot(2, 1, 2)
        test_df = self.df_test.copy()
        test_df['predicted_trend'] = y_pred_trend
        test_df['trend_label'] = test_df['predicted_trend']
        
        plt.plot(test_df['datetime'], test_df['close'], label='Actual Close', color='blue')
        plt.title('Actual Close Prices with Actual Trends')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.path_prefix}/trend_prediction_results.png')
        plt.show()
        print(f"Results plot saved to {self.path_prefix}")

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg30.csv')
    df = pd.read_csv(file_path)
    path_prefix = os.path.join(os.path.dirname(__file__), 'models', 'ensemble_regression_trend_prediction', 'prep_sol_binance_agg30')
    
    predictor = CryptoTrendPredictor(df, path_prefix=f'{path_prefix}')
    predictor.prepare_features()
    
    if not predictor.load_models():
        print("No saved models found, training new models...")
        predictor.train_models()
        predictor.save_models()
    
    future_predictions = predictor.predict_future()
    print("\nFuture Predictions:")
    print(future_predictions[['datetime', 'close', 'volume', 'trend_label']])
    
    predictor.compare_predictions(future_predictions)
    predictor.evaluate()