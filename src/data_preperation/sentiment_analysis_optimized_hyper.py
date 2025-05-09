import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import os
from sklearn.base import RegressorMixin, ClassifierMixin
from xgboost import XGBRegressor, XGBClassifier
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_initiation')))
from dataset_initiation import prepare_dataset

# Define hyperparameter grids for tuning
param_grids = {
    'RandomForestRegressor': {
        'n_estimators': [50, 100],
        'max_depth': [10, None]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    'LinearRegression': {},  # No hyperparameters to tune
    'XGBRegressor': {
        'n_estimators': [50, 100],
        'max_depth': [3, 6]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [10, None]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0]
    },
    'XGBClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [3, 6]
    },
    'BalancedRandomForestClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [10, None]
    }
}

def prepare_data(data_path, aggregation_window, pred_trend='Upward', direction='up'):
    """Prepare dataset with aggregation, trend detection, and target creation."""
    window_mapping = {
        'daily': {'group_length': 1440, 'freq': 'D', 'join_key': 'date', 'time_format': lambda x: x.dt.date},
        'hourly': {'group_length': 60, 'freq': 'H', 'join_key': 'datetime', 'time_format': lambda x: x.dt.floor('H')},
        'minutely': {'group_length': 1, 'freq': 'T', 'join_key': 'datetime', 'time_format': lambda x: x.dt.floor('T')}
    }
    window_config = window_mapping.get(aggregation_window.lower())
    if window_config is None:
        raise ValueError("Invalid aggregation window. Choose from 'daily', 'hourly', 'minutely'.")

    # Load and preprocess data
    merged_data = pd.read_csv(data_path)
    merged_data['datetime'] = pd.to_datetime(merged_data['timestamp'], unit='s')
    merged_data = merged_data.sort_values('datetime')

    # Aggregate and detect trends
    merged_data = prepare_dataset(
        df=merged_data,
        aggregate=True,
        group_length=window_config['group_length'],
        visualize_trends=False
    )

    # Prepare targets
    merged_data['next_close'] = merged_data['close'].shift(-1)
    merged_data['next_trend'] = (merged_data[pred_trend].shift(-1) == direction).astype(int)
    trend = 'Upward' if direction == 'up' else 'Downward'
    merged_data['next_trend_flug'] = (merged_data[trend].shift(-1) == 1).astype(int)
    merged_data = merged_data[:-1]  # Drop last row with no target
    merged_data.dropna(inplace=True)

    # Split data
    numerical_features = merged_data.select_dtypes(include=['number']).columns
    features = [col for col in numerical_features if col not in ['next_close', 'next_trend_flug']]
    train_size = int(len(merged_data) * 0.8)
    train_data = merged_data.iloc[:train_size]
    test_data = merged_data.iloc[train_size:]

    return {
        'X_train': train_data[features],
        'y_train_price': train_data['next_close'],
        'y_train_trend': train_data['next_trend_flug'],
        'X_test': test_data[features],
        'y_test_price': test_data['next_close'],
        'y_test_trend': test_data['next_trend_flug'],
        'train_data': train_data,
        'test_data': test_data
    }

def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, task, resampled='no'):
    """Train model with hyperparameter tuning and compute evaluation metrics."""
    # Get hyperparameter grid
    param_grid = param_grids.get(model_name, {})
    
    # Use GridSearchCV if hyperparameters are defined, otherwise use the model directly
    if param_grid:
        scoring = 'neg_mean_squared_error' if task == 'price_prediction' else 'f1'
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"{model_name} (Resampled={resampled}) - Best hyperparameters: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    
    if task == 'price_prediction':
        mse = mean_squared_error(y_test, y_pred)
        print(f'{model_name} - MSE for price prediction: {mse}')
        return [{'model': model_name, 'task': task, 'metric': 'MSE', 'value': mse, 'resampled': resampled}]
    else:  # trend_prediction
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f'{model_name} (Resampled={resampled}) - Accuracy for trend prediction: {accuracy}')
        print(f'{model_name} (Resampled={resampled}) - F1-score for trend prediction: {f1}')
        print(f'{model_name} (Resampled={resampled}) - Recall for trend prediction: {recall}')
        return [
            {'model': model_name, 'task': task, 'metric': 'Accuracy', 'value': accuracy, 'resampled': resampled},
            {'model': model_name, 'task': task, 'metric': 'F1-score', 'value': f1, 'resampled': resampled},
            {'model': model_name, 'task': task, 'metric': 'Recall', 'value': recall, 'resampled': resampled}
        ]

def generate_visualization(model_name, model, X_train, y_train, X_test, y_test, task, data, result_folder, aggregation_window, resampled='no'):
    """Generate visualizations for model predictions and feature importance."""
    # Use the best model from grid search or the original model
    param_grid = param_grids.get(model_name, {})
    if param_grid:
        scoring = 'neg_mean_squared_error' if task == 'price_prediction' else 'f1'
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    os.makedirs(result_folder, exist_ok=True)

    if task == 'price_prediction':
        y_pred = best_model.predict(X_test)
        # Actual vs. Predicted Prices
        plt.figure(figsize=(10, 6))
        plt.plot(data['test_data']['datetime'], y_test, label='Actual Price', color='blue')
        plt.plot(data['test_data']['datetime'], y_pred, label='Predicted Price', color='red')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'Actual vs. Predicted Prices ({model_name})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig(f'{result_folder}{model_name.lower()}_actual_vs_predicted_prices_{aggregation_window.lower()}.png')
        plt.close()

        # Prediction Errors
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, color='gray')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Prediction Errors ({model_name})')
        plt.savefig(f'{result_folder}{model_name.lower()}_prediction_errors_{aggregation_window.lower()}.png')
        plt.close()

    else:  # trend_prediction
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Up Trend', 'Up Trend'])
        plt.figure(figsize=(8, 6))
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for Trend Prediction ({model_name}, Resampled={resampled})')
        plt.savefig(f'{result_folder}{model_name.lower()}_trend_confusion_matrix_{aggregation_window.lower()}_resampled_{resampled}.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Trend Prediction ({model_name}, Resampled={resampled})')
        plt.legend()
        plt.savefig(f'{result_folder}{model_name.lower()}_trend_roc_curve_{aggregation_window.lower()}_resampled_{resampled}.png')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        ap = average_precision_score(y_test, y_pred_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({model_name}, Resampled={resampled})')
        plt.legend()
        plt.savefig(f'{result_folder}{model_name.lower()}_trend_pr_curve_{aggregation_window.lower()}_resampled_{resampled}.png')
        plt.close()

    # Feature Importance (for models that support it)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': best_model.feature_importances_})
        importance_df = importance_df.sort_values('importance', ascending=False).head(30)
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance for {task.capitalize()} ({model_name}, Resampled={resampled})')
        plt.gca().invert_yaxis()
        plt.savefig(f'{result_folder}{model_name.lower()}_{task}_feature_importance_{aggregation_window.lower()}_resampled_{resampled}.png')
        plt.close()

def generate_general_visualizations(data, result_folder, aggregation_window):
    """Generate visualizations not specific to a model."""
    os.makedirs(result_folder, exist_ok=True)

    # Training and Testing Data Split
    plt.figure(figsize=(10, 6))
    plt.plot(data['train_data']['datetime'], data['train_data']['close'], label='Training Data', color='green')
    plt.plot(data['test_data']['datetime'], data['test_data']['close'], label='Testing Data', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Training and Testing Data Split')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'{result_folder}train_test_split_{aggregation_window.lower()}.png')
    plt.close()

    # Sentiment Scores Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(data['train_data']['datetime'], data['train_data']['overall_sentiment_score'], label='Overall Sentiment', color='purple')
    plt.plot(data['test_data']['datetime'], data['test_data']['overall_sentiment_score'], label='Overall Sentiment', color='purple')
    plt.plot(data['train_data']['datetime'], data['train_data']['sol_sentiment_score'], label='Solana Sentiment', color='cyan')
    plt.plot(data['test_data']['datetime'], data['test_data']['sol_sentiment_score'], label='Solana Sentiment', color='cyan')
    plt.xlabel('Time')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Scores Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'{result_folder}sentiment_scores_{aggregation_window.lower()}.png')
    plt.close()

    # Moving Averages
    plt.figure(figsize=(10, 6))
    plt.plot(data['train_data']['datetime'], data['train_data']['ma_5'], label='5-Day MA', color='orange')
    plt.plot(data['test_data']['datetime'], data['test_data']['ma_5'], label='5-Day MA', color='orange')
    plt.plot(data['train_data']['datetime'], data['train_data']['ma_10'], label='10-Day MA', color='red')
    plt.plot(data['test_data']['datetime'], data['test_data']['ma_10'], label='10-Day MA', color='red')
    plt.plot(data['train_data']['datetime'], data['train_data']['close'], label='Close Price', color='blue')
    plt.plot(data['test_data']['datetime'], data['test_data']['close'], label='Close Price', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Moving Averages and Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'{result_folder}moving_averages_{aggregation_window.lower()}.png')
    plt.close()

def sentiment_process_data_and_models(models_list, aggregation_window, data_path, result_folder, pred_trend='Upward', direction='up'):
    """
    Process Solana price and sentiment data with specified aggregation window and evaluate models.

    Parameters:
    - models_list (dict): Dictionary of model names and their instances.
    - aggregation_window (str): Aggregation level ('daily', 'hourly', 'minutely').
    - data_path (str): Path to input data files.
    - result_folder (str): Path to save results and plots.
    """
    # Prepare data
    data = prepare_data(data_path, aggregation_window, pred_trend, direction)

    # Apply SMOTE for classification
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_trend_resampled = smote.fit_resample(data['X_train'], data['y_train_trend'])

    # Results storage
    results = []

    # Train and evaluate models
    for model_name, model in models_list.items():
        task = 'price_prediction' if isinstance(model, RegressorMixin) else 'trend_prediction'
        if task == 'price_prediction':
            # Regression: Train only on original data
            results.extend(train_and_evaluate_model(
                model_name, model, data['X_train'], data['y_train_price'], data['X_test'], data['y_test_price'], task, resampled='no'
            ))
            generate_visualization(
                model_name, model, data['X_train'], data['y_train_price'], data['X_test'], data['y_test_price'], task,
                data, result_folder, aggregation_window, resampled='no'
            )
        else:
            # Classification: Train on both original and resampled data
            for resampled, X_train, y_train in [
                ('no', data['X_train'], data['y_train_trend']),
                ('yes', X_train_resampled, y_train_trend_resampled)
            ]:
                results.extend(train_and_evaluate_model(
                    model_name, model, X_train, y_train, data['X_test'], data['y_test_trend'], task, resampled
                ))
                generate_visualization(
                    model_name, model, X_train, y_train, data['X_test'], data['y_test_trend'], task,
                    data, result_folder, aggregation_window, resampled
                )
                # Cross-validation
                param_grid = param_grids.get(model_name, {})
                if param_grid:
                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
                print(f"{model_name} (Resampled={resampled}) - Cross-validated F1 score: {scores.mean():.3f} ± {scores.std():.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_folder}model_results_{aggregation_window.lower()}.csv', index=False)
    print(f"Results saved to '{result_folder}model_results_{aggregation_window.lower()}.csv'")

    # Generate general visualizations
    generate_general_visualizations(data, result_folder, aggregation_window)

if __name__ == "__main__":
    models_list = {
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'XGBRegressor': XGBRegressor(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'XGBClassifier': XGBClassifier(random_state=42, scale_pos_weight=0.09),
        'BalancedRandomForestClassifier': BalancedRandomForestClassifier(random_state=42),
    }

    aggregation_window = 'hourly'  # or 'daily', 'minutely'
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'minutely_aggrigated_sol_sentiment_price.csv')
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', f'{aggregation_window}_sentiment_price_hyperparameter/')

    # Create results folder if it doesn't exist
    os.makedirs(result_folder, exist_ok=True)

    sentiment_process_data_and_models(models_list, aggregation_window, data_path, result_folder)