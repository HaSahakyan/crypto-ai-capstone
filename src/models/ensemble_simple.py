import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

class EnsembleModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='combined_target', drop_columns=None, weights=None):
        """
        Initialize the EnsembleModelEvaluator with result folder, datasets, and model weights.

        Args:
            result_folder (str): Path to save results
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
            weights (list): Weights for [XGBoost, CatBoost, RandomForest, DecisionTree]
        """
        self.result_folder = result_folder
        self.datasets = datasets
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'trend_label', 'combined_target'
        ]
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0]
        os.makedirs(self.result_folder, exist_ok=True)

    def _prepare_data(self, df, encode_labels=False, two_class=False):
        """
        Prepare features and target for modeling, with optional label encoding and NaN imputation.

        Args:
            df (pd.DataFrame): Input DataFrame
            encode_labels (bool): Whether to encode target labels to [0, 1, 2, ...]
            two_class (bool): Whether to encode labels [1, 2] to [0, 1] for binary classification

        Returns:
            tuple: (X, y, label_mapping) features, target, and label mapping (if encoded)
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        X = df.drop(columns=self.drop_columns, errors='ignore')
        if X.empty:
            raise ValueError("No valid features remain after dropping specified columns")

        if self.target_column in df.columns:
            y = df[self.target_column]
        else:
            raise ValueError(f"Target column {self.target_column} not found")

        if y.isna().any():
            raise ValueError("Target column contains missing values")

        # Impute NaNs in features
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        label_mapping = None
        if encode_labels:
            unique_labels = np.sort(y.unique())
            if len(unique_labels) < 2:
                raise ValueError("Target column has fewer than 2 unique classes")
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            reverse_mapping = {idx: label for label, idx in label_mapping.items()}
            y = y.map(label_mapping)
            return X, y, (label_mapping, reverse_mapping)

        if two_class:
            if not set(y.unique()).issubset({1, 2}):
                raise ValueError("Target column for two-class model must contain only values 1 and 2")
            label_mapping = {1: 0, 2: 1}
            reverse_mapping = {0: 1, 1: 2}
            y = y.map(label_mapping)
            return X, y, (label_mapping, reverse_mapping)

        return X, y, label_mapping

    def _save_results(self, name, report, accuracy, confusion_matrix_str, model_name, loss_df=None):
        """
        Save evaluation results and loss history to files.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            confusion_matrix_str (str): Confusion matrix as a formatted string
            model_name (str): Name of the model
            loss_df (pd.DataFrame, optional): DataFrame with loss history
        """
        filename = f"{self.result_folder}/{model_name}_{name}.txt"
        with open(filename, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {name}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(confusion_matrix_str + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
        print(f"Saved results to: {filename}")

        if loss_df is not None:
            loss_filename = f"{self.result_folder}/{model_name}_{name}_loss.csv"
            loss_df.to_csv(loss_filename, index=False)
            print(f"Saved loss history to: {loss_filename}")

    def _plot_confusion_matrix(self, cm, model_name, dataset_name, labels):
        """
        Plot and save the confusion matrix as a heatmap.

        Args:
            cm (np.ndarray): Confusion matrix
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            labels (list): List of class labels
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        filename = f"{self.result_folder}/{model_name}_{dataset_name}_cm.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to: {filename}")

    def _plot_loss(self, loss_df, model_name, dataset_name):
        """
        Plot and save the loss history for each model.

        Args:
            loss_df (pd.DataFrame): DataFrame with loss history
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
        """
        plt.figure(figsize=(10, 6))
        for column in loss_df.columns:
            plt.plot(loss_df[column], label=column)
        plt.title(f'Loss History: {model_name} ({dataset_name})')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.legend()
        filename = f"{self.result_folder}/{model_name}_{dataset_name}_loss.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved loss plot to: {filename}")

    def ensemble_multiclass(self):
        """
        Train and evaluate an ensemble classifier for multiclass [0, 1, 2].
        """
        model_name = 'ensemble_multiclass'
        xgb_params = {
            'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        cat_params = {
            'learning_rate': 0.05, 'depth': 6, 'iterations': 200,
            'random_seed': 42, 'verbose': 0, 'l2_leaf_reg': 10,
            'loss_function': 'MultiClass'
        }
        rf_params = {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
        dt_params = {'max_depth': 6, 'random_state': 42}

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, encode_labels=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Initialize models
                xgb_model = XGBClassifier(**xgb_params)
                cat_model = CatBoostClassifier(**cat_params)
                rf_model = RandomForestClassifier(**rf_params)
                dt_model = DecisionTreeClassifier(**dt_params)

                # Track loss
                xgb_loss = []
                cat_loss = []
                rf_loss = []

                # XGBoost: Use eval_set to track log loss
                xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
                xgb_results = xgb_model.evals_result()
                xgb_loss = xgb_results['validation_1']['mlogloss']

                # CatBoost: Use eval_metrics to track log loss
                cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
                evals_result = cat_model.get_evals_result()
                if 'validation' in evals_result and 'MultiClass' in evals_result['validation']:
                    cat_loss = evals_result['validation']['MultiClass']
                else:
                    print(f"Warning: MultiClass metric not found for {name}. Available metrics: {evals_result.get('validation', {})}")
                    cat_loss = [log_loss(y_test, cat_model.predict_proba(X_test))]

                # RandomForest: Compute log loss after each tree
                rf_model.fit(X_train, y_train)
                for n_trees in range(1, rf_params['n_estimators'] + 1):
                    partial_rf = RandomForestClassifier(n_estimators=n_trees, max_depth=6, random_state=42)
                    partial_rf.fit(X_train, y_train)
                    y_pred_proba = partial_rf.predict_proba(X_test)
                    rf_loss.append(log_loss(y_test, y_pred_proba))

                # DecisionTree: Single log loss
                dt_model.fit(X_train, y_train)
                dt_loss = [log_loss(y_test, dt_model.predict_proba(X_test))]

                # Ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('xgb', xgb_model),
                        ('cat', cat_model),
                        ('rf', rf_model),
                        ('dt', dt_model)
                    ],
                    voting='soft',
                    weights=self.weights
                )
                ensemble.fit(X_train, y_train)

                # Predictions
                y_pred = ensemble.predict(X_test)
                if label_mapping:
                    reverse_mapping = label_mapping[1]
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                # Evaluation
                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print("Accuracy:", accuracy)
                print("Confusion Matrix:\n", cm_str)

                # Plot confusion matrix
                labels = [0, 1, 2]
                self._plot_confusion_matrix(cm, model_name, name, labels)

                # Save loss history
                max_len = max(len(xgb_loss), len(cat_loss), len(rf_loss), len(dt_loss))
                loss_df = pd.DataFrame({
                    'XGBoost': np.pad(xgb_loss, (0, max_len - len(xgb_loss)), mode='edge'),
                    'CatBoost': np.pad(cat_loss, (0, max_len - len(cat_loss)), mode='edge'),
                    'RandomForest': np.pad(rf_loss, (0, max_len - len(rf_loss)), mode='edge'),
                    'DecisionTree': np.pad(dt_loss, (0, max_len - len(dt_loss)), mode='edge')
                })
                self._save_results(name, report, accuracy, cm_str, model_name, loss_df)

                # Plot loss
                self._plot_loss(loss_df, model_name, name)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def ensemble_two_classes(self, smote=False):
        """
        Train and evaluate an ensemble classifier for binary [1, 2], with optional SMOTE.

        Args:
            smote (bool): Whether to apply SMOTE oversampling
        """
        model_name = 'ensemble_two_classes' + ('_smote' if smote else '')
        xgb_params = {
            'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'eval_metric': 'logloss'
        }
        cat_params = {
            'learning_rate': 0.05, 'depth': 6, 'iterations': 200,
            'random_seed': 42, 'verbose': 0, 'l2_leaf_reg': 10,
            'loss_function': 'Logloss'
        }
        rf_params = {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
        dt_params = {'max_depth': 6, 'random_state': 42}

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, two_class=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Apply SMOTE
                if smote:
                    smote_obj = SMOTE(random_state=42)
                    try:
                        X_train, y_train = smote_obj.fit_resample(X_train, y_train)
                        print(f"Applied SMOTE to {name}: New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                    except ValueError as e:
                        print(f"SMOTE failed for {name}: {str(e)}. Using original training data.")

                # Initialize models
                xgb_model = XGBClassifier(**xgb_params)
                cat_model = CatBoostClassifier(**cat_params)
                rf_model = RandomForestClassifier(**rf_params)
                dt_model = DecisionTreeClassifier(**dt_params)

                # Track loss
                xgb_loss = []
                cat_loss = []
                rf_loss = []

                # XGBoost: Use eval_set to track log loss
                xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
                xgb_results = xgb_model.evals_result()
                xgb_loss = xgb_results['validation_1']['logloss']

                # CatBoost: Use eval_metrics to track log loss
                cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
                evals_result = cat_model.get_evals_result()
                if 'validation' in evals_result and 'Logloss' in evals_result['validation']:
                    cat_loss = evals_result['validation']['Logloss']
                else:
                    print(f"Warning: Logloss metric not found for {name}. Available metrics: {evals_result.get('validation', {})}")
                    cat_loss = [log_loss(y_test, cat_model.predict_proba(X_test))]

                # RandomForest: Compute log loss after each tree
                rf_model.fit(X_train, y_train)
                for n_trees in range(1, rf_params['n_estimators'] + 1):
                    partial_rf = RandomForestClassifier(n_estimators=n_trees, max_depth=6, random_state=42)
                    partial_rf.fit(X_train, y_train)
                    y_pred_proba = partial_rf.predict_proba(X_test)
                    rf_loss.append(log_loss(y_test, y_pred_proba))

                # DecisionTree: Single log loss
                dt_model.fit(X_train, y_train)
                dt_loss = [log_loss(y_test, dt_model.predict_proba(X_test))]

                # Ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('xgb', xgb_model),
                        ('cat', cat_model),
                        ('rf', rf_model),
                        ('dt', dt_model)
                    ],
                    voting='soft',
                    weights=self.weights
                )
                ensemble.fit(X_train, y_train)

                # Predictions
                y_pred = ensemble.predict(X_test)
                if label_mapping:
                    reverse_mapping = label_mapping[1]
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                # Evaluation
                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print("Accuracy:", accuracy)
                print("Confusion Matrix:\n", cm_str)

                # Plot confusion matrix
                labels = [1, 2]
                self._plot_confusion_matrix(cm, model_name, name, labels)

                # Save loss history
                max_len = max(len(xgb_loss), len(cat_loss), len(rf_loss), len(dt_loss))
                loss_df = pd.DataFrame({
                    'XGBoost': np.pad(xgb_loss, (0, max_len - len(xgb_loss)), mode='edge'),
                    'CatBoost': np.pad(cat_loss, (0, max_len - len(cat_loss)), mode='edge'),
                    'RandomForest': np.pad(rf_loss, (0, max_len - len(rf_loss)), mode='edge'),
                    'DecisionTree': np.pad(dt_loss, (0, max_len - len(dt_loss)), mode='edge')
                })
                self._save_results(name, report, accuracy, cm_str, model_name, loss_df)

                # Plot loss
                self._plot_loss(loss_df, model_name, name)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Define datasets
    result_folder =os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Ensembles', 'ensemble_simple')

    prep_sol_binance_agg30 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg30.csv'))
    prep_sol_binance_hourly = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly.csv'))
    prep_sol_binance_daily = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily.csv'))
    prep_sol_binance_google = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google.csv'))
    prep_sol_binance_hourly_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly_sentiment.csv'))
    prep_sol_binance_daily_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily_sentiment.csv'))
    prep_sol_binance_google_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google_sentiment.csv'))
    datasets = {
        'agg30': prep_sol_binance_agg30,
        'hourly': prep_sol_binance_hourly,
        'daily': prep_sol_binance_daily,
        'google_combined': prep_sol_binance_google,
        'hourly_sentiment': prep_sol_binance_hourly_sentiment,
        'daily_sentiment': prep_sol_binance_daily_sentiment,
        'google_sentiment': prep_sol_binance_google_sentiment,
    }

    # Filter datasets for two classes (1 and 2)
    filtered_datasets = {
        name: df[df['combined_target'].isin([1, 2])].copy() for name, df in datasets.items()
    }

    # Initialize evaluator for multiclass
    ensemble_evaluator = EnsembleModelEvaluator(
        result_folder=f'{result_folder}/three_classes',
        datasets=datasets,
        target_column='combined_target',
        weights=[2.0, 2.0, 1.0, 0.5]  # Example weights: higher for XGBoost/CatBoost
    )

    # Run multiclass ensemble
    ensemble_evaluator.ensemble_multiclass()

    # Initialize evaluator for two classes
    ensemble_evaluator_two_classes = EnsembleModelEvaluator(
        result_folder=f'{result_folder}/two_classes',
        datasets=filtered_datasets,
        target_column='combined_target',
        weights=[2.0, 2.0, 1.0, 0.5]
    )

    # Run two-class ensemble without SMOTE
    ensemble_evaluator_two_classes.ensemble_two_classes(smote=False)

    # Run two-class ensemble with SMOTE
    ensemble_evaluator_two_classes.ensemble_two_classes(smote=True)