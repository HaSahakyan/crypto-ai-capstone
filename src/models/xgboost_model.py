import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import optuna.logging
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='combined_target', drop_columns=None, n_splits=5):
        """
        Initialize the XGBoostModelEvaluator with result folder, datasets, and data preparation parameters.

        Args:
            result_folder (str): Path to save results and models
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
            n_splits (int): Number of train-test splits for model evaluation
        """
        self.result_folder = result_folder
        self.datasets = datasets
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'trend_label', 'combined_target'
        ]
        self.n_splits = n_splits
        os.makedirs(self.result_folder, exist_ok=True)

    def _prepare_data(self, df, use_target_column=True, encode_labels=False, two_class=False):
        """
        Prepare features and target for modeling, with optional label encoding.

        Args:
            df (pd.DataFrame): Input DataFrame
            use_target_column (bool): Whether to use specified target column or last column
            encode_labels (bool): Whether to encode target labels to [0, 1, 2, ...]
            two_class (bool): Whether to encode labels [1, 2] to [0, 1] for binary classification

        Returns:
            tuple: (X, y, label_mapping) features, target, and label mapping (if encoded)

        Raises:
            ValueError: If DataFrame is empty, target column is missing, or no valid features remain
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        X = df.drop(columns=self.drop_columns, errors='ignore')
        if X.empty:
            raise ValueError("No valid features remain after dropping specified columns")

        if use_target_column and self.target_column in df.columns:
            y = df[self.target_column]
        else:
            if df.shape[1] < 1:
                raise ValueError("DataFrame has no columns to use as target")
            y = df.iloc[:, -1]

        if y.isna().any():
            raise ValueError("Target column contains missing values")

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

    def _save_results(self, name, report, accuracy, recall_macro, f1_macro, y_test, y_pred, model_name='xgboost', best_params=None, param_str=None):
        """
        Save evaluation results, including classification report, metrics, and confusion matrix.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            recall_macro (float): Macro-averaged recall score
            f1_macro (float): Macro-averaged F1 score
            y_test (pd.Series or np.array): True labels
            y_pred (pd.Series or np.array): Predicted labels
            model_name (str): Name of the model (e.g., xgboost, xgboost_two_classes)
            best_params (dict, optional): Best parameters from optimization
            param_str (str, optional): Parameter string for filename
        """
        # Define base filename
        filename_base = os.path.join(self.result_folder, f"{model_name}_{name}")
        if param_str:
            filename_base += f"_{param_str}"

        # Save text results (classification report and metrics)
        text_filename = f"{filename_base}.txt"
        with open(text_filename, "w") as f:
            f.write(f"=== {model_name.capitalize()} Results for dataset: {name} ===\n")
            if best_params:
                f.write(f"Best Parameters: {best_params}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro-averaged Recall: {recall_macro:.4f}\n")
            f.write(f"Macro-averaged F1-score: {f1_macro:.4f}\n")

        print(f"Saved results to: {text_filename}")

        # Compute and save confusion matrix as CSV
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
        cm_filename = f"{filename_base}_confusion_matrix.csv"
        cm_df.to_csv(cm_filename)
        print(f"Saved confusion matrix to: {cm_filename}")

        # Optional: Save confusion matrix as a heatmap image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix - {model_name} - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_image_filename = f"{filename_base}_confusion_matrix.png"
        plt.savefig(cm_image_filename, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix heatmap to: {cm_image_filename}")

    def _save_best_model(self, model, model_name, dataset_name, param_str=None):
        """
        Save the best-performing model to a .joblib file.

        Args:
            model: Trained model object
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            param_str (str, optional): Parameter string for filename
        """
        filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_best")
        if param_str:
            filename += f"_{param_str}"
        filename += ".joblib"
        joblib.dump(model, filename)
        print(f"Saved best model to: {filename}")

    def _compute_composite_score(self, accuracy, recall_macro, f1_macro, weights=(0.2, 0.4, 0.4)):
        """
        Compute a composite score from accuracy, recall, and F1-score.

        Args:
            accuracy (float): Accuracy score
            recall_macro (float): Macro-averaged recall score
            f1_macro (float): Macro-averaged F1 score
            weights (tuple): Weights for (accuracy, recall_macro, f1_macro)

        Returns:
            float: Composite score
        """
        return weights[0] * accuracy + weights[1] * recall_macro + weights[2] * f1_macro

    def simple_xgboost(self, params=None):
        """
        Train and evaluate a simple XGBoost classifier, saving the best-performing model.
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softmax',
                'eval_metric': 'mlogloss',
                'scale_pos_weight': np.sqrt(20.0 / 4.0),
                'n_jobs': -1,
                'random_state': 42
            }

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Simple XGBoost on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, encode_labels=True)
                best_composite_score = -1
                best_model = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Split {split + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = model

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                y_pred = best_model.predict(X_test)
                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                self._save_results(name, report, accuracy, recall_macro, f1_macro, y_test, y_pred, model_name='xgboost')
                self._save_best_model(best_model, 'xgboost', name)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def optuna_xgboost(self, n_trials=50):
        """
        Train and evaluate an XGBoost classifier with hyperparameter tuning using Optuna, saving the best-performing model.
        """
        def objective(trial, X_train, y_train, X_test, y_test):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'objective': 'multi:softmax',
                'eval_metric': 'mlogloss',
                'n_jobs': -1,
                'random_state': 42
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            return self._compute_composite_score(accuracy, recall_macro, f1_macro)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Optuna XGBoost on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, encode_labels=True)
                best_composite_score = -1
                best_model = None
                best_params = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    study = optuna.create_study(direction='maximize', sampler=TPESampler())
                    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
                    split_params = study.best_params
                    split_params.update({
                        'objective': 'multi:softmax',
                        'eval_metric': 'mlogloss',
                        'n_jobs': -1,
                        'random_state': 42
                    })

                    model = xgb.XGBClassifier(**split_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Split {split + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = model
                        best_params = split_params

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                y_pred = best_model.predict(X_test)
                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                param_str = f"optuna_ntrials{n_trials}"
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                self._save_results(name, report, accuracy, recall_macro, f1_macro, y_test, y_pred, model_name='xgboost', best_params=best_params, param_str=param_str)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def simple_xgboost_two_classes(self, params=None):
        """
        Train and evaluate a simple XGBoost classifier on datasets with only classes 1 and 2, saving the best-performing model.
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'scale_pos_weight': np.sqrt(20.0 / 4.0),
                'n_jobs': -1,
                'random_state': 42
            }

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Simple XGBoost (two classes) on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, use_target_column=False, two_class=True)
                best_composite_score = -1
                best_model = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Split {split + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = model

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                y_pred = best_model.predict(X_test)
                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                self._save_results(name, report, accuracy, recall_macro, f1_macro, y_test, y_pred, model_name='xgboost_two_classes')
                self._save_best_model(best_model, 'xgboost_two_classes', name)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def optuna_xgboost_two_classes(self, n_trials=50):
        """
        Train and evaluate an XGBoost classifier with Optuna for datasets with only classes 1 and 2, saving the best-performing model.
        """
        def objective(trial, X_train, y_train, X_test, y_test):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_jobs': -1,
                'random_state': 42
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            return self._compute_composite_score(accuracy, recall_macro, f1_macro)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Optuna XGBoost (two classes) on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, use_target_column=False, two_class=True)
                best_composite_score = -1
                best_model = None
                best_params = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    study = optuna.create_study(direction='maximize', sampler=TPESampler())
                    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
                    split_params = study.best_params
                    split_params.update({
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'n_jobs': -1,
                        'random_state': 42
                    })

                    model = xgb.XGBClassifier(**split_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Split {split + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = model
                        best_params = split_params

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                y_pred = best_model.predict(X_test)
                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                param_str = f"optuna_ntrials{n_trials}"
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                # Update this line to include y_test and y_pred
                self._save_results(name, report, accuracy, recall_macro, f1_macro, y_test, y_pred, model_name='xgboost_two_classes', best_params=best_params, param_str=param_str)
                self._save_best_model(best_model, 'xgboost_two_classes', name, param_str=param_str)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

if __name__ == "__main__":
    # Define result folder and dataset paths using relative paths
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'xgboost')
    dataset_base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets')

    # Load datasets
    datasets = {
        'agg30': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg30.csv')),
        'hourly': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_hourly.csv')),
        'daily': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_daily.csv')),
        'google_combined': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_google.csv')),
    }

    # Filter datasets for two classes (1 and 2)
    filtered_datasets = {
        name: df[df['combined_target'].isin([1, 2])].copy() for name, df in datasets.items()
    }

    # Initialize evaluator for multiclass
    xgboost_evaluator = XGBoostModelEvaluator(
        result_folder=os.path.join(result_folder, 'multiclass'),
        datasets=datasets,
        target_column='combined_target',
        n_splits=5
    )

    # Run simple XGBoost
    xgboost_evaluator.simple_xgboost()

    # Run Optuna-tuned XGBoost
    xgboost_evaluator.optuna_xgboost(n_trials=20)

    # Initialize evaluator for two classes
    xgboost_evaluator_two_classes = XGBoostModelEvaluator(
        result_folder=os.path.join(result_folder, 'two_classes'),
        datasets=filtered_datasets,
        target_column='combined_target',
        n_splits=5
    )

    # Run simple XGBoost for two classes
    xgboost_evaluator_two_classes.simple_xgboost_two_classes()

    # Run Optuna-tuned XGBoost for two classes
    xgboost_evaluator_two_classes.optuna_xgboost_two_classes(n_trials=20)