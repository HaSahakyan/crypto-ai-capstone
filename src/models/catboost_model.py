import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, recall_score
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import optuna.logging
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CatBoostModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='combined_target', drop_columns=None, n_splits=5):
        """
        Initialize the CatBoostModelEvaluator with result folder, datasets, and data preparation parameters.

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
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'combined_target'
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

    def _save_results(self, name, report, accuracy, recall_macro, f1_macro, confusion_matrix_str, model_name='catboost', best_params=None, param_str=None):
        """
        Save evaluation results to a file, including metrics and confusion matrix.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            recall_macro (float): Macro-averaged recall score
            f1_macro (float): Macro-averaged F1 score
            confusion_matrix_str (str): Confusion matrix as a formatted string
            model_name (str): Name of the model
            best_params (dict, optional): Best parameters from optimization
            param_str (str, optional): Parameter string for filename
        """
        filename = os.path.join(self.result_folder, f"{model_name}_{name}")
        if param_str:
            filename += f"_{param_str}"
        filename += ".txt"

        with open(filename, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {name}\n\n")
            if best_params:
                f.write(f"Best Parameters: {best_params}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(confusion_matrix_str + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro-averaged Recall: {recall_macro:.4f}\n")
            f.write(f"Macro-averaged F1-score: {f1_macro:.4f}\n")

        print(f"Saved results to: {filename}")

    def _save_confusion_matrix(self, cm, y_test, accuracy, model_name, dataset_name, display_labels):
        """
        Save confusion matrix as a heatmap to a file.

        Args:
            cm (np.ndarray): Confusion matrix
            y_test (array-like): True labels
            accuracy (float): Accuracy score
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            display_labels (list): Labels for display
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = range(len(display_labels))
        plt.xticks(tick_marks, display_labels)
        plt.yticks(tick_marks, display_labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_cm.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to: {filename}")

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

    def simple_catboost(self, params=None):
        """
        Train and evaluate a simple CatBoost classifier for multiclass [0, 1, 2], saving the best model.
        """
        model_name = 'simple_catboost'
        if params is None:
            params = {
                'depth': 6,
                'iterations': 100,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': 0
            }

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, encode_labels=True)
                best_composite_score = -1
                best_model = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    model = CatBoostClassifier(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name)
                self._save_best_model(best_model, model_name, name)
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Flat', 'Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def optuna_catboost(self, n_trials=50):
        """
        Train and evaluate a CatBoost classifier with Optuna for multiclass [0, 1, 2], saving the best model.
        """
        model_name = 'optuna_catboost'
        def objective(trial, X_train, y_train, X_test, y_test):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'iterations': trial.suggest_int('iterations', 50, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                'loss_function': 'MultiClass',
                'random_seed': 42,
                'verbose': 0
            }

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if y_pred.ndim == 2:
                if y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")
            accuracy = accuracy_score(y_test, y_pred)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            return self._compute_composite_score(accuracy, recall_macro, f1_macro)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
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
                        'loss_function': 'MultiClass',
                        'random_seed': 42,
                        'verbose': 0
                    })

                    model = CatBoostClassifier(**split_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name, best_params=best_params, param_str=f"optuna_ntrials{n_trials}")
                self._save_best_model(best_model, model_name, name, param_str=f"optuna_ntrials{n_trials}")
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Flat', 'Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def simple_catboost_two_classes(self, params=None):
        """
        Train and evaluate a simple CatBoost classifier for binary [1, 2], saving the best model.
        """
        model_name = 'simple_catboost_two_classes'
        if params is None:
            params = {
                'depth': 6,
                'iterations': 100,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'loss_function': 'Logloss',
                'random_seed': 42,
                'verbose': 0
            }

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, use_target_column=False, two_class=True)
                best_composite_score = -1
                best_model = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    model = CatBoostClassifier(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name)
                self._save_best_model(best_model, model_name, name)
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def simple_catboost_two_classes_smote(self, params=None):
        """
        Train and evaluate a simple CatBoost classifier for binary [1, 2] with SMOTE, saving the best model.
        """
        model_name = 'simple_catboost_two_classes_smote'
        if params is None:
            params = {
                'depth': 6,
                'iterations': 100,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'loss_function': 'Logloss',
                'random_seed': 42,
                'verbose': 0
            }

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, use_target_column=False, two_class=True)
                best_composite_score = -1
                best_model = None
                reverse_mapping = label_mapping[1] if label_mapping else None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    # Apply SMOTE
                    smote = SMOTE(random_state=42)
                    try:
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                        print(f"Split {split + 1} Applied SMOTE to {name}: New training class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
                    except ValueError as e:
                        print(f"Split {split + 1} SMOTE failed for {name}: {str(e)}. Using original training data.")
                        X_train_res, y_train_res = X_train, y_train

                    model = CatBoostClassifier(**params)
                    model.fit(X_train_res, y_train_res)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                smote = SMOTE(random_state=42)
                try:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"Applied SMOTE to {name} for final model: New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except ValueError as e:
                    print(f"SMOTE failed for {name} for final model: {str(e)}. Using original training data.")

                best_model.fit(X_train, y_train)  # Retrain on final split
                y_pred = best_model.predict(X_test)
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name)
                self._save_best_model(best_model, model_name, name)
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def optuna_catboost_two_classes(self, n_trials=50):
        """
        Train and evaluate a CatBoost classifier with Optuna for binary [1, 2], saving the best model.
        """
        model_name = 'optuna_catboost_two_classes'
        def objective(trial, X_train, y_train, X_test, y_test):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'iterations': trial.suggest_int('iterations', 50, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                'loss_function': 'Logloss',
                'random_seed': 42,
                'verbose': 0
            }

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if y_pred.ndim == 2:
                if y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")
            accuracy = accuracy_score(y_test, y_pred)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            return self._compute_composite_score(accuracy, recall_macro, f1_macro)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
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
                        'loss_function': 'Logloss',
                        'random_seed': 42,
                        'verbose': 0
                    })

                    model = CatBoostClassifier(**split_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name, best_params=best_params, param_str=f"optuna_ntrials{n_trials}")
                self._save_best_model(best_model, model_name, name, param_str=f"optuna_ntrials{n_trials}")
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def optuna_catboost_two_classes_smote(self, n_trials=50):
        """
        Train and evaluate a CatBoost classifier with Optuna for binary [1, 2] with SMOTE, saving the best model.
        """
        model_name = 'optuna_catboost_two_classes_smote'
        def objective(trial, X_train, y_train, X_test, y_test):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'iterations': trial.suggest_int('iterations', 50, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                'loss_function': 'Logloss',
                'random_seed': 42,
                'verbose': 0
            }

            smote = SMOTE(random_state=42)
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            except ValueError:
                X_train_res, y_train_res = X_train, y_train

            model = CatBoostClassifier(**params)
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            if y_pred.ndim == 2:
                if y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")
            accuracy = accuracy_score(y_test, y_pred)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            return self._compute_composite_score(accuracy, recall_macro, f1_macro)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
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
                        'loss_function': 'Logloss',
                        'random_seed': 42,
                        'verbose': 0
                    })

                    smote = SMOTE(random_state=42)
                    try:
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                        print(f"Split {split + 1} Applied SMOTE to {name}: New training class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
                    except ValueError as e:
                        print(f"Split {split + 1} SMOTE failed for {name}: {str(e)}. Using original training data.")
                        X_train_res, y_train_res = X_train, y_train

                    model = CatBoostClassifier(**split_params)
                    model.fit(X_train_res, y_train_res)
                    y_pred = model.predict(X_test)
                    if y_pred.ndim == 2:
                        if y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                        else:
                            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

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
                smote = SMOTE(random_state=42)
                try:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"Applied SMOTE to {name} for final model: New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except ValueError as e:
                    print(f"SMOTE failed for {name} for final model: {str(e)}. Using original training data.")

                best_model.fit(X_train, y_train)  # Retrain on final split
                y_pred = best_model.predict(X_test)
                if y_pred.ndim == 2:
                    if y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                    else:
                        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

                if reverse_mapping:
                    y_pred = pd.Series(y_pred).map(reverse_mapping)
                    y_test = y_test.map(reverse_mapping)

                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                print("Confusion Matrix:\n", cm_str)
                self._save_results(name, report, accuracy, recall_macro, f1_macro, cm_str, model_name=model_name, best_params=best_params, param_str=f"optuna_ntrials{n_trials}")
                self._save_best_model(best_model, model_name, name, param_str=f"optuna_ntrials{n_trials}")
                self._save_confusion_matrix(cm, y_test, accuracy, model_name, name, ['Down', 'Up'])

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

if __name__ == "__main__":
    # Define result folder and dataset paths using relative paths
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'catboost')
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
    catboost_evaluator = CatBoostModelEvaluator(
        result_folder=os.path.join(result_folder, 'multiclass'),
        datasets=datasets,
        target_column='combined_target',
        n_splits=5
    )

    # Run simple CatBoost
    catboost_evaluator.simple_catboost()

    # Run Optuna-tuned CatBoost
    catboost_evaluator.optuna_catboost(n_trials=20)

    # Initialize evaluator for two classes
    catboost_evaluator_two_classes = CatBoostModelEvaluator(
        result_folder=os.path.join(result_folder, 'two_classes'),
        datasets=filtered_datasets,
        target_column='combined_target',
        n_splits=5
    )

    # Run simple CatBoost for two classes
    catboost_evaluator_two_classes.simple_catboost_two_classes()

    # Run simple CatBoost for two classes with SMOTE
    catboost_evaluator_two_classes.simple_catboost_two_classes_smote()

    # Run Optuna-tuned CatBoost for two classes
    catboost_evaluator_two_classes.optuna_catboost_two_classes(n_trials=20)

    # Run Optuna-tuned CatBoost for two classes with SMOTE
    catboost_evaluator_two_classes.optuna_catboost_two_classes_smote(n_trials=20)