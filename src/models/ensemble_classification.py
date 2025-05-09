import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, cohen_kappa_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class EnsembleClassifier:
    def __init__(self, mode='voting', result_folder='results', datasets=None, target_column='combined_target', drop_columns=None, random_state=42, n_splits=5):
        """
        Initialize the EnsembleClassifier with mode, result folder, datasets, and other parameters.

        Args:
            mode (str): 'stacking' or 'voting'
            result_folder (str): Path to save results and models
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
            random_state (int): Random seed for reproducibility
            n_splits (int): Number of splits for TimeSeriesSplit
        """
        self.mode = mode
        self.result_folder = result_folder
        self.datasets = datasets if datasets is not None else {}
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'combined_target'
        ]
        self.random_state = random_state
        self.n_splits = n_splits
        self.fitted = False

        # Create result folder
        os.makedirs(self.result_folder, exist_ok=True)

        # Initialize base models
        self.base_models = {
            'xgboost': XGBClassifier(random_state=random_state, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'catboost': CatBoostClassifier(random_state=random_state, verbose=0),
            'logistic': LogisticRegression(random_state=random_state, max_iter=1000),
            'gboost': GradientBoostingClassifier(random_state=random_state)
        }

        if mode == 'stacking':
            self.meta_learner = LogisticRegression(max_iter=1000)
            self.model = StackingClassifier(
                estimators=[(name, model) for name, model in self.base_models.items()],
                final_estimator=self.meta_learner,
                passthrough=True,
                cv=5
            )
        elif mode == 'voting':
            self.model = None  # Custom voting logic
        else:
            raise ValueError("Mode must be 'stacking' or 'voting'")

    def _prepare_data(self, df, two_class=False):
        """
        Prepare features and target, imputing NaNs and optionally filtering for two classes.

        Args:
            df (pd.DataFrame): Input DataFrame
            two_class (bool): If True, filter for classes [1, 2] and map to [0, 1]

        Returns:
            tuple: (X, y, label_mapping) features, target, and label mapping (if two_class=True)
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Filter for two classes if specified
        if two_class:
            df = df[df[self.target_column].isin([1, 2])].copy()
            if df.empty:
                raise ValueError("No data remains after filtering for classes [1, 2]")

        X = df.drop(columns=self.drop_columns, errors='ignore')
        if X.empty:
            raise ValueError("No valid features remain after dropping specified columns")

        y = df[self.target_column]
        if y.isna().any():
            raise ValueError("Target column contains missing values")

        # Impute NaNs in features
        X = X.fillna(X.mean()).astype(float)

        label_mapping = None
        if two_class:
            label_mapping = {1: 0, 2: 1}
            reverse_mapping = {0: 1, 1: 2}
            y = y.map(label_mapping)
            return X, y, (label_mapping, reverse_mapping)

        return X, y, label_mapping

    def _save_results(self, name, report, accuracy, qwk, recall_macro, f1_macro, confusion_matrix_str, model_name):
        """
        Save evaluation results to a text file.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            qwk (float): Quadratic Weighted Kappa score
            recall_macro (float): Macro-averaged recall score
            f1_macro (float): Macro-averaged F1 score
            confusion_matrix_str (str): Confusion matrix as a formatted string
            model_name (str): Name of the model
        """
        filename = os.path.join(self.result_folder, f"{model_name}_{name}.txt")
        with open(filename, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {name}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(confusion_matrix_str + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"QWK: {qwk:.4f}\n")
            f.write(f"Macro-averaged Recall: {recall_macro:.4f}\n")
            f.write(f"Macro-averaged F1-score: {f1_macro:.4f}\n")
        print(f"Saved results to: {filename}")

    def _plot_confusion_matrix(self, cm, model_name, dataset_name, labels):
        """
        Plot and save the confusion matrix as a heatmap.

        Args:
            cm (np.ndarray): Confusion matrix
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            labels (list): List of class labels
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_cm.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to: {filename}")

    def _plot_actual_vs_predicted(self, y_test, y_pred, model_name, dataset_name):
        """
        Plot and save actual vs predicted scatter plot.

        Args:
            y_test (np.ndarray): Actual labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
        """
        plt.figure(figsize=(6, 5))
        plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6)
        plt.title(f'Actual vs Predicted: {model_name} ({dataset_name})')
        plt.xlabel('Sample Index')
        plt.ylabel('Target')
        plt.legend()
        filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_actual_vs_pred.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved actual vs predicted plot to: {filename}")

    def _save_best_model(self, best_model, model_name, dataset_name, best_model_name=None):
        """
        Save the best-performing model to a .joblib file.

        Args:
            best_model: Trained model object
            model_name (str): Name of the ensemble model
            dataset_name (str): Name of the dataset
            best_model_name (str, optional): Name of the base model (for voting mode)
        """
        if best_model_name:
            filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_best_{best_model_name}.joblib")
        else:
            filename = os.path.join(self.result_folder, f"{model_name}_{dataset_name}_best.joblib")
        joblib.dump(best_model, filename)
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

    def fit(self, X_train, y_train):
        """
        Fit the ensemble model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        if self.mode == 'stacking':
            self.model.fit(X_train, y_train)
        else:
            for model in self.base_models.values():
                model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, X_test):
        """
        Predict using the ensemble model.

        Args:
            X_test (pd.DataFrame): Test features

        Returns:
            np.ndarray: Predicted labelsิง

        """
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        if self.mode == 'stacking':
            return self.model.predict(X_test)
        else:
            predictions = np.zeros((X_test.shape[0], len(self.base_models)))
            for idx, (name, model) in enumerate(self.base_models.items()):
                pred = model.predict(X_test)
                predictions[:, idx] = pred
            return np.array([Counter(row).most_common(1)[0][0] for row in predictions])

    def evaluate(self, two_class=False, optimize_thresholds=False):
        """
        Evaluate the ensemble model on all datasets using TimeSeriesSplit, saving only the best-performing model
        based on a composite score of accuracy, recall, and F1-score.

        Args:
            two_class (bool): If True, evaluate on two classes [1, 2]
            optimize_thresholds (bool): If True, optimize thresholds (placeholder for future implementation)
        """
        model_name = f"ensemble_{'two_classes' if two_class else 'multiclass'}_{self.mode}"
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating {model_name} on dataset: {name} ===")
            try:
                X, y, label_mapping = self._prepare_data(df, two_class=two_class)
                oof_preds = np.zeros(len(y))
                oof_targets = np.zeros(len(y))
                best_composite_score = -1
                best_fold = None
                best_models = {}

                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Reset models for this fold
                    if self.mode == 'stacking':
                        self.model = StackingClassifier(
                            estimators=[(name, model.__class__(**model.get_params())) for name, model in self.base_models.items()],
                            final_estimator=LogisticRegression(max_iter=1000),
                            passthrough=True,
                            cv=5
                        )
                    else:
                        self.base_models = {
                            'xgboost': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                            'random_forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                            'catboost': CatBoostClassifier(random_state=self.random_state, verbose=0),
                            'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
                            'gboost': GradientBoostingClassifier(random_state=self.random_state)
                        }

                    # Train and evaluate
                    self.fit(X_train, y_train)
                    y_pred = self.predict(X_test)
                    fold_accuracy = accuracy_score(y_test, y_pred)
                    fold_recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    fold_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    fold_composite_score = self._compute_composite_score(fold_accuracy, fold_recall_macro, fold_f1_macro)

                    oof_preds[test_idx] = y_pred
                    oof_targets[test_idx] = y_test

                    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}, Recall (macro): {fold_recall_macro:.4f}, F1 (macro): {fold_f1_macro:.4f}, Composite Score: {fold_composite_score:.4f}")

                    # Store model if this fold is the best so far
                    if fold_composite_score > best_composite_score:
                        best_composite_score = fold_composite_score
                        best_fold = fold + 1
                        if self.mode == 'stacking':
                            best_models = {'stacking': self.model}
                        else:
                            best_models = {name: model for name, model in self.base_models.items()}

                print(f"Best fold: {best_fold} with Composite Score: {best_composite_score:.4f}")

                # Save the best model(s)
                for model_key, model in best_models.items():
                    self._save_best_model(model, model_name, name, best_model_name=model_key if self.mode == 'voting' else None)

                # Reverse mapping for evaluation if two_class
                if label_mapping:
                    reverse_mapping = label_mapping[1]
                    oof_preds = pd.Series(oof_preds).map(reverse_mapping)
                    oof_targets = pd.Series(oof_targets).map(reverse_mapping)

                # Compute final metrics
                report = classification_report(oof_targets, oof_preds)
                accuracy = accuracy_score(oof_targets, oof_preds)
                recallKappa_score = cohen_kappa_score(oof_targets, oof_preds, weights='quadratic')
                recall_macro = recall_score(oof_targets, oof_preds, average='macro', zero_division=0)
                f1_macro = f1_score(oof_targets, oof_preds, average='macro', zero_division=0)
                cm = confusion_matrix(oof_targets, oof_preds)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')
                mse = mean_squared_error(oof_targets, oof_preds)

                print(f"Model: {model_name}")
                print(report)
                print(f"Accuracy: {accuracy:.4f}")
                print(f"QWK: {qwk:.4f}")
                print(f"Macro-averaged Recall: {recall_macro:.4f}")
                print(f"Macro-averaged F1-score: {f1_macro:.4f}")
                print(f"Mean Squared Error: {mse:.4f}")
                print("Confusion Matrix:\n", cm_str)

                # Plot and save
                labels = [1, 2] if two_class else [0, 1, 2]
                self._plot_confusion_matrix(cm, model_name, name, labels)
                self._plot_actual_vs_predicted(oof_targets, oof_preds, model_name, name)
                self._save_results(name, report, accuracy, qwk, recall_macro, f1_macro, cm_str, model_name)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

if __name__ == "__main__":
    # Define result folder and dataset paths using relative paths
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Ensembles', 'ensemble_classifier')
    dataset_base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets')

    # Load datasets
    datasets = {
        'agg5': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg5.csv')),
        'agg10': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg10.csv')),
        'agg30': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg30.csv')),
        'hourly': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_hourly.csv')),
        'daily': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_daily.csv')),
        'google_combined': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_google.csv')),
        'hourly_sentiment': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_hourly_sentiment.csv')),
        'daily_sentiment': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_daily_sentiment.csv')),
        'google_sentiment': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_google_sentiment.csv')),
    }

    # Filter datasets for two classes
    filtered_datasets = {
        name: df[df['combined_target'].isin([1, 2])].copy() for name, df in datasets.items()
    }

    # Initialize and evaluate for multiclass
    multiclass_evaluator = EnsembleClassifier(
        mode='stacking',
        result_folder=os.path.join(result_folder, 'multiclass'),
        datasets=datasets,
        random_state=42,
        n_splits=5
    )
    multiclass_evaluator.evaluate(two_class=False)

    # Initialize and evaluate for two classes
    two_class_evaluator = EnsembleClassifier(
        mode='stacking',
        result_folder=os.path.join(result_folder, 'two_classes'),
        datasets=filtered_datasets,
        random_state=42,
        n_splits=5
    )
    two_class_evaluator.evaluate(two_class=True)

    # Repeat for voting mode
    multiclass_voting_evaluator = EnsembleClassifier(
        mode='voting',
        result_folder=os.path.join(result_folder, 'multiclass_voting'),
        datasets=datasets,
        random_state=42,
        n_splits=5
    )
    multiclass_voting_evaluator.evaluate(two_class=False)

    two_class_voting_evaluator = EnsembleClassifier(
        mode='voting',
        result_folder=os.path.join(result_folder, 'two_classes_voting'),
        datasets=filtered_datasets,
        random_state=42,
        n_splits=5
    )
    two_class_voting_evaluator.evaluate(two_class=True)