import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss, cohen_kappa_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

class EnsembleModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='combined_target', drop_columns=None, weights=None, n_splits=5):
        """
        Initialize the EnsembleModelEvaluator with result folder, datasets, and model weights.

        Args:
            result_folder (str): Path to save results
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
            weights (list): Weights for [XGBoost, CatBoost, RandomForest, DecisionTree]
            n_splits (int): Number of folds for Stratified K-Fold CV
        """
        self.result_folder = result_folder
        self.datasets = datasets
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'combined_target'
        ]
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0]
        self.n_splits = n_splits
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

    def _save_results(self, name, report, accuracy, qwk, confusion_matrix_str, model_name, loss_df=None, qwk_scores=None):
        """
        Save evaluation results, loss history, and QWK scores to files.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            qwk (float): Quadratic Weighted Kappa score
            confusion_matrix_str (str): Confusion matrix as a formatted string
            model_name (str): Name of the model
            loss_df (pd.DataFrame, optional): DataFrame with loss history
            qwk_scores (list, optional): QWK scores per fold
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
            f.write(f"QWK: {qwk:.4f}\n")
        print(f"Saved results to: {filename}")

        if loss_df is not None:
            loss_filename = f"{self.result_folder}/{model_name}_{name}_loss.csv"
            loss_df.to_csv(loss_filename, index=False)
            print(f"Saved loss history to: {loss_filename}")

        if qwk_scores is not None:
            qwk_filename = f"{self.result_folder}/{model_name}_{name}_qwk.csv"
            pd.DataFrame({'Fold': range(1, len(qwk_scores) + 1), 'QWK': qwk_scores}).to_csv(qwk_filename, index=False)
            print(f"Saved QWK scores to: {qwk_filename}")

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
        filename = f"{self.result_folder}/{model_name}_{dataset_name}_cm.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to: {filename}")

    def _plot_qwk(self, qwk_scores, model_name, dataset_name):
        """
        Plot and save QWK scores per fold.

        Args:
            qwk_scores (list): QWK scores for each fold
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
        """
        plt.figure(figsize=(6, 5))
        plt.bar(range(1, len(qwk_scores) + 1), qwk_scores)
        plt.title(f'QWK per Fold: {model_name} ({dataset_name})')
        plt.xlabel('Fold')
        plt.ylabel('QWK')
        filename = f"{self.result_folder}/{model_name}_{dataset_name}_qwk.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved QWK plot to: {filename}")

    def _optimize_qwk(self, oof_preds, oof_targets, n_classes):
        """
        Optimize thresholds for QWK using out-of-fold predictions.

        Args:
            oof_preds (np.ndarray): Out-of-fold prediction probabilities
            oof_targets (np.ndarray): Out-of-fold true labels
            n_classes (int): Number of classes

        Returns:
            np.ndarray: Optimized thresholds
        """
        def qwk_loss(thresholds):
            cutoffs = np.sort(thresholds)
            preds = np.digitize(np.argmax(oof_preds, axis=1), cutoffs)
            return -cohen_kappa_score(oof_targets, preds, weights='quadratic')

        initial = np.arange(0.5, n_classes - 0.5, 1.0)
        bounds = [(0, n_classes - 1) for _ in initial]
        res = minimize(qwk_loss, initial, method='nelder-mead', bounds=bounds)
        return np.sort(res.x)

    def ensemble_multiclass(self, optimize_thresholds=True):
        """
        Train and evaluate an ensemble classifier for multiclass [0, 1, 2] using Stratified K-Fold CV.

        Args:
            optimize_thresholds (bool): Whether to optimize QWK thresholds
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
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                oof_preds = np.zeros((len(X), len(np.unique(y))))
                oof_targets = np.zeros(len(X))
                qwk_scores = []
                xgb_losses = []
                cat_losses = []

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Initialize models
                    xgb_model = XGBClassifier(**xgb_params)
                    cat_model = CatBoostClassifier(**cat_params)
                    rf_model = RandomForestClassifier(**rf_params)
                    dt_model = DecisionTreeClassifier(**dt_params)

                    # Train models and track loss
                    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
                    xgb_results = xgb_model.evals_result()
                    xgb_losses.append(xgb_results['validation_1']['mlogloss'])

                    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
                    evals_result = cat_model.get_evals_result()
                    if 'validation' in evals_result and 'MultiClass' in evals_result['validation']:
                        cat_losses.append(evals_result['validation']['MultiClass'])
                    else:
                        print(f"Warning: MultiClass metric not found for {name}, fold {fold+1}. Using manual log loss.")
                        cat_losses.append([log_loss(y_val, cat_model.predict_proba(X_val))])

                    rf_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)

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

                    # OOF predictions
                    probas = ensemble.predict_proba(X_val)
                    oof_preds[val_idx] = probas
                    oof_targets[val_idx] = y_val

                    # QWK for fold
                    preds_raw = np.argmax(probas, axis=1)
                    qwk = cohen_kappa_score(y_val, preds_raw, weights='quadratic')
                    qwk_scores.append(qwk)
                    print(f"Fold {fold + 1} QWK (raw): {qwk:.4f}")

                print(f"\nMean QWK (raw): {np.mean(qwk_scores):.4f}")

                # Optimize thresholds
                if optimize_thresholds:
                    best_thresholds = self._optimize_qwk(oof_preds, oof_targets, n_classes=len(np.unique(y)))
                    final_preds = np.digitize(np.argmax(oof_preds, axis=1), best_thresholds)
                    print(f"Optimized thresholds: {best_thresholds.round(2)}")
                else:
                    final_preds = np.argmax(oof_preds, axis=1)

                # Final evaluation
                if label_mapping:
                    reverse_mapping = label_mapping[1]
                    final_preds = pd.Series(final_preds).map(reverse_mapping)
                    oof_targets = pd.Series(oof_targets).map(reverse_mapping)

                report = classification_report(oof_targets, final_preds)
                accuracy = accuracy_score(oof_targets, final_preds)
                qwk = cohen_kappa_score(oof_targets, final_preds, weights='quadratic')
                cm = confusion_matrix(oof_targets, final_preds)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print("Accuracy:", accuracy)
                print("QWK (final):", qwk)
                print("Confusion Matrix:\n", cm_str)

                # Compute final log loss
                rf_loss = [log_loss(oof_targets, oof_preds)]
                dt_loss = [log_loss(oof_targets, dt_model.predict_proba(X))]
                loss_df = pd.DataFrame({
                    'XGBoost': [np.mean([loss[-1] for loss in xgb_losses])],
                    'CatBoost': [np.mean([loss[-1] for loss in cat_losses])],
                    'RandomForest': rf_loss,
                    'DecisionTree': dt_loss
                })

                # Plot and save
                labels = [0, 1, 2]
                self._plot_confusion_matrix(cm, model_name, name, labels)
                self._plot_qwk(qwk_scores, model_name, name)
                self._save_results(name, report, accuracy, qwk, cm_str, model_name, loss_df, qwk_scores)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

    def ensemble_two_classes(self, smote=False, optimize_thresholds=True):
        """
        Train and evaluate an ensemble classifier for binary [1, 2] using Stratified K-Fold CV.

        Args:
            smote (bool): Whether to apply SMOTE oversampling
            optimize_thresholds (bool): Whether to optimize QWK thresholds
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
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                oof_preds = np.zeros((len(X), len(np.unique(y))))
                oof_targets = np.zeros(len(X))
                qwk_scores = []
                xgb_losses = []
                cat_losses = []

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Apply SMOTE
                    if smote:
                        smote_obj = SMOTE(random_state=42)
                        try:
                            X_train, y_train = smote_obj.fit_resample(X_train, y_train)
                            print(f"Fold {fold + 1} SMOTE applied: New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                        except ValueError as e:
                            print(f"Fold {fold + 1} SMOTE failed for {name}: {str(e)}. Using original training data.")

                    # Initialize models
                    xgb_model = XGBClassifier(**xgb_params)
                    cat_model = CatBoostClassifier(**cat_params)
                    rf_model = RandomForestClassifier(**rf_params)
                    dt_model = DecisionTreeClassifier(**dt_params)

                    # Train models and track loss
                    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
                    xgb_results = xgb_model.evals_result()
                    xgb_losses.append(xgb_results['validation_1']['logloss'])

                    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
                    evals_result = cat_model.get_evals_result()
                    if 'validation' in evals_result and 'Logloss' in evals_result['validation']:
                        cat_losses.append(evals_result['validation']['Logloss'])
                    else:
                        print(f"Warning: Logloss metric not found for {name}, fold {fold+1}. Using manual log loss.")
                        cat_losses.append([log_loss(y_val, cat_model.predict_proba(X_val))])

                    rf_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)

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

                    # OOF predictions
                    probas = ensemble.predict_proba(X_val)
                    oof_preds[val_idx] = probas
                    oof_targets[val_idx] = y_val

                    # QWK for fold
                    preds_raw = np.argmax(probas, axis=1)
                    qwk = cohen_kappa_score(y_val, preds_raw, weights='quadratic')
                    qwk_scores.append(qwk)
                    print(f"Fold {fold + 1} QWK (raw): {qwk:.4f}")

                print(f"\nMean QWK (raw): {np.mean(qwk_scores):.4f}")

                # Optimize thresholds
                if optimize_thresholds:
                    best_thresholds = self._optimize_qwk(oof_preds, oof_targets, n_classes=len(np.unique(y)))
                    final_preds = np.digitize(np.argmax(oof_preds, axis=1), best_thresholds)
                    print(f"Optimized thresholds: {best_thresholds.round(2)}")
                else:
                    final_preds = np.argmax(oof_preds, axis=1)

                # Final evaluation
                if label_mapping:
                    reverse_mapping = label_mapping[1]
                    final_preds = pd.Series(final_preds).map(reverse_mapping)
                    oof_targets = pd.Series(oof_targets).map(reverse_mapping)

                report = classification_report(oof_targets, final_preds)
                accuracy = accuracy_score(oof_targets, final_preds)
                qwk = cohen_kappa_score(oof_targets, final_preds, weights='quadratic')
                cm = confusion_matrix(oof_targets, final_preds)
                cm_str = np.array2string(cm, separator=', ', prefix='    ')

                print(f"Model: {model_name}")
                print(report)
                print("Accuracy:", accuracy)
                print("QWK (final):", qwk)
                print("Confusion Matrix:\n", cm_str)

                # Compute final log loss
                rf_loss = [log_loss(oof_targets, oof_preds)]
                dt_loss = [log_loss(oof_targets, dt_model.predict_proba(X))]
                loss_df = pd.DataFrame({
                    'XGBoost': [np.mean([loss[-1] for loss in xgb_losses])],
                    'CatBoost': [np.mean([loss[-1] for loss in cat_losses])],
                    'RandomForest': rf_loss,
                    'DecisionTree': dt_loss
                })

                # Plot and save
                labels = [1, 2]
                self._plot_confusion_matrix(cm, model_name, name, labels)
                self._plot_qwk(qwk_scores, model_name, name)
                self._save_results(name, report, accuracy, qwk, cm_str, model_name, loss_df, qwk_scores)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Define datasets
    result_folder =os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Ensembles', 'ensemble_qwk')

    prep_sol_binance_agg5 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg5.csv'))
    prep_sol_binance_agg10 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg10.csv'))
    prep_sol_binance_agg30 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg30.csv'))
    prep_sol_binance_hourly = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly.csv'))
    prep_sol_binance_daily = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily.csv'))
    prep_sol_binance_google = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google.csv'))
    prep_sol_binance_hourly_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly_sentiment.csv'))
    prep_sol_binance_daily_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily_sentiment.csv'))
    prep_sol_binance_google_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google_sentiment.csv'))

    datasets = {
        'agg5': prep_sol_binance_agg5,
        'agg10': prep_sol_binance_agg10,
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
        result_folder=f'{result_folder}/three_classes_qwk',
        datasets=datasets,
        target_column='combined_target',
        weights=[2.0, 2.0, 1.0, 0.5],  # Higher weights for XGBoost/CatBoost
        n_splits=5
    )

    # Run multiclass ensemble
    ensemble_evaluator.ensemble_multiclass(optimize_thresholds=True)

    # Initialize evaluator for two classes
    ensemble_evaluator_two_classes = EnsembleModelEvaluator(
        result_folder=f'{result_folder}/two_classes_qwk',
        datasets=filtered_datasets,
        target_column='combined_target',
        weights=[2.0, 2.0, 1.0, 0.5],
        n_splits=5
    )

    # Run two-class ensemble without SMOTE
    ensemble_evaluator_two_classes.ensemble_two_classes(smote=False, optimize_thresholds=True)

    # Run two-class ensemble with SMOTE
    ensemble_evaluator_two_classes.ensemble_two_classes(smote=True, optimize_thresholds=True)