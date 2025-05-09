import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class RandomForestModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='trend_target', drop_columns=None):
        """
        Initialize the RandomForestModelEvaluator with result folder, datasets, and data preparation parameters.

        Args:
            result_folder (str): Base path to save results
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
        """
        self.base_result_folder = result_folder
        self.datasets = datasets
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'trend_label', 'combined_target'
        ]
        os.makedirs(self.base_result_folder, exist_ok=True)
        os.makedirs(f"{self.base_result_folder}/two_classes", exist_ok=True)

    def _prepare_data(self, df, use_target_column=True):
        """
        Prepare features and target for modeling, handling NaN/infinite values.

        Args:
            df (pd.DataFrame): Input DataFrame
            use_target_column (bool): Whether to use specified target column or last column

        Returns:
            tuple: (X, y) features and target
        """
        if df.isna().any().any() or np.isinf(df.select_dtypes(include=np.number)).any().any():
            print("Warning: Data contains NaN or infinite values. Filling NaNs with 0.")
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
        X = df.drop(columns=self.drop_columns, errors='ignore')
        if use_target_column and self.target_column in df.columns:
            y = df[self.target_column]
        else:
            y = df.iloc[:, -1]
        return X, y

    def _save_results(self, name, report, accuracy, scoring, best_params):
        """
        Save evaluation results to a file.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            scoring (str): Scoring method used
            best_params (dict): Best parameters from grid search
        """
        param_str = (
            f"n{best_params['n_estimators']}_md{best_params['max_depth']}_"
            f"split{best_params['min_samples_split']}_leaf{best_params['min_samples_leaf']}_"
            f"{best_params['criterion']}"
        )
        result_folder = f"{self.base_result_folder}/two_classes" if scoring == 'f1_two_classes' else self.base_result_folder
        result_filename = f"{result_folder}/random_forest_{name}_{param_str}.txt"

        with open(result_filename, "w") as f:
            f.write(f"=== Random Forest Results for dataset: {name} ===\n")
            f.write(f"Best Parameters ({'F1 for classes 1 & 2' if scoring == 'f1_two_classes' else 'F1 on -1 & 1'}): {best_params}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")

        print(f"Saved results to: {result_filename}")

    def evaluate_random_forest(self, two_class=False):
        """
        Train and evaluate a Random Forest with GridSearchCV.

        Args:
            two_class (bool): If True, use two-class task (combined_target [1, 2]); else use trend_target
        """
        # Define parameter grid based on task
        if two_class:
            rf_param_grid = {
                'n_estimators': [100],
                # 'max_depth': [5, 10],
                'max_depth': [8, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1],
                'criterion': ['gini']
            }
            scoring = 'f1_two_classes'
            display_labels = ['Down', 'Up']
        else:
            rf_param_grid = {
                'n_estimators': [100, 200],
                # 'max_depth': [5, 10, None],
                'max_depth': [8, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
            scoring = 'f1'
            display_labels = ['Down', 'Flat', 'Up']

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Random Forest on dataset: {name} {'(Two Classes)' if two_class else ''} ===")

            # Handle two-class task
            if two_class:
                if 'combined_target' not in df.columns:
                    print(f"Skipping {name}: 'combined_target' column missing")
                    continue
                df_processed = df[df['combined_target'].isin([1, 2])].copy()
                if df_processed.empty:
                    print(f"Skipping {name}: No data for combined_target [1, 2]")
                    continue
                use_target_column = False  # Use last column (combined_target)
            else:
                df_processed = df
                use_target_column = True

            try:
                X, y = self._prepare_data(df_processed, use_target_column=use_target_column)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Check class distribution
                unique_classes = np.unique(y_train)
                print(f"Classes in training data: {unique_classes}")
                if two_class and not all(cls in unique_classes for cls in [1, 2]):
                    print(f"Warning: Not all classes [1, 2] present in training data for {name}. Skipping.")
                    continue

                # Grid Search
                scorer = make_scorer(f1_score, average='weighted' if not two_class else 'binary', pos_label=2 if two_class else None)
                rf_grid_search = GridSearchCV(
                    estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                    param_grid=rf_param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring=scorer,
                    n_jobs=-1,
                    error_score=np.nan
                )

                print("Starting Grid Search...")
                rf_grid_search.fit(X_train, y_train)
                best_params = rf_grid_search.best_params_
                print("Best Parameters:", best_params)

                # Train best model
                rf_model = RandomForestClassifier(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    min_samples_split=best_params['min_samples_split'],
                    min_samples_leaf=best_params['min_samples_leaf'],
                    criterion=best_params['criterion'],
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = rf_model.predict(X_test)
                report = classification_report(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)

                print("\nClassification Report:")
                print(report)
                print(f"Accuracy: {accuracy:.4f}")

                # Save results
                self._save_results(name, report, accuracy, scoring, best_params)

                # Confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                disp.plot(cmap='Blues', values_format='d')
                plt.title(f'Accuracy: {accuracy:.9f}')

                # Save the confusion matrix plot
                param_str = (
                    f"n{best_params['n_estimators']}_md{best_params['max_depth']}_"
                    f"split{best_params['min_samples_split']}_leaf{best_params['min_samples_leaf']}_"
                    f"{best_params['criterion']}"
                )
                result_folder = f"{self.base_result_folder}/two_classes" if two_class else self.base_result_folder
                cm_filename = f"{result_folder}/confusion_matrix_{name}_{param_str}.png"
                plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
                plt.close()  # Close the plot to free memory
                print(f"Saved confusion matrix plot to: {cm_filename}")

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Define datasets
    result_folder =os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'random_forest')

    

    # prep_sol_binance_agg5 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg5.csv'))
    prep_sol_binance_agg10 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg10.csv'))
    prep_sol_binance_agg30 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg30.csv'))
    prep_sol_binance_hourly = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly.csv'))
    prep_sol_binance_daily = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily.csv'))
    prep_sol_binance_google = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google.csv'))
    prep_sol_binance_hourly_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_hourly_sentiment.csv'))
    prep_sol_binance_daily_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_daily_sentiment.csv'))
    prep_sol_binance_google_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_google_sentiment.csv'))
    # prep_sol_binance = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets',  'prep_sol_binance.csv'))
    prep_sol_binance_agg30_sentiment = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', 'prep_sol_binance_agg30_sentiment.csv'))


    datasets = {
        # 'minutely': prep_sol_binance,
        # 'agg5': prep_sol_binance_agg5,
        # 'agg10': prep_sol_binance_agg10,
        'agg30': prep_sol_binance_agg30,
        'hourly': prep_sol_binance_hourly,
        'daily': prep_sol_binance_daily,
        'google_combined': prep_sol_binance_google,
        'agg30_sentiment': prep_sol_binance_agg30_sentiment,
        'hourly_sentiment': prep_sol_binance_hourly_sentiment,
        'daily_sentiment': prep_sol_binance_daily_sentiment,
        'google_sentiment': prep_sol_binance_google_sentiment,
    }

    # Filter datasets for two classes (1 and 2) for two_class=True
    filtered_datasets = {
        name: df[df['combined_target'].isin([1, 2])].copy() for name, df in datasets.items()
    }

    # Initialize evaluator
    evaluator = RandomForestModelEvaluator(
        result_folder=f'{result_folder}',
        datasets=datasets,
        target_column='trend_target'
    )

    # Run Random Forest for three-class task (trend_target)
    print("\n=== Running Random Forest (Three Classes) ===")
    evaluator.evaluate_random_forest(two_class=False)

    # Run Random Forest for two-class task (combined_target [1, 2])
    print("\n=== Running Random Forest (Two Classes) ===")
    evaluator.evaluate_random_forest(two_class=True)