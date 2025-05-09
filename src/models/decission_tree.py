import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import joblib

class DecisionTreeModelEvaluator:
    def __init__(self, result_folder, datasets, target_column='trend_target', drop_columns=None, n_splits=5):
        """
        Initialize the DecisionTreeModelEvaluator with result folder, datasets, and data preparation parameters.

        Args:
            result_folder (str): Path to save results and models
            datasets (dict): Dictionary of dataset names and DataFrames
            target_column (str): Name of the target column
            drop_columns (list): List of columns to drop from features
            n_splits (int): Number of train-test splits for simple_decision_tree
        """
        self.result_folder = result_folder
        self.datasets = datasets
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else [
            'datetime', 'date', 'Upward', 'Downward', 'trend_target', 'combined_target'
        ]
        self.n_splits = n_splits
        os.makedirs(self.result_folder, exist_ok=True)

    def _prepare_data(self, df, use_target_column=True):
        """
        Prepare features and target for modeling.

        Args:
            df (pd.DataFrame): Input DataFrame
            use_target_column (bool): Whether to use specified target column or last column

        Returns:
            tuple: (X, y) features and target
        """
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
        return X, y

    def _save_results(self, name, report, accuracy, recall_macro, f1_macro, scoring, best_params=None):
        """
        Save evaluation results to a file.

        Args:
            name (str): Dataset name
            report (str): Classification report
            accuracy (float): Accuracy score
            recall_macro (float): Macro-averaged recall score
            f1_macro (float): Macro-averaged F1 score
            scoring (str): Scoring method used
            best_params (dict, optional): Best parameters from grid search
        """
        param_str = f"{scoring}_md{best_params['max_depth']}_split{best_params['min_samples_split']}_leaf{best_params['min_samples_leaf']}" if best_params else scoring
        filename = os.path.join(self.result_folder, f"decision_tree_{name}_{param_str}.txt")

        with open(filename, "w") as f:
            f.write(f"=== Decision Tree Results for dataset: {name} (Scoring: {scoring}) ===\n")
            if best_params:
                f.write(f"Best Parameters: {best_params}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
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

    def simple_decision_tree(self, max_depth=5, criterion='gini', class_weight=None, two_class=False):
        """
        Train and evaluate a simple Decision Tree, saving the best-performing model.

        Args:
            max_depth (int): Maximum depth of the tree
            criterion (str): Splitting criterion ('gini' or 'entropy')
            class_weight (str or dict, optional): Weights for classes (e.g., 'balanced')
            two_class (bool): If True, use two-class task (combined_target [1, 2]); else use trend_target
        """
        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Simple Decision Tree on dataset: {name} {'(Two Classes)' if two_class else ''} ===")

            # Handle two-class task
            if two_class:
                if 'combined_target' not in df.columns:
                    print(f"Skipping {name}: 'combined_target' column missing")
                    continue
                df_processed = df[df['combined_target'].isin([1, 2])].copy()
                if df_processed.empty:
                    print(f"Skipping {name}: No data for combined_target [1, 2]")
                    continue
                target_column = 'combined_target'
                use_target_column = False
                scoring = f"simple_two_classes_crit{criterion}_cw{'balanced' if class_weight else 'none'}"
                display_labels = ['Down', 'Up']
            else:
                df_processed = df
                target_column = self.target_column
                use_target_column = True
                scoring = f"simple_crit{criterion}_cw{'balanced' if class_weight else 'none'}"
                display_labels = ['Down', 'Flat', 'Up']

            # Save original target column and set temporary one
            original_target_column = self.target_column
            self.target_column = target_column

            try:
                X, y = self._prepare_data(df_processed, use_target_column=use_target_column)
                best_composite_score = -1
                best_model = None

                for split in range(self.n_splits):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False, random_state=42 + split
                    )

                    tree = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        random_state=42,
                        class_weight=class_weight
                    )
                    tree.fit(X_train, y_train)
                    y_pred = tree.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Split {split + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = tree

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                y_pred = best_model.predict(X_test)
                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                self._save_results(name, report, accuracy, recall_macro, f1_macro, scoring=scoring)
                self._save_best_model(best_model, 'decision_tree', name, param_str=scoring)

                # Save confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
                self._save_confusion_matrix(cm, y_test, accuracy, f"decision_tree_{scoring}", name, display_labels)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

            # Restore original target column
            self.target_column = original_target_column

    def grid_search(self, param_grid=None, scoring='f1_weighted', labels=None, class_weight=None):
        """
        Train and evaluate a Decision Tree with GridSearchCV, saving the best-performing model.

        Args:
            param_grid (dict, optional): Parameter grid for GridSearchCV
            scoring (str): Scoring method ('f1_weighted', 'f1', 'f1_labels', 'f1_two_classes')
            labels (list, optional): Labels for f1_labels scoring (e.g., [-1, 1])
            class_weight (str or dict, optional): Class weights for the classifier
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini']
            }

        # Define scorer and CV based on scoring parameter
        if scoring == 'f1_weighted':
            scorer = make_scorer(f1_score, average='weighted')
            cv = 5
            use_target_column = True
            display_labels = ['Down', 'Flat', 'Up']
        elif scoring == 'f1':
            scorer = make_scorer(f1_score)
            cv = 5
            use_target_column = True
            display_labels = ['Down', 'Flat', 'Up']
        elif scoring == 'f1_labels':
            if labels is None:
                labels = [-1, 1]
            def minority_f1_score(y_true, y_pred):
                return f1_score(y_true, y_pred, labels=labels, average='macro')
            scorer = make_scorer(minority_f1_score)
            cv = TimeSeriesSplit(n_splits=5)
            use_target_column = True
            display_labels = ['Down', 'Up']
        elif scoring == 'f1_two_classes':
            scorer = make_scorer(f1_score)
            cv = TimeSeriesSplit(n_splits=5)
            use_target_column = False
            display_labels = ['Down', 'Up']
        else:
            raise ValueError(f"Unsupported scoring method: {scoring}")

        for name, df in self.datasets.items():
            print(f"\n=== Evaluating Grid Search Decision Tree on dataset: {name} (Scoring: {scoring}) ===")
            try:
                X, y = self._prepare_data(df, use_target_column=use_target_column)
                best_composite_score = -1
                best_model = None
                best_params = None

                # Perform cross-validation manually to select best model
                tscv = cv if isinstance(cv, TimeSeriesSplit) else TimeSeriesSplit(n_splits=5)
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    grid_search = GridSearchCV(
                        estimator=DecisionTreeClassifier(random_state=42, class_weight=class_weight),
                        param_grid=param_grid,
                        cv=5 if isinstance(cv, int) else TimeSeriesSplit(n_splits=5),
                        scoring=scorer,
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    fold_params = grid_search.best_params_

                    tree = DecisionTreeClassifier(
                        criterion=fold_params['criterion'],
                        max_depth=fold_params['max_depth'],
                        min_samples_split=fold_params['min_samples_split'],
                        min_samples_leaf=fold_params['min_samples_leaf'],
                        random_state=42,
                        class_weight=class_weight
                    )
                    tree.fit(X_train, y_train)
                    y_pred = tree.predict(X_test)

                    # Compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    composite_score = self._compute_composite_score(accuracy, recall_macro, f1_macro)

                    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}, Composite Score: {composite_score:.4f}")

                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_model = tree
                        best_params = fold_params

                # Evaluate and save best model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                report = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

                print(f"Best Parameters: {best_params}")
                print(report)
                print(f"Best Model Accuracy: {accuracy:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                self._save_results(name, report, accuracy, recall_macro, f1_macro, scoring, best_params)
                self._save_best_model(best_model, 'decision_tree', name, param_str=scoring)

                # Save confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
                self._save_confusion_matrix(cm, y_test, accuracy, f"decision_tree_{scoring}", name, display_labels)

            except Exception as e:
                print(f"Error processing dataset {name}: {str(e)}")

if __name__ == "__main__":
    # Define result folder and dataset paths using relative paths
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'decision_tree')
    dataset_base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets')

    # Load datasets
    datasets = {
        # 'no_agg': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance.csv')),
        'agg5': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg5.csv')),
        'agg10': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg10.csv')),
        'agg30': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_agg30.csv')),
        'hourly': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_hourly.csv')),
        'daily': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_daily.csv')),
        'google_combined': pd.read_csv(os.path.join(dataset_base_path, 'prep_sol_binance_google.csv')),
    }

    # Filter datasets for two classes (1 and 2)
    filtered_datasets = {
        name: df[df['combined_target'].isin([1, 2])].copy() for name, df in datasets.items()
    }

    # Custom parameter grid
    custom_param_grid = {
        'max_depth': [3, 7, 12],
        'min_samples_split': [2, 8],
        'min_samples_leaf': [1, 3],
        'criterion': ['gini']
    }

    # Initialize evaluator for multiclass
    evaluator = DecisionTreeModelEvaluator(
        result_folder=os.path.join(result_folder, 'multiclass'),
        datasets=datasets,
        target_column='trend_target',
        n_splits=5
    )

    # Run simple decision tree for three-class task
    print("\n=== Running simple_decision_tree (Three Classes) ===")
    evaluator.simple_decision_tree(max_depth=5, criterion='gini', class_weight=None, two_class=False)

    # Run simple decision tree for two-class task
    print("\n=== Running simple_decision_tree (Two Classes) ===")
    two_class_evaluator = DecisionTreeModelEvaluator(
        result_folder=os.path.join(result_folder, 'two_classes'),
        datasets=filtered_datasets,
        target_column='combined_target',
        n_splits=5
    )
    two_class_evaluator.simple_decision_tree(max_depth=10, criterion='gini', class_weight=None, two_class=True)
    two_class_evaluator.simple_decision_tree(max_depth=10, criterion='entropy', class_weight=None, two_class=True)
    two_class_evaluator.simple_decision_tree(max_depth=10, criterion='entropy', class_weight='balanced', two_class=True)

    # Run grid search with different scoring options
    scoring_options = [
        {'scoring': 'f1', 'labels': None, 'class_weight': None},
        {'scoring': 'f1_weighted', 'labels': None, 'class_weight': None},
        {'scoring': 'f1_labels', 'labels': [-1, 1], 'class_weight': 'balanced'},
        {'scoring': 'f1_two_classes', 'labels': None, 'class_weight': None},
        {'scoring': 'f1_two_classes', 'labels': None, 'class_weight': 'balanced'}
    ]

    for opt in scoring_options:
        scoring = opt['scoring']
        print(f"\n=== Running grid_search with scoring: {scoring} ===")
        datasets_to_use = filtered_datasets if scoring == 'f1_two_classes' else datasets
        target_column = 'combined_target' if scoring == 'f1_two_classes' else 'trend_target'
        temp_evaluator = DecisionTreeModelEvaluator(
            result_folder=os.path.join(result_folder, 'multiclass' if scoring != 'f1_two_classes' else 'two_classes'),
            datasets=datasets_to_use,
            target_column=target_column,
            n_splits=5
        )
        temp_evaluator.grid_search(
            param_grid=custom_param_grid,
            scoring=scoring,
            labels=opt['labels'],
            class_weight=opt['class_weight']
        )