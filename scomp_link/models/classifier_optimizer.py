# -*- coding: utf-8 -*-
"""

 █████╗ ██╗      █████╗  ██████╗ ██████╗██╗███████╗██╗███████╗██████╗ 
██╔══██╗██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗
██║  ╚═╝██║     ███████║╚█████╗ ╚█████╗ ██║█████╗  ██║█████╗  ██████╔╝
██║  ██╗██║     ██╔══██║ ╚═══██╗ ╚═══██╗██║██╔══╝  ██║██╔══╝  ██╔══██╗
╚█████╔╝███████╗██║  ██║██████╔╝██████╔╝██║██║     ██║███████╗██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝

 █████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚════██║██╔════╝██╔══██╗
██║  ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔═╝█████╗  ██████╔╝
██║  ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║██╔══╝  ██╔══╝  ██╔══██╗
╚█████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
 ╚════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .regressor_optimizer import Boruta

class ClassifierOptimizer:
    """
    ClassifierOptimizer is a class designed to optimize classification models by performing feature selection,
    hyperparameter tuning, and evaluating multiple classification algorithms.
    """
    def __init__(self, df, y_col, x_cols, models_to_test, select_features=False):
        if select_features:
            # Boruta for classification needs a classifier, usually RandomForestClassifier
            rf = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight='balanced')
            selector = Boruta(rf, n_estimators='auto', random_state=42)
            
            # Basic preprocessing for Boruta (it needs numeric input)
            X_temp = df[x_cols].copy()
            for col in X_temp.select_dtypes(include=['object']).columns:
                X_temp[col] = X_temp[col].astype('category').cat.codes
            
            selector.fit(X_temp.values, df[y_col].values)
            self.support_ = selector.support_
            selected_x_cols = [x_cols[i] for i, s in enumerate(self.support_) if s]
            self.X = df[selected_x_cols]
            print(f"Selected features: {selected_x_cols}")
        else:
            self.X = df[x_cols]
        
        self.y = df[y_col]
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.binary_cols = self.X.select_dtypes(include=['bool']).columns.tolist()
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        self.binary_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)
        self.numeric_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', self.categorical_transformer, self.categorical_cols),
                ('binary', self.binary_transformer, self.binary_cols),
                ('numeric', self.numeric_transformer, self.numeric_cols)
            ]
        )

        self.models_to_test = models_to_test
        self.model_results = {}
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=15121)

    def select_hyperparameters(self, classifier, params_grid):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', classifier)
        ])
        
        grid_search = GridSearchCV(
            pipeline, 
            {'classifier__' + k: v for k, v in params_grid.items()}, 
            cv=self.cv_strategy, 
            scoring='accuracy', 
            verbose=True
        )
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def test_models_classification(self):
        for model_name, model_data in self.models_to_test.items():
            print(f"\n\t... Testing {model_name}:\n\t", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n\t... Optimizing HyperParameters ...\n\t")
            model = model_data['model']
            params_grid = model_data['params_grid']

            best_classifier, best_params = self.select_hyperparameters(model, params_grid)
            print(f"\n\t", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"\n\t... Training the Final Model ...\n\t")

            best_classifier.fit(self.X_train, self.y_train)
            y_pred = best_classifier.predict(self.X_test)
            y_proba = None
            if hasattr(best_classifier, "predict_proba"):
                y_proba = best_classifier.predict_proba(self.X_test)

            self.model_results[model_name] = {
                'Model': best_classifier,
                'Params': best_params,
                'Fitted_Test': y_pred,
                'Proba_Test': y_proba,
                'True_Test': self.y_test,
                'Report': classification_report(self.y_test, y_pred),
                'Confusion_Matrix': confusion_matrix(self.y_test, y_pred)
            }
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"\n\t... Finished ...\n\t")

    def print_results(self):
        for model_name, results in self.model_results.items():
            print(f"--- {model_name} ---")
            print(f"Best Params: {results['Params']}")
            print("Classification Report:")
            print(results['Report'])
            print("Confusion Matrix:")
            print(results['Confusion_Matrix'])
            print("\n")
