# -*- coding: utf-8 -*-
"""

██████╗  █████╗ ████████╗ █████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

██████╗ ██████╗ ███████╗██████╗ ██████╗  █████╗  █████╗ ███████╗ ██████╗ ██████╗██╗███╗  ██╗ ██████╗ 
██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║████╗ ██║██╔════╝ 
██████╔╝██████╔╝█████╗  ██████╔╝██████╔╝██║  ██║██║  ╚═╝█████╗  ╚█████╗ ╚█████╗ ██║██╔██╗██║██║  ██╗ 
██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ ██╔══██╗██║  ██║██║  ██╗██╔══╝   ╚═══██╗ ╚═══██╗██║██║╚████║██║  ╚██╗
██║     ██║  ██║███████╗██║     ██║  ██║╚█████╔╝╚█████╔╝███████╗██████╔╝██████╔╝██║██║ ╚███║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚════╝ ╚══════╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚══╝ ╚═════╝ 
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocessor:
    """
    Handles the preprocessing phases (P1-P12) as described in the scomp-link schema.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()

    def clean_data(self, remove_outliers: bool = True, outlier_threshold: float = 3.0):
        """
        P4: Pulizia dei dati (rimozione di errori formali e logici e di outlier).
        """
        print("P4: Cleaning data...")
        # Basic cleaning: drop duplicates
        self.df = self.df.drop_duplicates()
        
        if remove_outliers:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < outlier_threshold]
        
        print(f"Data cleaned. Current shape: {self.df.shape}")
        return self.df

    def integrate_data(self, other_df: pd.DataFrame, on: str, how: str = 'left'):
        """
        P5: Integrazione dei dati (combinazione di fonti diverse) RECORD LINKAGE.
        """
        print("P5: Integrating data...")
        self.df = pd.merge(self.df, other_df, on=on, how=how)
        print(f"Data integrated. Current shape: {self.df.shape}")
        return self.df

    def transform_data(self):
        """
        P7: Trasformazione dei dati (manipolazione dei dati in forme adeguate al data mining).
        """
        print("P7: Transforming data...")
        # Placeholder for scaling, encoding, etc.
        # In a real scenario, this would involve standardizing numeric features
        # and encoding categorical features.
        return self.df

    def feature_selection(self, target_col: str, n_features: Optional[int] = None):
        """
        P10: Feature Selection.
        """
        print("P10: Selecting features...")
        # Simple correlation-based selection for now
        if target_col in self.df.columns:
            numeric_df = self.df.select_dtypes(include=[np.number])
            correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations.index[1:n_features+1 if n_features else None].tolist()
            print(f"Top features selected: {top_features}")
            return top_features
        return self.df.columns.tolist()

    def run_eda(self):
        """
        P11: EDA: Presentazione della conoscenza.
        """
        print("P11: Running Exploratory Data Analysis...")
        summary = {
            "shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "dtypes": self.df.dtypes.to_dict(),
            "description": self.df.describe().to_dict()
        }
        return summary

    def prepare_datasets(self, target_col: str, test_size: float = 0.2):
        """
        P12: Preparazione set di dati.
        """
        print("P12: Preparing datasets...")
        from sklearn.model_selection import train_test_split
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
