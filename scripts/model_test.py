import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder, WOEEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sdv.single_table import CTGANSynthesizer
from figen_class import FiGen  # Assuming FiGen is imported from figen_class

class ImbalancedCompare:
    
    def __init__(self, data_path: str, categorical_columns: List[str], generation_rate: float):
        self.data_path = data_path
        self.categorical_columns = categorical_columns
        self.generation_rate = generation_rate

    def load_dataset(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the dataset from the given file path.

        Args:
            file_path (str): Path to the CSV dataset file.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        """
        data = pd.read_csv(file_path)
        X = data.drop(columns=['target'])
        y = data['target']
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = self.generation_rate, random_state: int = 1004) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into train and test sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for random number generator.

        Returns:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target.
            y_test (pd.Series): Testing target.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def under_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply random under-sampling to balance the classes.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            X_resampled (pd.DataFrame): Resampled features.
            y_resampled (pd.Series): Resampled target.
        """
        rus = RandomUnderSampler()
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    # ... (Other functions)

    def generate_synthetic_data(self, X_resampled: pd.DataFrame, y_resampled: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic data using FiGen.

        Args:
            X_resampled (pd.DataFrame): Resampled features.
            y_resampled (pd.Series): Resampled target.

        Returns:
            synthetic_X (pd.DataFrame): Synthetic features.
            synthetic_y (pd.Series): Synthetic target.
        """
        figen = FiGen(ratio=0.3, categorical_columns=self.categorical_columns)
        synthetic_X, synthetic_y = figen.fit(X_resampled, y_resampled, X_train, y_train)
        return synthetic_X, synthetic_y
    
    
    def fit(self, data: str):
        """
        Fit the model and evaluate various techniques.

        Args:
            data (str): Path to the dataset file.
        """
        # Load and preprocess data
        X, y = self.load_dataset(data)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
    
        # Create LGBM model
        lgbm_model = LGBMClassifier()
    
        # Under Sampling
        X_resampled, y_resampled = self.under_sampling(X_train, y_train)
        roc_auc_us, f1_us, report_us = self.evaluate_model(lgbm_model, X_resampled, y_resampled, X_test, y_test)
    
        # Ensemble with Under Sampling
        brf = self.ensemble_under_sampling(X_train, y_train)
        roc_auc_brf, f1_brf, report_brf = self.evaluate_model(brf, X_train, y_train, X_test, y_test)
    
        # Generate Synthetic Data
        synthetic_X, synthetic_y = self.generate_synthetic_data(X_resampled, y_resampled)
        roc_auc_syn, f1_syn, report_syn = self.evaluate_model(lgbm_model, synthetic_X, synthetic_y, X_test, y_test)
    
         # Encode Data
        X_ohe, X_mean_encoded, X_woe_encoded = self.encode_data(X_train, y_train)
        roc_auc_ohe, f1_ohe, report_ohe = self.evaluate_model(lgbm_model, X_ohe, y_train, X_test, y_test)
        roc_auc_mean_encoded, f1_mean_encoded, report_mean_encoded = self.evaluate_model(lgbm_model, X_mean_encoded, y_train, X_test, y_test)
        roc_auc_woe_encoded, f1_woe_encoded, report_woe_encoded = self.evaluate_model(lgbm_model, X_woe_encoded, y_train, X_test, y_test)
    
        # Print Results
        print("Under Sampling Results:")
        print("ROC AUC:", roc_auc_us)
        print("F1 Score:", f1_us)
        print(report_us)
    
        print("\nBalanced Random Forest Results:")
        print("ROC AUC:", roc_auc_brf)
        print("F1 Score:", f1_brf)
        print(report_brf)
    
        print("\nSynthetic Data (FiGen) Results:")
        print("ROC AUC:", roc_auc_syn)
        print("F1 Score:", f1_syn)
        print(report_syn)
    
        print("\nOne-Hot Encoding Results:")
        print("ROC AUC:", roc_auc_ohe)
        print("F1 Score:", f1_ohe)
        print(report_ohe)
    
        print("\nMean Encoding Results:")
        print("ROC AUC:", roc_auc_mean_encoded)
        print("F1 Score:", f1_mean_encoded)
        print(report_mean_encoded)
    
        print("\nWOE Encoding Results:")
        print("ROC AUC:", roc_auc_woe_encoded)
        print("F1 Score:", f1_woe_encoded)
        print(report_woe_encoded)
    

