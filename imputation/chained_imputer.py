"""Chained multiple imputation using regression models."""

import numpy as np
import pandas as pd
from typing import Optional, List
import warnings


class ChainedImputer:
    """Chained multiple imputation using any regression model.
    
    Implements the MICE (Multivariate Imputation by Chained Equations) algorithm
    with support for hierarchical models.
    """
    
    def __init__(self, base_model, random_effects: Optional[List[str]] = None,
                 n_imputations: int = 5, max_iter: int = 10, 
                 convergence_threshold: float = 0.01, random_state: Optional[int] = None):
        """Initialize the chained imputer.
        
        Args:
            base_model: Regression model instance (must implement BaseModel interface)
            random_effects: List of column names for random effects (hierarchical mode)
            n_imputations: Number of imputed datasets to generate
            max_iter: Maximum number of chained iterations
            convergence_threshold: Threshold for convergence check
            random_state: Random seed for reproducibility
        """
        self.base_model = base_model
        self.random_effects = random_effects
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_transform(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """Perform multiple imputation on the dataset.
        
        Args:
            X: DataFrame with missing values
        
        Returns:
            imputed_datasets: List of imputed DataFrames
        """
        # Check for missing values
        if not X.isnull().any().any():
            warnings.warn("No missing values found in the dataset")
            return [X.copy() for _ in range(self.n_imputations)]
        
        # Identify columns with missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        
        # Generate multiple imputed datasets
        imputed_datasets = []
        for imp_idx in range(self.n_imputations):
            if self.random_state is not None:
                np.random.seed(self.random_state + imp_idx)
            
            imputed_data = self._single_imputation(X, missing_cols)
            imputed_datasets.append(imputed_data)
        
        return imputed_datasets
    
    def _single_imputation(self, X: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
        """Perform a single imputation using chained equations.
        
        Args:
            X: DataFrame with missing values
            missing_cols: List of columns with missing values
        
        Returns:
            imputed_data: Single imputed DataFrame
        """
        # Initialize with mean imputation
        X_imputed = X.copy()
        for col in missing_cols:
            if X_imputed[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X_imputed[col].fillna(X_imputed[col].mean(), inplace=True)
            else:
                X_imputed[col].fillna(X_imputed[col].mode()[0], inplace=True)
        
        # Chained iterations
        prev_values = {}
        for col in missing_cols:
            prev_values[col] = X_imputed[col].copy()
        
        for iteration in range(self.max_iter):
            # Impute each variable with missing values
            for col in missing_cols:
                X_imputed = self._impute_column(X, X_imputed, col)
            
            # Check convergence
            if iteration > 0:
                max_change = 0
                for col in missing_cols:
                    change = np.abs(X_imputed[col] - prev_values[col]).mean()
                    max_change = max(max_change, change)
                
                if max_change < self.convergence_threshold:
                    break
            
            # Update previous values
            for col in missing_cols:
                prev_values[col] = X_imputed[col].copy()
        
        return X_imputed
    
    def _impute_column(self, X_original: pd.DataFrame, X_imputed: pd.DataFrame, 
                       target_col: str) -> pd.DataFrame:
        """Impute a single column using regression.
        
        Args:
            X_original: Original DataFrame with missing values
            X_imputed: Current imputed DataFrame
            target_col: Column to impute
        
        Returns:
            X_imputed: Updated DataFrame with imputed values for target_col
        """
        # Identify rows with missing values in target column
        missing_mask = X_original[target_col].isnull()
        
        if not missing_mask.any():
            return X_imputed
        
        # Prepare training data (rows without missing values in target)
        train_mask = ~missing_mask
        
        # Select predictor columns (all except target)
        predictor_cols = [col for col in X_imputed.columns if col != target_col]
        
        X_train = X_imputed.loc[train_mask, predictor_cols]
        y_train = X_imputed.loc[train_mask, target_col]
        
        X_missing = X_imputed.loc[missing_mask, predictor_cols]
        
        # Clone the base model for this column
        from copy import deepcopy
        model = deepcopy(self.base_model)
        
        # Fit model
        try:
            model.fit(X_train, y_train, random_effects=self.random_effects)
            
            # Sample from posterior to impute missing values
            imputed_values = model.sample(X_missing, n_samples=1)[0]
            
            # Update imputed data
            X_imputed.loc[missing_mask, target_col] = imputed_values
            
        except Exception as e:
            warnings.warn(f"Failed to impute column {target_col}: {str(e)}")
        
        return X_imputed
