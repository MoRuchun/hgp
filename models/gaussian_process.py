"""Gaussian process regression model with ordinary and hierarchical modes."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from .base_model import BaseModel


class GaussianProcessRegression(BaseModel):
    """Gaussian process regression supporting ordinary and hierarchical modes.
    
    In ordinary mode, uses standard GP regression.
    In hierarchical mode, adds group-specific random effects to the kernel.
    """
    
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=10):
        """Initialize the Gaussian process model.
        
        Args:
            kernel: Kernel function (default: RBF + WhiteKernel)
            alpha: Noise level regularization
            n_restarts_optimizer: Number of restarts for hyperparameter optimization
        """
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = None
        self.is_hierarchical = False
        self.random_effects = None
        self.feature_names = None
        self.group_effects = {}  # Store group-specific offsets
        self.groups_train = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            random_effects: Optional[list] = None) -> 'GaussianProcessRegression':
        """Train the Gaussian process model.
        
        Args:
            X: Input features
            y: Target variable
            random_effects: List of column names for random effects (hierarchical mode)
        
        Returns:
            self: Fitted model
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='y')
            
        self.feature_names = X.columns.tolist()
        
        # Determine mode
        self.is_hierarchical = random_effects is not None and len(random_effects) > 0
        self.random_effects = random_effects
        
        if self.is_hierarchical:
            self._fit_hierarchical(X, y)
        else:
            self._fit_ordinary(X, y)
            
        return self
    
    def _fit_ordinary(self, X: pd.DataFrame, y: pd.Series):
        """Fit ordinary Gaussian process regression."""
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True
        )
        self.model.fit(X.values, y.values)
        
    def _fit_hierarchical(self, X: pd.DataFrame, y: pd.Series):
        """Fit hierarchical Gaussian process with group-specific random effects."""
        # Separate fixed effects and random effects
        fixed_cols = [col for col in X.columns if col not in self.random_effects]
        
        if len(fixed_cols) == 0:
            raise ValueError("At least one fixed effect column is required")
        
        # Get grouping variable
        group_col = self.random_effects[0]
        groups = X[group_col].values
        self.groups_train = groups
        
        # Calculate group-specific means (random intercepts)
        unique_groups = np.unique(groups)
        overall_mean = y.mean()
        
        for group in unique_groups:
            group_mask = groups == group
            group_mean = y[group_mask].mean()
            self.group_effects[group] = group_mean - overall_mean
        
        # Center y by removing group effects
        y_centered = y.copy()
        for group in unique_groups:
            group_mask = groups == group
            y_centered[group_mask] -= self.group_effects[group]
        
        # Fit GP on centered data with fixed effects only
        X_fixed = X[fixed_cols]
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True
        )
        self.model.fit(X_fixed.values, y_centered.values)
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions.
        
        Args:
            X: Input features
            return_std: If True, return prediction standard deviation
        
        Returns:
            predictions: Predicted values
            std: Standard deviation (if return_std=True)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        if self.is_hierarchical:
            return self._predict_hierarchical(X, return_std)
        else:
            return self.model.predict(X.values, return_std=return_std)
    
    def _predict_hierarchical(self, X: pd.DataFrame, return_std: bool = False):
        """Make predictions with hierarchical model."""
        fixed_cols = [col for col in X.columns if col not in self.random_effects]
        X_fixed = X[fixed_cols]
        
        # Get base predictions from GP
        if return_std:
            base_pred, std = self.model.predict(X_fixed.values, return_std=True)
        else:
            base_pred = self.model.predict(X_fixed.values, return_std=False)
        
        # Add group-specific effects
        group_col = self.random_effects[0]
        groups = X[group_col].values
        
        predictions = base_pred.copy()
        for i, group in enumerate(groups):
            if group in self.group_effects:
                # Known group: add learned offset
                predictions[i] += self.group_effects[group]
            # Unknown group: use population mean (no offset)
        
        if return_std:
            return predictions, std
        return predictions
    
    def sample(self, X: Union[pd.DataFrame, np.ndarray],
               n_samples: int = 1) -> np.ndarray:
        """Sample from posterior predictive distribution.
        
        Args:
            X: Input features
            n_samples: Number of samples to draw
        
        Returns:
            samples: Samples from posterior, shape (n_samples, n_observations)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self.predict(X, return_std=True)
        
        # Sample from Gaussian distribution
        samples = np.random.normal(
            loc=mean_pred[np.newaxis, :],
            scale=std_pred[np.newaxis, :],
            size=(n_samples, len(mean_pred))
        )
        
        return samples
