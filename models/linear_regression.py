"""Linear regression model with ordinary and hierarchical modes."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from .base_model import BaseModel


class LinearRegression(BaseModel):
    """Linear regression model supporting both ordinary and hierarchical modes.
    
    In ordinary mode, uses Bayesian Ridge regression for posterior sampling.
    In hierarchical mode, uses mixed linear models with random effects.
    """
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.model = None
        self.is_hierarchical = False
        self.random_effects = None
        self.feature_names = None
        self.sigma_ = None  # Residual standard deviation
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            random_effects: Optional[list] = None) -> 'LinearRegression':
        """Train the linear regression model.
        
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
            # Hierarchical mode using MixedLM
            self._fit_hierarchical(X, y)
        else:
            # Ordinary mode using BayesianRidge
            self._fit_ordinary(X, y)
            
        return self
    
    def _fit_ordinary(self, X: pd.DataFrame, y: pd.Series):
        """Fit ordinary Bayesian linear regression."""
        self.model = BayesianRidge(compute_score=True)
        self.model.fit(X.values, y.values)
        
        # Estimate residual standard deviation
        y_pred = self.model.predict(X.values)
        residuals = y.values - y_pred
        self.sigma_ = np.std(residuals)
        
    def _fit_hierarchical(self, X: pd.DataFrame, y: pd.Series):
        """Fit hierarchical linear regression with random effects."""
        # Separate fixed effects and random effects
        fixed_cols = [col for col in X.columns if col not in self.random_effects]
        
        if len(fixed_cols) == 0:
            raise ValueError("At least one fixed effect column is required")
        
        # Use the first random effect as grouping variable
        groups = X[self.random_effects[0]]
        
        # Prepare data for MixedLM
        X_fixed = X[fixed_cols]
        
        # Fit mixed linear model
        self.model = MixedLM(y.values, X_fixed.values, groups=groups.values)
        self.model = self.model.fit(method='lbfgs', reml=True)
        
        # Store residual standard deviation
        self.sigma_ = np.sqrt(self.model.scale)
        
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
            predictions = self._predict_hierarchical(X)
        else:
            predictions = self.model.predict(X.values)
            
        if return_std:
            # Return constant uncertainty estimate
            std = np.full(len(predictions), self.sigma_)
            return predictions, std
        
        return predictions
    
    def _predict_hierarchical(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with hierarchical model."""
        fixed_cols = [col for col in X.columns if col not in self.random_effects]
        X_fixed = X[fixed_cols]
        
        # Get fixed effects predictions
        predictions = self.model.predict(X_fixed.values)
        
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
            
        # Get mean predictions
        if return_std := True:
            mean_pred, std_pred = self.predict(X, return_std=True)
        else:
            mean_pred = self.predict(X)
            std_pred = np.full(len(mean_pred), self.sigma_)
        
        # Sample from normal distribution
        samples = np.random.normal(
            loc=mean_pred[np.newaxis, :],
            scale=std_pred[np.newaxis, :],
            size=(n_samples, len(mean_pred))
        )
        
        return samples
