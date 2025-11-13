"""Base model interface for regression models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


class BaseModel(ABC):
    """Abstract base class for regression models used in imputation.
    
    All regression models must implement fit, predict, and sample methods
    to be compatible with the ChainedImputer.
    """
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            random_effects: Optional[list] = None) -> 'BaseModel':
        """Train the model on the provided data.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target variable, shape (n_samples,)
            random_effects: List of column names for random effects (optional).
                           Used for hierarchical models to specify grouping variables.
        
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions on new data.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            return_std: If True, return prediction standard deviation
        
        Returns:
            predictions: Predicted values, shape (n_samples,)
            std: Standard deviation of predictions (if return_std=True), shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def sample(self, X: Union[pd.DataFrame, np.ndarray],
               n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior predictive distribution.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            n_samples: Number of samples to draw from the posterior
        
        Returns:
            samples: Samples from posterior distribution, shape (n_samples, X.shape[0])
        """
        pass
