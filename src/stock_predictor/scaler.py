import torch
import numpy as np


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
        
    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X):
        if self.data_min_ is None or self.scale_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        is_tensor = isinstance(X, torch.Tensor)
        if is_tensor:
            X_np = X.numpy()
        elif isinstance(X, np.ndarray):
            X_np = X.copy()
        else:
            X_np = np.array(X)
        
        original_shape = X_np.shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        X_scaled = X_np * self.scale_ + self.min_
        
        if len(original_shape) == 1:
            X_scaled = X_scaled.reshape(-1)
        
        if is_tensor:
            return torch.from_numpy(X_scaled).float()
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.data_min_ is None or self.scale_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        is_tensor = isinstance(X, torch.Tensor)
        if is_tensor:
            X_np = X.numpy()
        elif isinstance(X, np.ndarray):
            X_np = X.copy()
        else:
            X_np = np.array(X)
        
        original_shape = X_np.shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        X_original = (X_np - self.min_) / self.scale_
        
        if len(original_shape) == 1:
            X_original = X_original.reshape(-1)
        
        if is_tensor:
            return torch.from_numpy(X_original).float()
        return X_original

