"""
Stock Trend Predictor Package

A PyTorch-based package for stock price prediction using LSTM neural networks.
"""

from .model import StockPredictor
from .scaler import MinMaxScaler

__all__ = ['StockPredictor', 'MinMaxScaler']
__version__ = '1.0.0'

