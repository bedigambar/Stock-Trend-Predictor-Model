# Stock Trend Predictor

A PyTorch-based machine learning project for predicting stock price trends using historical data from Yahoo Finance. This repository includes training scripts, Jupyter notebooks for model development, and a Streamlit web app for interactive visualization and prediction.

## Features

- **Data Collection**: Downloads historical stock data using `yfinance` for any ticker symbol.
- **Data Preprocessing**: Handles MultiIndex columns from yfinance, computes moving averages, percentage changes, and scales data using PyTorch-only MinMaxScaler.
- **Model Training**: Builds and trains an LSTM neural network using PyTorch to predict closing prices based on past 100 days of data.
- **Evaluation**: Computes metrics like MAE, RMSE, MAPE, and R2; plots actual vs predicted prices.
- **Web App**: Streamlit interface for uploading models, selecting stocks, and visualizing predictions.
- **Model Persistence**: Saves trained models as `.pth` files for reuse.
- **Pure PyTorch**: No scikit-learn dependencies - all preprocessing and scaling done with PyTorch.

## Project Structure

```
Stock-Trend-Predictor/
├── main.py                 # Streamlit web app for prediction and visualization
├── src/
│   └── stock_predictor/    # Main package
│       ├── __init__.py     # Package initialization
│       ├── model.py        # PyTorch LSTM model architecture
│       └── scaler.py       # PyTorch-based MinMaxScaler implementation
├── scripts/
│   └── train.py            # Training script for PyTorch model
├── notebooks/
│   └── stock.ipynb         # Jupyter notebook for data exploration, training, and evaluation
├── model.pth               # Pre-trained PyTorch model (generated after training, not included in repo)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
└── LICENSE                 # License file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bedigambar/Stock-Trend-Predictor-Model
   cd Stock-Trend-Predictor-Model
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

Train a new PyTorch model using the training script:

```bash
python scripts/train.py
```

This will:
- Download 20 years of GOOG stock data
- Preprocess and scale the data using PyTorch MinMaxScaler
- Train an LSTM model for 20 epochs
- Save the model as `model.pth`

To customize training:
- Edit `stock_symbol` in `scripts/train.py` to change the stock
- Adjust `epochs` to change training duration
- Modify `seq_len` to change sequence length (default: 100)

### Running the Streamlit App

The Streamlit app allows you to load stock data, upload or auto-detect a trained model, and run predictions.

```bash
streamlit run main.py
```

- Enter a stock symbol (e.g., "GOOG").
- Adjust years of data and prediction settings.
- Upload a `.pth` model file or let it auto-detect `model.pth`.
- View plots and metrics.

### Running the Jupyter Notebook

Use the notebook for development, training, and evaluation.

1. Open `notebooks/stock.ipynb` in Jupyter Lab or VS Code.
2. Run cells in order:
   - Import libraries and download data.
   - Preprocess data (normalize columns, compute features).
   - Prepare sequences for LSTM (100-day windows).
   - Train the PyTorch model.
   - Evaluate and save the model.
   - Load the saved model and re-evaluate.

Key cells:
- Data download: Uses `yf.download()` with MultiIndex normalization.
- Model architecture: LSTM(128) -> LSTM(64) -> Linear(25) -> Linear(1).
- Training: PyTorch training loop with Adam optimizer.
- Evaluation: Computes MAE, RMSE, MAPE, R2 and plots results.

## Model Details

- **Architecture**: PyTorch LSTM model with two LSTM layers (128 and 64 hidden units), followed by linear layers.
- **Input**: Sequences of 100 days of scaled closing prices (shape: `(batch_size, 100, 1)`).
- **Output**: Predicted closing price for the next day.
- **Training**: Adam optimizer (lr=0.001), MSE loss, batch size 32.
- **Preprocessing**: PyTorch-based MinMaxScaler (no scikit-learn dependency).
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R-squared (R2)

## Dependencies

See `requirements.txt` for the full list. Key packages:
- `torch`: PyTorch for model building and training.
- `streamlit`: Web app framework.
- `yfinance`: Data download.
- `pandas`, `numpy`: Data handling.
- `plotly`: Interactive plotting.
- `matplotlib`: Additional plotting support.


## Notes

- This project uses **pure PyTorch** - no scikit-learn or TensorFlow/Keras dependencies.
- The `src/stock_predictor/scaler.py` module provides a drop-in replacement for sklearn's MinMaxScaler.
- Models are saved as `.pth` files using PyTorch's state_dict format.
- The project follows a clean package structure with organized directories for better maintainability.

This project is for educational purposes. See [LICENSE](https://github.com/bedigambar/Stock-Trend-Predictor-Model/blob/main/LICENSE) file for details.

Developed by Digambar.
