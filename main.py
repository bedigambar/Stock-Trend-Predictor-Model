import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.stock_predictor import StockPredictor, MinMaxScaler


st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")


st.markdown("""
    <style>
    /* Dark theme adjustments */
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #90CAF9;
    }
    /* main page background */
    .stApp {
        background-color: #0f1720;
        color: #e6eef6;
    }
    /* sidebar background */
    .css-1d391kg { /* streamlit class for sidebar container may vary */
        background-color: #071022 !important;
        color: #e6eef6 !important;
    }
    /* cards and widgets */
    .stButton>button, .stTextInput>div>div>input, .stSlider>div input {
        background-color: #12202b !important;
        color: #e6eef6 !important;
    }
    /* metric and text colors */
    .stMetric, .stText {
        color: #e6eef6 !important;
    }
    /* adjust table background */
    .stDataFrame, .stTable {
        background-color: #0b1520 !important;
        color: #e6eef6 !important;
    }
    a { color: #82b1ff }
    </style>
    """, unsafe_allow_html=True)


st.markdown('<p class="big-font">Stock Trend Predictor</p>', unsafe_allow_html=True)


with st.sidebar:
    st.header("Configuration")
    stock = st.text_input("Enter the Stock Symbol", "GOOG").upper().strip()
    years = st.slider("Years of historical data", 1, 20, 10)

    st.markdown("---")
    st.subheader("Model (optional)")
    model_uploader = st.file_uploader("Upload a PyTorch model (.pth)", type=["pth"], accept_multiple_files=False)
    model_path_input = st.text_input("Or provide model path (local)", "")

    st.markdown("---")
    st.subheader("Prediction settings")
    seq_len = st.slider("Sequence length (timesteps)", 10, 200, 100)
    split_ratio = st.slider("Train/Test split (use for testing portion)", 50, 90, 70)

    predict_button = st.button("Predict Trends")


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Stock Data")

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    @st.cache_data
    def load_data(symbol, start, end):
        df = yf.download(symbol, start, end, progress=False)
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in df.columns]
            df.columns = df.columns.astype(str).str.strip()
        except Exception:
            pass
        return df

    if not stock:
        st.info("Enter a stock symbol in the sidebar to load data.")
        stock_data = pd.DataFrame()
    else:
        data_load_state = st.text('Loading data...')
        try:
            stock_data = load_data(stock, start, end)
            if stock_data is None or stock_data.empty:
                st.warning(f"Download returned no data for '{stock}'. Trying fallback using yf.Ticker().history()...")
                try:
                    ticker = yf.Ticker(stock)
                    alt = ticker.history(start=start, end=end)
                    if isinstance(alt, pd.DataFrame) and not alt.empty:
                        if isinstance(alt.columns, pd.MultiIndex):
                            alt.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in alt.columns]
                        alt.columns = alt.columns.astype(str).str.strip()
                        stock_data = alt
                        data_load_state.text('Loading data... done! (fallback)')
                        st.info(f"Loaded data for '{stock}' via yf.Ticker().history(). Rows: {len(stock_data)}")
                        st.dataframe(stock_data.style.highlight_max(axis=0))
                    else:
                        st.error(f"No data found for symbol '{stock}'. Check the symbol and try again.")
                        stock_data = pd.DataFrame()
                except Exception as e2:
                    data_load_state.text('')
                    st.error(f"Error during fallback fetch for {stock}: {e2}")
                    stock_data = pd.DataFrame()
            else:
                data_load_state.text('Loading data... done!')
                st.dataframe(stock_data.style.highlight_max(axis=0))
        except Exception as e:
            data_load_state.text('')
            st.error(f"Error downloading data for {stock}: {e}")
            stock_data = pd.DataFrame()

with col2:
    st.subheader("Stock Info")
    if stock:
        try:
            ticker = yf.Ticker(stock)
            info = ticker.info or {}
            if info.get('logo_url'):
                st.image(info['logo_url'], width=100)
            st.write(f"**{info.get('longName', stock)}**")
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Industry: {info.get('industry', 'N/A')}")

            current = info.get('currentPrice')
            change_pct = info.get('regularMarketChangePercent')
            if current is not None and change_pct is not None:
                st.metric("Current Price", f"${current:.2f}", f"{change_pct:.2f}%")
            else:
                st.write("Current price information not available")
        except Exception:
            st.write("Ticker info not available")
    else:
        st.info("Ticker info will appear here after entering a symbol.")


st.subheader("Stock Price Trend")
if not stock_data.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Close Price"))
    if len(stock_data) >= 100:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'].rolling(100).mean(), name="100-day MA"))
    if len(stock_data) >= 250:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'].rolling(250).mean(), name="250-day MA"))
    fig.update_layout(title=f"{stock} Stock Price", xaxis_title="Date", yaxis_title="Price", legend_title="Indicators")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No price data to plot.")


if predict_button:
    st.subheader("Price Prediction")

    if stock_data.empty:
        st.error("No stock data available. Please enter a valid symbol and try again.")
    else:
        model_file_path = None
        if model_uploader is not None:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(model_uploader.name).suffix)
                tmp.write(model_uploader.getbuffer())
                tmp.flush()
                tmp.close()
                model_file_path = tmp.name
            except Exception as e:
                st.error(f"Failed to save uploaded model: {e}")
        elif model_path_input:
            if os.path.exists(model_path_input):
                model_file_path = model_path_input
            else:
                st.error("Provided model path does not exist.")

        if model_file_path is None:
            possible_models = [
                os.path.join(os.getcwd(), 'model.pth')
            ]
            detected = None
            for p in possible_models:
                if os.path.exists(p):
                    detected = p
                    break

            if detected:
                st.info(f"Auto-detected model: {os.path.basename(detected)}")
                model_file_path = detected
            else:
                st.info("No model provided. Please upload a PyTorch model or provide a local path to run predictions.")
        else:
            with st.spinner("Loading model..."):
                try:
                    model = StockPredictor()
                    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
                    model.eval()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    model = None

            if model is not None:
                splitting_len = int(len(stock_data) * (split_ratio / 100.0))
                x_test = pd.DataFrame(stock_data.Close[splitting_len:])

                if x_test.empty or len(x_test) <= seq_len:
                    st.error("Not enough data in the test split for the selected sequence length. Reduce sequence length or increase data years.")
                else:
                    try:
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(x_test[['Close']])

                        x_data = []
                        y_data = []
                        for i in range(seq_len, len(scaled_data)):
                            x_data.append(scaled_data[i - seq_len:i])
                            y_data.append(scaled_data[i])

                        x_data, y_data = np.array(x_data), np.array(y_data)

                        with st.spinner("Running predictions..."):
                            x_tensor = torch.tensor(x_data, dtype=torch.float32)
                            with torch.no_grad():
                                predictions = model(x_tensor).numpy()

                        inv_pre = scaler.inverse_transform(predictions)
                        inv_y_test = scaler.inverse_transform(y_data)

                        plotting_data = pd.DataFrame(
                            {
                                'Original': inv_y_test.reshape(-1),
                                'Predicted': inv_pre.reshape(-1)
                            },
                            index=stock_data.index[splitting_len + seq_len:]
                        )

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=plotting_data.index, y=plotting_data['Original'], name="Actual Price"))
                        fig.add_trace(go.Scatter(x=plotting_data.index, y=plotting_data['Predicted'], name="Predicted Price"))
                        fig.update_layout(title="Actual vs Predicted Stock Price", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig, use_container_width=True)

                        original = plotting_data['Original'].values
                        predicted = plotting_data['Predicted'].values
                        non_zero_mask = original != 0
                        if non_zero_mask.sum() == 0:
                            mape = np.nan
                        else:
                            mape = np.mean(np.abs((original[non_zero_mask] - predicted[non_zero_mask]) / original[non_zero_mask])) * 100

                        rmse = np.sqrt(np.mean((original - predicted)**2))

                        cols = st.columns(3)
                        cols[0].metric("Data used for test", f"{len(x_test)} rows")
                        cols[1].metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
                        cols[2].metric("RMSE", f"{rmse:.4f}")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")


st.markdown("---")
with st.expander("About this app"):
    st.write("This app downloads historical stock data and runs a pre-trained PyTorch model to predict prices. Upload a compatible model or provide a path. Adjust sequence length and test split to match your model's training settings.")