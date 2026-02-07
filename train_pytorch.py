import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pytorch_scaler import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_model import StockPredictor

def train():
    print("Loading data...")
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    stock_symbol = "GOOG"
    
    data = yf.download(stock_symbol, start, end, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in data.columns]
    data.columns = data.columns.astype(str).str.strip()
    
    adj_close = data[['Close']]
    
    print("Preprocessing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(adj_close)
    
    x_data = []
    y_data = []
    seq_len = 100
    
    for i in range(seq_len, len(scaled_data)):
        x_data.append(scaled_data[i-seq_len:i])
        y_data.append(scaled_data[i])
        
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    split_ratio = 0.7
    splitting_len = int(len(x_data) * split_ratio)
    
    x_train = x_data[:splitting_len]
    y_train = y_data[:splitting_len]
    
    x_test = x_data[splitting_len:]
    y_test = y_data[splitting_len:]
    
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    batch_size = 32
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    print("Setting up model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = StockPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
        
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as 'model.pth'")

if __name__ == "__main__":
    train()
