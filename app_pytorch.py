import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import math
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class StockTransformer(nn.Module):
    """Transformer model for stock price prediction"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=512, dropout=0.1, max_seq_length=200):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.fc2 = nn.Linear(dim_feedforward // 2, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Project to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Output layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_synthetic_stock_data(num_days=1000, start_price=100):
    """Generate synthetic stock data for demonstration"""
    
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')
    
    # Generate realistic stock price movements with trend and volatility
    returns = np.random.normal(0.001, 0.02, num_days)
    
    # Add some trend and cycles
    trend = np.linspace(0, 0.5, num_days) * 0.001
    cycle = 0.0005 * np.sin(2 * np.pi * np.arange(num_days) / 252)  # Annual cycle
    
    returns = returns + trend + cycle
    
    prices = [start_price]
    for i in range(1, num_days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1))  # Prevent negative prices
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        daily_volatility = abs(returns[i]) + 0.01
        open_price = price * np.random.uniform(0.98, 1.02)
        high_price = max(open_price, price) * (1 + daily_volatility * np.random.uniform(0, 1))
        low_price = min(open_price, price) * (1 - daily_volatility * np.random.uniform(0, 1))
        close_price = price
        volume = np.random.randint(500000, 5000000) * (1 + abs(returns[i]) * 10)
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': int(volume)
        })
    
    return pd.DataFrame(data)

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
    
    # Price changes and volatility
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    # Volatility (rolling standard deviation)
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    
    return df

def load_and_preprocess_data(df):
    """Load and preprocess stock data"""
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Select features for model
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_histogram',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
        'Volatility'
    ]
    
    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < 100:
        st.error("Not enough data after preprocessing. Need at least 100 rows.")
        return None, None, None, None
    
    data = df[feature_columns].values
    
    # Initialize scalers
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Scale all features
    scaled_data = scaler_features.fit_transform(data)
    
    # Scale target (Close price) separately for inverse transformation
    target_data = df[['Close']].values
    scaled_target = scaler_target.fit_transform(target_data)
    
    return scaled_data, scaler_features, scaler_target, df

def create_sequences(data, sequence_length, target_column_index=3):
    """Create sequences for time series prediction"""
    
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Input sequence (past sequence_length days)
        X.append(data[i-sequence_length:i])
        # Target (next day's close price)
        y.append(data[i, target_column_index])
    
    return np.array(X), np.array(y)

def train_test_split_temporal(X, y, test_size=0.2):
    """Split data chronologically for time series"""
    
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, progress_callback=None):
    """Train the transformer model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if progress_callback:
            progress_callback(epoch, num_epochs, train_loss, val_loss)
    
    return train_losses, val_losses

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Prices"):
    """Plot actual vs predicted prices"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label='Actual', alpha=0.8, linewidth=2)
    ax.plot(y_pred, label='Predicted', alpha=0.8, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate directional accuracy
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(true_direction == pred_direction) * 100
    
    return rmse, mae, mape, directional_accuracy

# Streamlit App
def main():
    st.title("🚀 Stock Price Prediction with PyTorch Transformer")
    st.markdown("*Advanced time series forecasting using state-of-the-art transformer architecture*")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Model Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "📊 Select Data Source:",
        ["Upload CSV File", "Generate Synthetic Data"]
    )
    
    # Model hyperparameters
    st.sidebar.subheader("🎛️ Transformer Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60)
    d_model = st.sidebar.selectbox("Model Dimension", [64, 128, 256], index=1)
    nhead = st.sidebar.selectbox("Number of Attention Heads", [4, 8, 16], index=1)
    num_layers = st.sidebar.slider("Transformer Layers", 2, 8, 4)
    
    # Training parameters
    st.sidebar.subheader("🏋️ Training Parameters")
    epochs = st.sidebar.slider("Training Epochs", 20, 200, 100)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=1)
    
    # Load data
    df = None
    
    if data_source == "Upload CSV File":
        st.subheader("📁 Upload Stock Data")
        st.info("Upload a CSV file with columns: Date, Open, High, Low, Close, Volume")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    return
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
    
    else:
        st.subheader("🎲 Generate Synthetic Data")
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.number_input("Number of Days", 500, 3000, 1500)
        with col2:
            start_price = st.number_input("Starting Price", 50, 500, 100)
        
        if st.button("Generate Data", type="primary"):
            with st.spinner("Generating realistic synthetic stock data..."):
                df = generate_synthetic_stock_data(num_days, start_price)
                st.success(f"Synthetic data generated! Shape: {df.shape}")
    
    if df is not None:
        # Display data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10))
        
        # Display basic statistics
        st.subheader("📈 Data Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            st.metric("Average Close", f"${df['Close'].mean():.2f}")
        with col3:
            st.metric("Min Close", f"${df['Close'].min():.2f}")
        with col4:
            st.metric("Max Close", f"${df['Close'].max():.2f}")
        with col5:
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.1f}%")
        
        # Plot price history
        st.subheader("📉 Price History")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Price chart
        ax1.plot(df.index, df['Close'], linewidth=2)
        ax1.set_title("Historical Closing Prices")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(df.index, df['Volume'], alpha=0.7)
        ax2.set_title("Trading Volume")
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Train model button
        if st.button("🚀 Train Transformer Model", type="primary"):
            
            with st.spinner("Preprocessing data and calculating technical indicators..."):
                # Preprocess data
                result = load_and_preprocess_data(df)
                if result[0] is None:
                    return
                
                scaled_data, scaler_features, scaler_target, original_df = result
                
                # Create sequences
                X, y = create_sequences(scaled_data, sequence_length)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
                
                st.success(f"Data preprocessed! Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
                st.info(f"Feature dimensions: {X.shape[-1]} features per timestep")
            
            with st.spinner("Building PyTorch Transformer model..."):
                # Build model
                input_dim = X.shape[-1]
                
                model = StockTransformer(
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    max_seq_length=sequence_length + 10
                )
                
                # Model summary
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                st.success(f"Transformer model built successfully!")
                st.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
                
                # Display model architecture
                with st.expander("View Model Architecture"):
                    st.text(str(model))
            
            # Prepare data loaders
            train_dataset = StockDataset(X_train, y_train)
            test_dataset = StockDataset(X_test, y_test)
            
            # Split training data for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Training progress
            st.subheader("🏋️ Model Training")
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            
            # Store losses for plotting
            train_losses = []
            val_losses = []
            
            def update_progress(epoch, total_epochs, train_loss, val_loss):
                progress = (epoch + 1) / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Update loss chart every 10 epochs
                if (epoch + 1) % 10 == 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(train_losses, label='Training Loss', alpha=0.8)
                    ax.plot(val_losses, label='Validation Loss', alpha=0.8)
                    ax.set_title('Training Progress')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    loss_chart.pyplot(fig)
                    plt.close()
            
            # Train model
            with st.spinner("Training transformer model..."):
                train_losses_final, val_losses_final = train_model(
                    model, train_loader, val_loader, epochs, learning_rate, device, update_progress
                )
            
            st.success("Model training completed!")
            
            # Final training history plot
            st.subheader("📊 Training History")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_losses_final, label='Training Loss', alpha=0.8)
            ax.plot(val_losses_final, label='Validation Loss', alpha=0.8)
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Make predictions
            st.subheader("🎯 Model Predictions")
            
            with st.spinner("Generating predictions..."):
                model.eval()
                predictions = []
                actuals = []
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        predictions.extend(outputs.cpu().numpy())
                        actuals.extend(batch_y.numpy())
                
                # Convert to numpy arrays
                y_pred_scaled = np.array(predictions).flatten()
                y_test_scaled = np.array(actuals)
                
                # Inverse transform predictions
                y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_test_original = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                rmse, mae, mape, directional_accuracy = calculate_metrics(y_test_original, y_pred)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{rmse:.2f}")
            with col2:
                st.metric("MAE", f"{mae:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            with col4:
                st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%")
            
            # Plot predictions
            st.subheader("📈 Prediction Results")
            
            # Test set predictions
            fig = plot_predictions(y_test_original, y_pred, "Test Set: Actual vs Predicted Prices")
            st.pyplot(fig)
            
            # Prediction error analysis
            st.subheader("📊 Prediction Error Analysis")
            
            errors = y_test_original - y_pred
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Error distribution
            ax1.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title('Prediction Error Distribution')
            ax1.set_xlabel('Error (Actual - Predicted)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Error over time
            ax2.plot(errors, alpha=0.7)
            ax2.set_title('Prediction Errors Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Model insights
            st.subheader("🧠 Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance:**")
                st.write(f"- The transformer model achieved an RMSE of {rmse:.2f}")
                st.write(f"- Mean absolute error is {mae:.2f}")
                st.write(f"- Directional accuracy: {directional_accuracy:.1f}%")
                
            with col2:
                st.write("**Technical Details:**")
                st.write(f"- Model dimension: {d_model}")
                st.write(f"- Attention heads: {nhead}")
                st.write(f"- Transformer layers: {num_layers}")
                st.write(f"- Sequence length: {sequence_length}")

if __name__ == "__main__":
    main()