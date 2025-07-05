import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PositionalEmbedding(layers.Layer):
    """Custom positional embedding layer for Transformer"""
    
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.position_embedding = None
        
    def build(self, input_shape):
        # Create positional embedding weights during build phase
        d_model = input_shape[-1]
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.sequence_length, d_model),
            initializer="uniform",
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return inputs + self.position_embedding

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Single Transformer encoder block"""
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    
    # Add & Norm
    attention_output = layers.Dropout(dropout)(attention_output)
    x1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = layers.Dense(ff_dim, activation="relu")(x1)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    
    # Add & Norm
    return layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, 
                          num_transformer_blocks, mlp_units, dropout=0.1, mlp_dropout=0.1):
    """Build complete Transformer model for stock prediction"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Add positional encoding
    x = PositionalEmbedding(input_shape[0])(inputs)
    
    # Stack Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # MLP head for prediction
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # Output layer for single value prediction
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

### --- Galformer Model Code --- ###
# The following classes implement the Galformer architecture, adapted for this application.
# Original source: https://github.com/AnswerXuan/Galformer-Improved-Transformer-for-Time-Series-Prediction

class SeasonalLayer(layers.Layer):
    def __init__(self, d_model, n_seasons=4, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_seasons = n_seasons
        self.season_convs = [layers.Conv1D(filters=d_model, kernel_size=k, padding='causal')
                             for k in range(1, n_seasons + 1)]

    def call(self, inputs):
        seasonal_outputs = []
        for conv in self.season_convs:
            seasonal_outputs.append(conv(inputs))
        
        # Stack and average the seasonal features
        stacked_seasons = tf.stack(seasonal_outputs, axis=-1)
        season_output = tf.reduce_mean(stacked_seasons, axis=-1)
        
        return season_output

class AutoCorrelation(layers.Layer):
    def __init__(self, factor=1, **kwargs):
        super(AutoCorrelation, self).__init__(**kwargs)
        self.factor = factor

    def call(self, queries, keys, values):
        B, L, _ = tf.unstack(tf.shape(queries))
        _, S, _ = tf.unstack(tf.shape(keys))
        
        # Compute autocorrelation using FFT
        q_fft = tf.signal.rfft(queries, fft_length=[L])
        k_fft = tf.signal.rfft(keys, fft_length=[S])
        
        # Conjugate and multiply
        R = q_fft * tf.math.conj(k_fft)
        corr = tf.signal.irfft(R, fft_length=[L + S -1])

        # Find top-k delays
        top_k = int(self.factor * tf.math.log(tf.cast(L, dtype=tf.float32)))
        mean_value = tf.reduce_mean(corr, axis=1)
        
        # Ensure top_k is not larger than the correlation length
        top_k = min(top_k, tf.shape(mean_value)[-1])

        top_k_val, top_k_ind = tf.math.top_k(mean_value, k=top_k)
        
        # Gather values based on top delays
        tmp_corr = tf.nn.softmax(top_k_val, axis=-1)
        
        # Roll values and sum
        V_rolled = []
        for i in range(top_k):
            index = top_k_ind[:, i]
            # Ensure index is a scalar for tf.roll
            rolled = tf.roll(values, shift=-index[0], axis=1)
            V_rolled.append(rolled)
        
        V_stacked = tf.stack(V_rolled, axis=-2)
        
        # Weighted sum of rolled values
        output = tf.einsum('bld,bl->bd', V_stacked, tmp_corr)
        return output, None # Return None for compatibility with MHA layer

class AutoCorrelationLayer(layers.Layer):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, **kwargs):
        super(AutoCorrelationLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.correlation = correlation
        self.query_projection = layers.Dense(d_model)
        self.key_projection = layers.Dense(d_model)
        self.value_projection = layers.Dense(d_model)
        self.out_projection = layers.Dense(d_model)

    def call(self, queries, keys, values):
        B, L, _ = tf.unstack(tf.shape(queries))
        _, S, _ = tf.unstack(tf.shape(keys))
        
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Reshape for multi-head
        queries = tf.reshape(queries, (B, L, self.n_heads, -1))
        keys = tf.reshape(keys, (B, S, self.n_heads, -1))
        values = tf.reshape(values, (B, S, self.n_heads, -1))

        out, _ = self.correlation(queries, keys, values)
        out = tf.reshape(out, (B, L, -1))
        
        return self.out_projection(out)

class DataEmbedding(layers.Layer):
    def __init__(self, c_in, d_model, dropout=0.1, **kwargs):
        super(DataEmbedding, self).__init__(**kwargs)
        self.value_embedding = layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')
        self.position_embedding = PositionalEmbedding(sequence_length=5000) # Use a large capacity
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", factor=1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        d_ff = d_ff or 4 * d_model
        self.attention = AutoCorrelationLayer(AutoCorrelation(factor), d_model, n_heads)
        self.conv1 = layers.Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        self.activation = layers.Activation(activation)

    def call(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x + y)

class Encoder(layers.Layer):
    def __init__(self, attn_layers, norm_layer=None, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.attn_layers = attn_layers
        self.norm = norm_layer

    def call(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", factor=1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        d_ff = d_ff or 4 * d_model
        self.self_attention = AutoCorrelationLayer(AutoCorrelation(factor), d_model, n_heads)
        self.cross_attention = AutoCorrelationLayer(AutoCorrelation(factor), d_model, n_heads)
        self.conv1 = layers.Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        self.activation = layers.Activation(activation)

    def call(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm3(x + y)

class Decoder(layers.Layer):
    def __init__(self, layers, norm_layer=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.layers = layers
        self.norm = norm_layer

    def call(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        if self.norm is not None:
            x = self.norm(x)
        return x

def generate_synthetic_stock_data(num_days=1000, start_price=100):
    """Generate synthetic stock data for demonstration"""
    
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')
    
    # Generate realistic stock price movements
    returns = np.random.normal(0.001, 0.02, num_days)  # Daily returns
    prices = [start_price]
    
    for i in range(1, num_days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1))  # Prevent negative prices
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        open_price = price * np.random.uniform(0.98, 1.02)
        high_price = max(open_price, price) * np.random.uniform(1.0, 1.05)
        low_price = min(open_price, price) * np.random.uniform(0.95, 1.0)
        close_price = price
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data)

def fetch_market_indices(start_date, end_date):
    """Fetch SPX and NASDAQ indices data"""
    try:
        st.info("Fetching market indices data...")
        
        # Add some buffer days to ensure we get data
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch S&P 500 (^GSPC) and NASDAQ (^IXIC)
        spx = yf.download('^GSPC', start=start_date_buffer, end=end_date_buffer, progress=False)
        nasdaq = yf.download('^IXIC', start=start_date_buffer, end=end_date_buffer, progress=False)
        
        if spx.empty or nasdaq.empty:
            st.warning("No market data available for the given date range")
            return None, None
        
        # Reset index to make Date a column
        spx = spx.reset_index()
        nasdaq = nasdaq.reset_index()
        
        # Select only Close prices and rename columns
        spx_data = spx[['Date', 'Close']].rename(columns={'Close': 'SPX_Close'})
        nasdaq_data = nasdaq[['Date', 'Close']].rename(columns={'Close': 'NASDAQ_Close'})
        
        st.success(f"✅ Successfully fetched market data: SPX ({len(spx_data)} days), NASDAQ ({len(nasdaq_data)} days)")
        
        return spx_data, nasdaq_data
    except Exception as e:
        st.error(f"Could not fetch market indices: {str(e)}")
        st.info("Will proceed without market indices data")
        return None, None

def create_volume_features(df):
    """Create enhanced volume-based features"""
    volume_features = df.copy()
    
    # Volume moving averages
    volume_features['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    volume_features['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Volume ratio (current vs average)
    volume_features['Volume_Ratio'] = df['Volume'] / volume_features['Volume_MA_20']
    
    # Price-volume features
    volume_features['Price_Volume'] = df['Close'] * df['Volume']
    volume_features['Volume_Price_Trend'] = (df['Close'].pct_change() * df['Volume']).rolling(window=5).mean()
    
    # Volume momentum
    volume_features['Volume_Change'] = df['Volume'].pct_change()
    volume_features['Volume_Momentum'] = df['Volume'].rolling(window=5).apply(lambda x: x[-1] / x[0] - 1)
    
    return volume_features

def add_market_context_features(df, include_indices=True, enhance_volume=True):
    """Add market context features including indices and enhanced volume features"""
    enhanced_df = df.copy()
    
    if 'Date' in enhanced_df.columns:
        enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date'])
        start_date = enhanced_df['Date'].min()
        end_date = enhanced_df['Date'].max()
        
        if include_indices:
            # Fetch market indices
            spx_data, nasdaq_data = fetch_market_indices(start_date, end_date)
            
            if spx_data is not None and nasdaq_data is not None:
                # Merge with main dataframe
                enhanced_df = enhanced_df.merge(spx_data, on='Date', how='left')
                enhanced_df = enhanced_df.merge(nasdaq_data, on='Date', how='left')
                
                # Forward fill missing values
                enhanced_df['SPX_Close'] = enhanced_df['SPX_Close'].fillna(method='ffill').fillna(enhanced_df['SPX_Close'].bfill())
                enhanced_df['NASDAQ_Close'] = enhanced_df['NASDAQ_Close'].fillna(method='ffill').fillna(enhanced_df['NASDAQ_Close'].bfill())
                
                # Create relative performance features
                enhanced_df['Stock_vs_SPX'] = enhanced_df['Close'] / enhanced_df['SPX_Close']
                enhanced_df['Stock_vs_NASDAQ'] = enhanced_df['Close'] / enhanced_df['NASDAQ_Close']
                
                # Market momentum features
                enhanced_df['SPX_Returns'] = enhanced_df['SPX_Close'].pct_change()
                enhanced_df['NASDAQ_Returns'] = enhanced_df['NASDAQ_Close'].pct_change()
                enhanced_df['Stock_Returns'] = enhanced_df['Close'].pct_change()
                
                # Correlation features (rolling correlation with market)
                enhanced_df['SPX_Correlation'] = enhanced_df['Stock_Returns'].rolling(window=20).corr(enhanced_df['SPX_Returns'])
                enhanced_df['NASDAQ_Correlation'] = enhanced_df['Stock_Returns'].rolling(window=20).corr(enhanced_df['NASDAQ_Returns'])
    
    if enhance_volume:
        # Add enhanced volume features
        volume_enhanced = create_volume_features(enhanced_df)
        enhanced_df = volume_enhanced
    
    # Drop rows with NaN values (due to rolling calculations)
    enhanced_df = enhanced_df.dropna()
    
    return enhanced_df

def load_and_preprocess_data_enhanced(df, include_indices=True, enhance_volume=True, include_industry_etf=True, ticker_symbol=None):
    """Enhanced data preprocessing with market context, volume features, and industry ETF data"""
    
    # Add market context and volume features
    enhanced_df = add_market_context_features(df, include_indices, enhance_volume)
    
    # Add industry ETF features if requested and ticker is provided
    if include_industry_etf and ticker_symbol:
        try:
            ticker_info = get_ticker_industry_info(ticker_symbol)
            enhanced_df = add_industry_context_features(enhanced_df, ticker_info, include_industry_etf)
            st.success(f"Added industry ETF features for {ticker_symbol}")
        except Exception as e:
            st.warning(f"Could not add industry ETF features: {str(e)}")
            include_industry_etf = False
    
    # Ensure Date column is datetime and sort
    if 'Date' in enhanced_df.columns:
        enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date'])
        enhanced_df = enhanced_df.sort_values('Date').reset_index(drop=True)
    
    # Define feature columns based on what's available
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    volume_features = []
    if enhance_volume:
        volume_features = ['Volume_MA_5', 'Volume_MA_20', 'Volume_Ratio', 'Price_Volume', 
                          'Volume_Price_Trend', 'Volume_Change', 'Volume_Momentum']
    
    market_features = []
    if include_indices and 'SPX_Close' in enhanced_df.columns:
        market_features = ['SPX_Close', 'NASDAQ_Close', 'Stock_vs_SPX', 'Stock_vs_NASDAQ',
                          'SPX_Returns', 'NASDAQ_Returns', 'Stock_Returns', 
                          'SPX_Correlation', 'NASDAQ_Correlation']
    
    industry_features = []
    if include_industry_etf and 'Industry_ETF_Close' in enhanced_df.columns:
        industry_features = ['Industry_ETF_Close', 'Industry_ETF_Volume', 'Stock_vs_Industry_ETF',
                           'Industry_ETF_Returns', 'Stock_Industry_Correlation', 'Stock_Industry_Beta',
                           'Industry_ETF_Relative_Volume']
    
    # Combine all features
    feature_columns = base_features + volume_features + market_features + industry_features
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in enhanced_df.columns]
    
    data = enhanced_df[available_features].values
    
    # Initialize scalers
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Scale all features
    scaled_data = scaler_features.fit_transform(data)
    
    # Scale target (Close price) separately for inverse transformation
    target_data = enhanced_df[['Close']].values
    scaled_target = scaler_target.fit_transform(target_data)
    
    return scaled_data, scaler_features, scaler_target, enhanced_df, available_features

def load_and_preprocess_data(df):
    """Load and preprocess stock data"""
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Select features for model
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[feature_columns].values
    
    # Initialize scalers
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Scale all features
    scaled_data = scaler_features.fit_transform(data)
    
    # Scale target (Close price) separately for inverse transformation
    target_data = df[['Close']].values
    scaler_target.fit(target_data)
    scaled_target = scaler_target.transform(target_data)
    
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

def create_sequences_for_galformer(data, sequence_length, label_len, target_column_index=3):
    """
    Create sequences for the Galformer (Encoder-Decoder) model.
    This model requires three parts:
    1. Encoder Input: The historical sequence.
    2. Decoder Input: The known part of the target sequence (for teacher forcing).
    3. Target (y): The value to be predicted.
    """
    encoder_inputs, decoder_inputs, targets = [], [], []

    # We need at least `sequence_length` data points to form the first sequence.
    # The loop will go up to the second to last element, as the last element will be the target.
    for i in range(sequence_length, len(data)):
        # 1. Encoder input: The full look-back window.
        enc_in = data[i - sequence_length:i]
        encoder_inputs.append(enc_in)

        # 2. Decoder input: The last `label_len` part of the known sequence.
        # This part acts as a "prompt" or "start token" for the decoder.
        dec_in = data[i - label_len:i]
        decoder_inputs.append(dec_in)

        # 3. Target: The 'Close' price of the next day.
        target = data[i, target_column_index]
        targets.append(target)

    
def train_test_split_temporal(X, y, test_size=0.2):
    """Split data chronologically for time series"""
    
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Prices"):
    """Plot actual vs predicted prices"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label='Actual', alpha=0.8)
    ax.plot(y_pred, label='Predicted', alpha=0.8)
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
    
    return rmse, mae, mape

def predict_future_prices(model, last_sequence, scaler_target, num_days=30):
    """Predict future stock prices for a given number of days"""
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        # Predict next day
        next_pred_scaled = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        next_pred = scaler_target.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        # Take the last sequence, remove first element, add prediction as new last element
        # We need to scale the prediction back and create a full feature vector
        next_pred_scaled_val = next_pred_scaled[0, 0]
        
        # Create a new row with the predicted close price
        # For simplicity, we'll replicate the close price across OHLC and keep volume constant
        new_row = current_sequence[-1].copy()
        new_row[0] = next_pred_scaled_val  # Open
        new_row[1] = next_pred_scaled_val * 1.01  # High (slightly higher)
        new_row[2] = next_pred_scaled_val * 0.99  # Low (slightly lower)
        new_row[3] = next_pred_scaled_val  # Close
        # Volume stays the same as the last day
        
        # Shift sequence and add new prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_row
    
    return np.array(predictions)

def plot_future_predictions(historical_prices, future_predictions, num_historical_days=100, title="Stock Price Forecast"):
    """Plot historical prices with future predictions"""
    
    # Take last num_historical_days for context
    historical_subset = historical_prices[-num_historical_days:] if len(historical_prices) > num_historical_days else historical_prices
    
    # Create time indices
    historical_indices = range(len(historical_subset))
    future_indices = range(len(historical_subset), len(historical_subset) + len(future_predictions))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot historical prices
    ax.plot(historical_indices, historical_subset, label='Historical Prices', color='blue', linewidth=2)
    
    # Plot future predictions
    ax.plot(future_indices, future_predictions, label='Future Predictions', color='red', linewidth=2, linestyle='--')
    
    # Add a connecting line between last historical and first prediction
    ax.plot([historical_indices[-1], future_indices[0]], 
            [historical_subset[-1], future_predictions[0]], 
            color='orange', linewidth=2, alpha=0.7)
    
    # Add vertical line to separate historical and future
    ax.axvline(x=len(historical_subset)-1, color='gray', linestyle=':', alpha=0.7, label='Today')
    
    # Add confidence band (simple approach - you could make this more sophisticated)
    future_std = np.std(historical_subset[-30:]) if len(historical_subset) >= 30 else np.std(historical_subset)
    upper_bound = future_predictions + future_std
    lower_bound = future_predictions - future_std
    
    ax.fill_between(future_indices, lower_bound, upper_bound, alpha=0.2, color='red', label='Confidence Band')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    return fig

def create_download_link(df, filename="stock_data.csv"):
    """Create a download link for the dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def fetch_ticker_data(ticker, start_date, end_date):
    """Fetch historical data for a specific ticker"""
    try:
        st.info(f"Fetching data for {ticker}...")
        
        # Add some buffer days to ensure we get data
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch ticker data
        ticker_data = yf.download(ticker, start=start_date_buffer, end=end_date_buffer, progress=False)
        
        if ticker_data.empty:
            st.error(f"No data available for ticker {ticker} in the given date range")
            return None
        
        # Reset index to make Date a column
        ticker_data = ticker_data.reset_index()
        
        # Rename columns to match expected format
        ticker_data = ticker_data.rename(columns={
            'Adj Close': 'Close'  # Use adjusted close if available
        })
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in ticker_data.columns for col in required_columns):
            ticker_data = ticker_data[required_columns]
            st.success(f"✅ Successfully fetched {len(ticker_data)} days of data for {ticker}")
            return ticker_data
        else:
            st.error(f"Missing required columns for {ticker}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def fetch_multiple_tickers(tickers, start_date, end_date):
    """Fetch data for multiple tickers"""
    try:
        st.info(f"Fetching data for multiple tickers: {', '.join(tickers)}")
        
        # Add buffer days
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch data for all tickers at once
        data = yf.download(tickers, start=start_date_buffer, end=end_date_buffer, progress=False, group_by='ticker')
        
        if data.empty:
            st.error("No data available for the specified tickers")
            return None
        
        ticker_dataframes = {}
        
        if len(tickers) == 1:
            # Single ticker case
            ticker = tickers[0]
            ticker_data = data.reset_index()
            ticker_data = ticker_data.rename(columns={'Adj Close': 'Close'})
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in ticker_data.columns for col in required_columns):
                ticker_dataframes[ticker] = ticker_data[required_columns]
        else:
            # Multiple tickers case
            for ticker in tickers:
                try:
                    ticker_data = data[ticker].reset_index()
                    ticker_data = ticker_data.rename(columns={'Adj Close': 'Close'})
                    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in ticker_data.columns for col in required_columns):
                        ticker_dataframes[ticker] = ticker_data[required_columns].dropna()
                except:
                    st.warning(f"Could not process data for {ticker}")
        
        st.success(f"✅ Successfully fetched data for {len(ticker_dataframes)} tickers")
        return ticker_dataframes
        
    except Exception as e:
        st.error(f"Error fetching multiple ticker data: {str(e)}")
        return None

def get_popular_tickers():
    """Return a list of popular stock tickers and indices"""
    return {
        "Popular Stocks": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
            "CRM", "INTC", "CSCO", "PFE", "VZ", "KO", "PEP", "T", "XOM", "CVX"
        ],
        "Market Indices": [
            "^GSPC",  # S&P 500
            "^IXIC",  # NASDAQ
            "^DJI",   # Dow Jones
            "^RUT",   # Russell 2000
            "^VIX",   # VIX
            "^TNX",   # 10-Year Treasury
            "^FTSE",  # FTSE 100
            "^N225",  # Nikkei 225
        ],
        "ETFs": [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", 
            "BND", "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLV"
        ],
        "Crypto": [
            "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOGE-USD"
        ]
    }

def search_ticker_info(ticker):
    """Get basic info about a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Currency': info.get('currency', 'N/A')
        }
    except:
        return None

def create_ticker_search_widget():
    """Create a ticker search widget"""
    with st.expander("🔍 Ticker Search & Info"):
        search_ticker = st.text_input("Search for ticker information:", placeholder="e.g., AAPL")
        
        if search_ticker and st.button("Search"):
            ticker_info = search_ticker_info(search_ticker.upper())
            if ticker_info:
                st.write("**Company Information:**")
                for key, value in ticker_info.items():
                    st.write(f"• **{key}**: {value}")
            else:
                st.error("Could not fetch information for this ticker")

def create_ticker_selection_interface():
    """Create the ticker selection interface"""
    st.subheader("📈 Fetch Real Market Data")
    
    # Add ticker search widget
    create_ticker_search_widget()
    
    # Get popular tickers
    popular_tickers = get_popular_tickers()
    
    # Selection method
    selection_method = st.radio(
        "How would you like to select data?",
        ["Choose from Popular Lists", "Enter Custom Ticker(s)", "Upload Ticker List"],
        key="ticker_selection_method"
    )
    
    selected_tickers = []
    
    if selection_method == "Choose from Popular Lists":
        st.write("**Select from popular categories:**")
        
        # Create tabs for different categories
        tabs = st.tabs(list(popular_tickers.keys()))
        
        for i, (category, tickers) in enumerate(popular_tickers.items()):
            with tabs[i]:
                selected_from_category = st.multiselect(
                    f"Select {category}:",
                    tickers,
                    key=f"select_{category}"
                )
                selected_tickers.extend(selected_from_category)
    
    elif selection_method == "Enter Custom Ticker(s)":
        ticker_input = st.text_input(
            "Enter ticker symbols (comma-separated):",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter one or more ticker symbols separated by commas"
        )
        if ticker_input:
            selected_tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    elif selection_method == "Upload Ticker List":
        uploaded_tickers = st.file_uploader(
            "Upload a text file with ticker symbols (one per line)",
            type=['txt'],
            key="ticker_file"
        )
        if uploaded_tickers:
            ticker_content = uploaded_tickers.read().decode('utf-8')
            selected_tickers = [ticker.strip().upper() for ticker in ticker_content.split('\n') if ticker.strip()]
            st.write(f"Loaded {len(selected_tickers)} tickers from file")
    
    return selected_tickers

def plot_ticker_comparison(ticker_dataframes, selected_ticker=None):
    """Plot comparison of multiple tickers"""
    if len(ticker_dataframes) <= 1:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize prices to starting value for comparison
    for ticker, data in ticker_dataframes.items():
        normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
        line_style = '-' if ticker == selected_ticker else '--'
        line_width = 2.5 if ticker == selected_ticker else 1.5
        alpha = 1.0 if ticker == selected_ticker else 0.7
        
        ax.plot(range(len(normalized_prices)), normalized_prices, 
               label=ticker, linestyle=line_style, linewidth=line_width, alpha=alpha)
    
    ax.set_title("Ticker Performance Comparison (Normalized to 100)", fontsize=16, fontweight='bold')
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Normalized Price (Starting Price = 100)', fontsize=12)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Highlight selected ticker
    if selected_ticker:
        ax.text(0.02, 0.98, f"Training on: {selected_ticker}", 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               verticalalignment='top')
    
    plt.tight_layout()
    return fig

def display_ticker_data_interface():
    """Display the complete ticker data fetching interface"""
    st.markdown("---")
    
    # Ticker selection
    selected_tickers = create_ticker_selection_interface()
    
    if selected_tickers:
        st.write(f"**Selected tickers:** {', '.join(selected_tickers)}")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.now() - pd.Timedelta(days=365),
                key="ticker_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp.now(),
                key="ticker_end_date"
            )
        
        # Fetch data button
        if st.button("📊 Fetch Market Data", type="primary", use_container_width=True):
            if len(selected_tickers) == 1:
                # Single ticker
                ticker_data = fetch_ticker_data(selected_tickers[0], pd.Timestamp(start_date), pd.Timestamp(end_date))
                if ticker_data is not None:
                    st.session_state.df = ticker_data
                    st.session_state.current_ticker = selected_tickers[0]
                    st.success(f"✅ Data for {selected_tickers[0]} loaded successfully!")
                    
                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ticker", selected_tickers[0])
                    with col2:
                        st.metric("Days of Data", len(ticker_data))
                    with col3:
                        st.metric("Current Price", f"${ticker_data['Close'].iloc[-1]:.2f}")
                    
                    # Show preview
                    st.write("**Data Preview:**")
                    st.dataframe(ticker_data.head())
            
            else:
                # Multiple tickers
                ticker_dataframes = fetch_multiple_tickers(selected_tickers, pd.Timestamp(start_date), pd.Timestamp(end_date))
                if ticker_dataframes:
                    # Let user choose which ticker to use for training
                    st.success(f"✅ Fetched data for {len(ticker_dataframes)} tickers")
                    
                    available_tickers = list(ticker_dataframes.keys())
                    selected_for_training = st.selectbox(
                        "Select ticker for model training:",
                        available_tickers,
                        key="training_ticker"
                    )
                    
                    if selected_for_training:
                        st.session_state.df = ticker_dataframes[selected_for_training]
                        st.session_state.current_ticker = selected_for_training
                        st.session_state.all_ticker_data = ticker_dataframes
                        
                        # Display comparison metrics
                        st.write("**Ticker Comparison:**")
                        comparison_data = []
                        for ticker, data in ticker_dataframes.items():
                            comparison_data.append({
                                'Ticker': ticker,
                                'Days': len(data),
                                'Current Price': f"${data['Close'].iloc[-1]:.2f}",
                                'Price Change (%)': f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%",
                                'Avg Volume': f"{data['Volume'].mean():,.0f}"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Show preview of selected ticker
                        st.write(f"**Data Preview for {selected_for_training}:**")
                        st.dataframe(ticker_dataframes[selected_for_training].head())
                        
                        # Plot comparison chart
                        st.subheader("📊 Ticker Performance Comparison")
                        fig = plot_ticker_comparison(ticker_dataframes, selected_for_training)
                        st.pyplot(fig)
    
    return len(selected_tickers) > 0 if selected_tickers else False

# Streamlit App
def main():
    st.title("🔮 Stock Price Prediction with Transformer Model")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Upload CSV File", "Fetch Market Data (yfinance)", "Generate Synthetic Data"]
    )
    
    # Model hyperparameters
    st.sidebar.subheader("Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60)
    num_heads = st.sidebar.selectbox("Number of Attention Heads", [4, 8, 12], index=1)
    num_transformer_blocks = st.sidebar.slider("Transformer Blocks", 1, 4, 2)
    ff_dim = st.sidebar.selectbox("Feed Forward Dimension", [64, 128, 256], index=1)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    
    # Enhanced features
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Enhanced Features")
    st.sidebar.markdown("*Boost prediction accuracy with additional market context*")
    
    include_market_indices = st.sidebar.checkbox("📊 Include SPX & NASDAQ", value=False, 
                                                help="Add S&P 500 and NASDAQ indices for market context")
    enhance_volume_features = st.sidebar.checkbox("📈 Enhanced Volume Features", value=True,
                                                 help="Add volume moving averages, ratios, and momentum")
    include_industry_etf = st.sidebar.checkbox("🏭 Industry ETF Features", value=True,
                                              help="Add industry ETF data for sector-specific context")
    
    # Model type selection
    st.sidebar.markdown("---")
    model_type = st.sidebar.radio("🎯 Model Configuration:", 
                                ["Basic Model (OHLCV only)", 
                                 "Enhanced Model (with selected features)"],
                                help="Choose between basic model or enhanced model with additional features")
    
    # Information about enhanced features
    if model_type == "Enhanced Model (with selected features)":
        with st.sidebar.expander("ℹ️ Enhanced Features Info"):
            if include_market_indices:
                st.write("**Market Indices Features:**")
                st.write("• SPX & NASDAQ close prices")
                st.write("• Relative performance vs indices")
                st.write("• Market returns & correlations")
            
            if enhance_volume_features:
                st.write("**Volume Features:**")
                st.write("• Volume moving averages (5, 20 days)")
                st.write("• Volume ratios & momentum")
                st.write("• Price-volume relationships")
            
            if include_industry_etf:
                st.write("**Industry ETF Features:**")
                st.write("• Industry ETF price & volume data")
                st.write("• Stock vs industry performance")
                st.write("• Industry correlation & beta")
                st.write("• Sector-specific context")
    
    # Load data
    # Use session state to store the dataframe across reruns
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = data_source
    
    # Reset dataframe if data source changed
    if st.session_state.data_source != data_source:
        st.session_state.df = None
        st.session_state.data_source = data_source
    
    if data_source == "Upload CSV File":
        st.subheader("📁 Upload Stock Data")
        st.info("Upload a CSV file with columns: Date, Open, High, Low, Close, Volume")
        
        # Add a key to the uploader to help Streamlit manage state
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success(f"Data loaded successfully! Shape: {st.session_state.df.shape}")
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in st.session_state.df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.session_state.df = None
                else:
                    st.sidebar.success("✅ CSV file validated successfully!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.df = None
    
    elif data_source == "Fetch Market Data (yfinance)":
        # Display ticker selection interface
        has_tickers = display_ticker_data_interface()
        
        # Show enhanced features info for real market data
        if has_tickers and model_type == "Enhanced Model (with selected features)":
            st.info("💡 **Enhanced features available with real market data:**")
            features_available = []
            if enhance_volume_features:
                features_available.append("✅ Enhanced Volume Features")
            if include_market_indices:
                features_available.append("✅ Market Indices (SPX, NASDAQ)")
            else:
                features_available.append("⚠️ Market Indices (enable in sidebar for market context)")
            if include_industry_etf:
                features_available.append("✅ Industry ETF Features")
            else:
                features_available.append("⚠️ Industry ETF Features (enable in sidebar for sector context)")
            
            for feature in features_available:
                st.write(feature)
    
    else:
        st.subheader("🎲 Generate Synthetic Data")
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.number_input("Number of Days", 500, 2000, 1000, key="num_days")
        with col2:
            start_price = st.number_input("Starting Price", 50, 500, 100, key="start_price")
        
        if st.button("Generate Data", key="generate_data"):
            with st.spinner("Generating synthetic stock data..."):
                st.session_state.df = generate_synthetic_stock_data(num_days, start_price)
                st.success(f"Synthetic data generated! Shape: {st.session_state.df.shape}")
                
                # Show a preview of available features when using enhanced model
                if model_type == "Enhanced Model (with selected features)":
                    st.info("💡 **Enhanced features available with synthetic data:**")
                    features_available = ["✅ Enhanced Volume Features"]
                    if not include_market_indices:
                        features_available.append("❌ Market Indices (not available with synthetic data)")
                    else:
                        features_available.append("❌ Market Indices (would be available with real CSV data)")
                    if not include_industry_etf:
                        features_available.append("❌ Industry ETF Features (not available with synthetic data)")
                    else:
                        features_available.append("❌ Industry ETF Features (require real ticker data)")
                    
                    for feature in features_available:
                        st.write(feature)
    
    # Check if data is available for training
    if st.session_state.df is None:
        st.info("ℹ️ Please load data first before training the model.")
    
    if st.session_state.df is not None:
        # Assign the DataFrame from session state to a local variable for convenience
        df = st.session_state.df
        
        # Debug info
        st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows")
        
        # Show current ticker info if available
        if hasattr(st.session_state, 'current_ticker'):
            st.sidebar.info(f"📊 Current Ticker: **{st.session_state.current_ticker}**")
            
            # Add industry information if industry ETF features are enabled
            if include_industry_etf:
                try:
                    ticker_info = get_ticker_industry_info(st.session_state.current_ticker)
                    if ticker_info and ticker_info.get('industry_etf'):
                        st.sidebar.info(f"🏭 Industry: **{ticker_info['industry']}**\n📈 ETF: **{ticker_info['industry_etf']}**")
                except:
                    pass
            
            # Add quick stats
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 and df['Close'].iloc[-2] != 0 else 0
            
            st.sidebar.metric(
                "Current Price", 
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        # Show current configuration
        if model_type == "Enhanced Model (with selected features)":
            config_info = "🔧 **Current Configuration:**\n"
            config_info += f"- Model: Enhanced\n"
            config_info += f"- Market Indices: {'✅' if include_market_indices else '❌'}\n"
            config_info += f"- Volume Features: {'✅' if enhance_volume_features else '❌'}\n"
            config_info += f"- Industry ETF: {'✅' if include_industry_etf else '❌'}\n"
            st.sidebar.info(config_info)
        else:
            st.sidebar.info("🔧 **Current Configuration:**\n- Model: Basic (OHLCV only)")

        # Display data preview
        st.subheader("📊 Data Preview")
        
        # Add download option and ticker info
        col1, col2 = st.columns([3, 1])
        with col1:
            if hasattr(st.session_state, 'current_ticker'):
                st.write(f"**Data for: {st.session_state.current_ticker}**")
        with col2:
            if st.button("📥 Download Data", help="Download current dataset as CSV"):
                filename = f"{st.session_state.current_ticker}_data.csv" if hasattr(st.session_state, 'current_ticker') else "stock_data.csv"
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
        
        st.dataframe(df.head(10))
        
        # Test market data fetching if enhanced features are enabled
        if st.session_state.get('show_test_market_data', False):
            st.subheader("🧪 Market Data Test")
            if st.button("Test Market Data Fetch"):
                if 'Date' in df.columns:
                    start_date = pd.to_datetime(df['Date'].min())
                    end_date = pd.to_datetime(df['Date'].max())
                    spx_data, nasdaq_data = fetch_market_indices(start_date, end_date)
                    if spx_data is not None:
                        st.success("Market data fetching working correctly!")
                        st.write("SPX sample:", spx_data.head())
                        st.write("NASDAQ sample:", nasdaq_data.head())
                    else:
                        st.error("Market data fetching failed")
                else:
                    st.warning("No Date column found for market data test")
        
        # Toggle for test mode
        if st.checkbox("🔧 Show Market Data Test", value=False):
            st.session_state['show_test_market_data'] = True
        else:
            st.session_state['show_test_market_data'] = False
        
        # Display basic statistics
        st.subheader("📈 Data Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            st.metric("Average Close", f"${df['Close'].mean():.2f}")
        with col3:
            st.metric("Min Close", f"${df['Close'].min():.2f}")
        with col4:
            st.metric("Max Close", f"${df['Close'].max():.2f}")
        
        # Plot price history
        st.subheader("📉 Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'])
        
        # Add ticker name to title if available
        title = "Historical Closing Prices"
        if hasattr(st.session_state, 'current_ticker'):
            title = f"Historical Closing Prices - {st.session_state.current_ticker}"
        
        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        
        st.pyplot(fig)
        
        # Show ticker comparison if multiple tickers were fetched
        if hasattr(st.session_state, 'all_ticker_data') and len(st.session_state.all_ticker_data) > 1:
            st.subheader("📊 Ticker Performance Comparison")
            comparison_fig = plot_ticker_comparison(
                st.session_state.all_ticker_data, 
                st.session_state.current_ticker if hasattr(st.session_state, 'current_ticker') else None
            )
            if comparison_fig:
                st.pyplot(comparison_fig)
        
        # Train model button
        if st.button("🚀 Train Transformer Model", type="primary"):
            
            with st.spinner("Preprocessing data..."):
                # Choose preprocessing method based on model type
                if model_type == "Enhanced Model (with selected features)":
                    # Get ticker symbol for industry ETF features
                    ticker_symbol = getattr(st.session_state, 'current_ticker', None)
                    
                    # Enhanced preprocessing with market context, volume features, and industry ETF
                    scaled_data, scaler_features, scaler_target, enhanced_df, feature_names = load_and_preprocess_data_enhanced(
                        df, 
                        include_indices=include_market_indices, 
                        enhance_volume=enhance_volume_features,
                        include_industry_etf=include_industry_etf,
                        ticker_symbol=ticker_symbol
                    )
                    
                    st.info(f"Using enhanced features: {', '.join(feature_names)}")
                    
                    # Display enhanced feature info
                    if include_market_indices and 'SPX_Close' in enhanced_df.columns:
                        st.success("✅ Market indices (SPX, NASDAQ) successfully integrated!")
                    elif include_market_indices:
                        st.warning("⚠️ Market indices requested but not available (using synthetic data or no date column)")
                    
                    if enhance_volume_features:
                        st.success("✅ Enhanced volume features integrated!")
                    
                    if include_industry_etf and 'Industry_ETF_Close' in enhanced_df.columns:
                        st.success(f"✅ Industry ETF features integrated for {ticker_symbol}!")
                    elif include_industry_etf and ticker_symbol:
                        st.warning(f"⚠️ Industry ETF features requested but not available for {ticker_symbol}")
                    elif include_industry_etf:
                        st.warning("⚠️ Industry ETF features requested but no ticker symbol available")
                    
                else:
                    # Basic preprocessing
                    scaled_data, scaler_features, scaler_target, enhanced_df = load_and_preprocess_data(df)
                    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
                    st.info("Using basic OHLCV features")
                
                # Create sequences
                X, y = create_sequences(scaled_data, sequence_length)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
                
                st.success(f"Data preprocessed! Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
                st.info(f"Input shape: {X.shape}, Features: {len(feature_names)}")
            
            with st.spinner("Building Transformer model..."):
                # Build model
                input_shape = (sequence_length, X.shape[-1])
                
                model = build_transformer_model(
                    input_shape=input_shape,
                    head_size=64,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    num_transformer_blocks=num_transformer_blocks,
                    mlp_units=[128, 64],
                    dropout=0.1,
                    mlp_dropout=0.1
                )
                
                # Compile model
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse",
                    metrics=["mae"]
                )
                
                st.success("Model built successfully!")
                
                # Display model summary
                with st.expander("View Model Architecture"):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text('\n'.join(model_summary))
            
            # Training progress
            st.subheader("🏋️ Model Training")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback to update Streamlit progress
            class StreamlitCallback(keras.callbacks.Callback):
                def __init__(self, progress_bar, status_text, total_epochs):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.total_epochs
                    self.progress_bar.progress(progress)
                    self.status_text.text(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs['loss']:.4f} - MAE: {logs['mae']:.4f}")
            
            # Train model
            callbacks = [
                StreamlitCallback(progress_bar, status_text, epochs),
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            st.success("Model training completed!")
            
            # Plot training history
            st.subheader("📊 Training History")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_title('Model Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['mae'], label='Training MAE')
                ax.plot(history.history['val_mae'], label='Validation MAE')
                ax.set_title('Model MAE')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MAE')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Make predictions
            st.subheader("🎯 Model Predictions")
            
            with st.spinner("Generating predictions..."):
                # Predict on test set
                y_pred_scaled = model.predict(X_test, verbose=0)
                
                # Inverse transform predictions
                y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                rmse, mae, mape = calculate_metrics(y_test_original, y_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSE", f"{rmse:.2f}")
            with col2:
                st.metric("MAE", f"{mae:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Plot predictions
            st.subheader("📈 Prediction Results")
            
            # Test set predictions
            fig = plot_predictions(y_test_original, y_pred, "Test Set: Actual vs Predicted Prices")
            st.pyplot(fig)
            
            # Training set predictions for comparison
            if st.checkbox("Show Training Set Predictions"):
                y_pred_train_scaled = model.predict(X_train, verbose=0)
                y_pred_train = scaler_target.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
                y_train_original = scaler_target.inverse_transform(y_train.reshape(-1, 1)).flatten()
                
                fig = plot_predictions(y_train_original, y_pred_train, "Training Set: Actual vs Predicted Prices")
                st.pyplot(fig)
            
            # Future prediction
            st.subheader("🔮 Next Day Prediction")
            
            # Use the last sequence to predict next day
            last_sequence = X_test[-1].reshape(1, sequence_length, -1)
            next_day_scaled = model.predict(last_sequence, verbose=0)
            next_day_price = scaler_target.inverse_transform(next_day_scaled.reshape(-1, 1))[0, 0]
            
            current_price = enhanced_df['Close'].iloc[-1]
            price_change = next_day_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${next_day_price:.2f}", f"${price_change:.2f}")
            with col3:
                st.metric("Predicted Change", f"{price_change_pct:.2f}%")
            
            # Future Price Predictions
            st.markdown("---")
            st.subheader("📈 Future Price Predictions")
            st.info("Configure the prediction settings below and click 'Generate Future Predictions' to forecast stock prices.")
            
            # Controls for future predictions
            col1, col2 = st.columns(2)
            with col1:
                future_days = st.slider("Days to Predict", 7, 90, 30, key="future_days", 
                                       help="Number of days into the future to predict")
            with col2:
                historical_context_days = st.slider("Historical Context Days", 30, 200, 100, key="context_days",
                                                   help="Number of historical days to show for context in the chart")
            
            # Make the button more prominent
            st.markdown("### 🔮 Generate Predictions")
            if st.button("🔮 Generate Future Predictions", type="secondary", use_container_width=True):
                with st.spinner(f"Predicting stock prices for the next {future_days} days..."):
                    # Get the last sequence for prediction
                    last_sequence_scaled = X_test[-1]  # This is already scaled
                    
                    # Predict future prices
                    future_predictions = predict_future_prices(
                        model, 
                        last_sequence_scaled, 
                        scaler_target, 
                        num_days=future_days
                    )
                    
                    # Get historical prices for context
                    historical_prices = enhanced_df['Close'].values
                    
                    # Plot future predictions
                    future_fig = plot_future_predictions(
                        historical_prices, 
                        future_predictions, 
                        num_historical_days=historical_context_days,
                        title=f"Stock Price Forecast - Next {future_days} Days"
                    )
                    
                    st.pyplot(future_fig)
                    
                    # Display future prediction statistics
                    st.subheader("📊 Future Prediction Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Predicted Min", f"${future_predictions.min():.2f}")
                    with col2:
                        st.metric("Predicted Max", f"${future_predictions.max():.2f}")
                    with col3:
                        st.metric("Predicted Mean", f"${future_predictions.mean():.2f}")
                    with col4:
                        volatility = np.std(future_predictions)
                        st.metric("Predicted Volatility", f"${volatility:.2f}")
                    
                    # Show trend analysis
                    trend_change = future_predictions[-1] - future_predictions[0]
                    trend_pct = (trend_change / future_predictions[0]) * 100
                    
                    if trend_change > 0:
                        st.success(f"📈 Predicted upward trend: +${trend_change:.2f} ({trend_pct:.1f}%) over {future_days} days")
                    else:
                        st.error(f"📉 Predicted downward trend: ${trend_change:.2f} ({trend_pct:.1f}%) over {future_days} days")
                    
                    # Show detailed predictions table
                    with st.expander("View Detailed Predictions"):
                        # Create future dates
                        last_date = pd.to_datetime(enhanced_df['Date'].iloc[-1]) if 'Date' in enhanced_df.columns else pd.Timestamp.now()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
                        
                        predictions_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Price': future_predictions,
                            'Day_Change': np.concatenate([[0], np.diff(future_predictions)]),
                            'Day_Change_Pct': np.concatenate([[0], np.diff(future_predictions) / future_predictions[:-1] * 100])
                        })
                        
                        # Format the dataframe for display
                        predictions_df['Predicted_Price'] = predictions_df['Predicted_Price'].apply(lambda x: f"${x:.2f}")
                        predictions_df['Day_Change'] = predictions_df['Day_Change'].apply(lambda x: f"${x:.2f}")
                        predictions_df['Day_Change_Pct'] = predictions_df['Day_Change_Pct'].apply(lambda x: f"{x:.2f}%")
                        
                        st.dataframe(predictions_df, use_container_width=True)
            
            # Model insights
            st.subheader("💡 Model Insights")
            
            # Feature analysis
            if model_type == "Enhanced Model (with selected features)":
                feature_info = f"**Enhanced Model with {len(feature_names)} features:**\n"
                feature_info += f"- Base features: {', '.join(['Open', 'High', 'Low', 'Close', 'Volume'])}\n"
                
                if enhance_volume_features:
                    volume_feats = [f for f in feature_names if 'Volume' in f and f != 'Volume']
                    if volume_feats:
                        feature_info += f"- Volume features: {', '.join(volume_feats)}\n"
                
                if include_market_indices and any('SPX' in f or 'NASDAQ' in f for f in feature_names):
                    market_feats = [f for f in feature_names if 'SPX' in f or 'NASDAQ' in f]
                    feature_info += f"- Market features: {', '.join(market_feats)}\n"
                
                if include_industry_etf and any('Industry_ETF' in f for f in feature_names):
                    industry_feats = [f for f in feature_names if 'Industry_ETF' in f or 'Industry' in f]
                    feature_info += f"- Industry ETF features: {', '.join(industry_feats)}\n"
                
                st.info(feature_info)
            
            st.info(f"""
            **Model Performance Summary:**
            - The Transformer model was trained on {len(X_train)} sequences using {sequence_length} days of historical data
            - Test set RMSE: ${rmse:.2f} (lower is better)
            - Test set MAE: ${mae:.2f} (average prediction error)
            - Test set MAPE: {mape:.2f}% (percentage error)
            
            **Architecture Details:**
            - {num_transformer_blocks} Transformer encoder blocks
            - {num_heads} attention heads
            - {ff_dim} feed-forward dimension
            - Trained for {len(history.history['loss'])} epochs
            - Input features: {len(feature_names)}
            """)
            
            if mape < 5:
                st.success("🎉 Excellent model performance! MAPE < 5%")
            elif mape < 10:
                st.warning("⚠️ Good model performance, but could be improved. MAPE < 10%")
            else:
                st.error("❌ Model performance needs improvement. Consider tuning hyperparameters.")

if __name__ == "__main__":
    main()

def fetch_ticker_data(ticker, start_date, end_date):
    """Fetch historical data for a specific ticker"""
    try:
        st.info(f"Fetching data for {ticker}...")
        
        # Add some buffer days to ensure we get data
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch ticker data
        ticker_data = yf.download(ticker, start=start_date_buffer, end=end_date_buffer, progress=False)
        
        if ticker_data.empty:
            st.error(f"No data available for ticker {ticker} in the given date range")
            return None
        
        # Reset index to make Date a column
        ticker_data = ticker_data.reset_index()
        
        # Rename columns to match expected format
        ticker_data = ticker_data.rename(columns={
            'Adj Close': 'Close'  # Use adjusted close if available
        })
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in ticker_data.columns for col in required_columns):
            ticker_data = ticker_data[required_columns]
            st.success(f"✅ Successfully fetched {len(ticker_data)} days of data for {ticker}")
            return ticker_data
        else:
            st.error(f"Missing required columns for {ticker}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def fetch_multiple_tickers(tickers, start_date, end_date):
    """Fetch data for multiple tickers"""
    try:
        st.info(f"Fetching data for multiple tickers: {', '.join(tickers)}")
        
        # Add buffer days
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch data for all tickers at once
        data = yf.download(tickers, start=start_date_buffer, end=end_date_buffer, progress=False, group_by='ticker')
        
        if data.empty:
            st.error("No data available for the specified tickers")
            return None
        
        ticker_dataframes = {}
        
        if len(tickers) == 1:
            # Single ticker case
            ticker = tickers[0]
            ticker_data = data.reset_index()
            ticker_data = ticker_data.rename(columns={'Adj Close': 'Close'})
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in ticker_data.columns for col in required_columns):
                ticker_dataframes[ticker] = ticker_data[required_columns]
        else:
            # Multiple tickers case
            for ticker in tickers:
                try:
                    ticker_data = data[ticker].reset_index()
                    ticker_data = ticker_data.rename(columns={'Adj Close': 'Close'})
                    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in ticker_data.columns for col in required_columns):
                        ticker_dataframes[ticker] = ticker_data[required_columns].dropna()
                except:
                    st.warning(f"Could not process data for {ticker}")
        
        st.success(f"✅ Successfully fetched data for {len(ticker_dataframes)} tickers")
        return ticker_dataframes
        
    except Exception as e:
        st.error(f"Error fetching multiple ticker data: {str(e)}")
        return None

def get_popular_tickers():
    """Return a list of popular stock tickers and indices"""
    return {
        "Popular Stocks": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
            "CRM", "INTC", "CSCO", "PFE", "VZ", "KO", "PEP", "T", "XOM", "CVX"
        ],
        "Market Indices": [
            "^GSPC",  # S&P 500
            "^IXIC",  # NASDAQ
            "^DJI",   # Dow Jones
            "^RUT",   # Russell 2000
            "^VIX",   # VIX
            "^TNX",   # 10-Year Treasury
            "^FTSE",  # FTSE 100
            "^N225",  # Nikkei 225
        ],
        "ETFs": [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", 
            "BND", "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLV"
        ],
        "Crypto": [
            "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOGE-USD"
        ]
    }

def create_ticker_selection_interface():
    """Create the ticker selection interface"""
    st.subheader("📈 Fetch Real Market Data")
    
    # Add ticker search widget
    create_ticker_search_widget()
    
    # Get popular tickers
    popular_tickers = get_popular_tickers()
    
    # Selection method
    selection_method = st.radio(
        "How would you like to select data?",
        ["Choose from Popular Lists", "Enter Custom Ticker(s)", "Upload Ticker List"],
        key="ticker_selection_method"
    )
    
    selected_tickers = []
    
    if selection_method == "Choose from Popular Lists":
        st.write("**Select from popular categories:**")
        
        # Create tabs for different categories
        tabs = st.tabs(list(popular_tickers.keys()))
        
        for i, (category, tickers) in enumerate(popular_tickers.items()):
            with tabs[i]:
                selected_from_category = st.multiselect(
                    f"Select {category}:",
                    tickers,
                    key=f"select_{category}"
                )
                selected_tickers.extend(selected_from_category)
    
    elif selection_method == "Enter Custom Ticker(s)":
        ticker_input = st.text_input(
            "Enter ticker symbols (comma-separated):",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter one or more ticker symbols separated by commas"
        )
        if ticker_input:
            selected_tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    elif selection_method == "Upload Ticker List":
        uploaded_tickers = st.file_uploader(
            "Upload a text file with ticker symbols (one per line)",
            type=['txt'],
            key="ticker_file"
        )
        if uploaded_tickers:
            ticker_content = uploaded_tickers.read().decode('utf-8')
            selected_tickers = [ticker.strip().upper() for ticker in ticker_content.split('\n') if ticker.strip()]
            st.write(f"Loaded {len(selected_tickers)} tickers from file")
    
    return selected_tickers

def display_ticker_data_interface():
    """Display the complete ticker data fetching interface"""
    st.markdown("---")
    
    # Ticker selection
    selected_tickers = create_ticker_selection_interface()
    
    if selected_tickers:
        st.write(f"**Selected tickers:** {', '.join(selected_tickers)}")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.now() - pd.Timedelta(days=365),
                key="ticker_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp.now(),
                key="ticker_end_date"
            )
        
        # Fetch data button
        if st.button("📊 Fetch Market Data", type="primary", use_container_width=True):
            if len(selected_tickers) == 1:
                # Single ticker
                ticker_data = fetch_ticker_data(selected_tickers[0], pd.Timestamp(start_date), pd.Timestamp(end_date))
                if ticker_data is not None:
                    st.session_state.df = ticker_data
                    st.session_state.current_ticker = selected_tickers[0]
                    st.success(f"✅ Data for {selected_tickers[0]} loaded successfully!")
                    
                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ticker", selected_tickers[0])
                    with col2:
                        st.metric("Days of Data", len(ticker_data))
                    with col3:
                        st.metric("Current Price", f"${ticker_data['Close'].iloc[-1]:.2f}")
                    
                    # Show preview
                    st.write("**Data Preview:**")
                    st.dataframe(ticker_data.head())
            
            else:
                # Multiple tickers
                ticker_dataframes = fetch_multiple_tickers(selected_tickers, pd.Timestamp(start_date), pd.Timestamp(end_date))
                if ticker_dataframes:
                    # Let user choose which ticker to use for training
                    st.success(f"✅ Fetched data for {len(ticker_dataframes)} tickers")
                    
                    available_tickers = list(ticker_dataframes.keys())
                    selected_for_training = st.selectbox(
                        "Select ticker for model training:",
                        available_tickers,
                        key="training_ticker"
                    )
                    
                    if selected_for_training:
                        st.session_state.df = ticker_dataframes[selected_for_training]
                        st.session_state.current_ticker = selected_for_training
                        st.session_state.all_ticker_data = ticker_dataframes
                        
                        # Display comparison metrics
                        st.write("**Ticker Comparison:**")
                        comparison_data = []
                        for ticker, data in ticker_dataframes.items():
                            comparison_data.append({
                                'Ticker': ticker,
                                'Days': len(data),
                                'Current Price': f"${data['Close'].iloc[-1]:.2f}",
                                'Price Change (%)': f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%",
                                'Avg Volume': f"{data['Volume'].mean():,.0f}"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Show preview of selected ticker
                        st.write(f"**Data Preview for {selected_for_training}:**")
                        st.dataframe(ticker_dataframes[selected_for_training].head())
                        
                        # Plot comparison chart
                        st.subheader("📊 Ticker Performance Comparison")
                        fig = plot_ticker_comparison(ticker_dataframes, selected_for_training)
                        st.pyplot(fig)
    
    return len(selected_tickers) > 0 if selected_tickers else False

def plot_ticker_comparison(ticker_dataframes, selected_ticker=None):
    """Plot comparison of multiple tickers"""
    if len(ticker_dataframes) <= 1:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize prices to starting value for comparison
    for ticker, data in ticker_dataframes.items():
        normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
        line_style = '-' if ticker == selected_ticker else '--'
        line_width = 2.5 if ticker == selected_ticker else 1.5
        alpha = 1.0 if ticker == selected_ticker else 0.7
        
        ax.plot(range(len(normalized_prices)), normalized_prices, 
               label=ticker, linestyle=line_style, linewidth=line_width, alpha=alpha)
    
    ax.set_title("Ticker Performance Comparison (Normalized to 100)", fontsize=16, fontweight='bold')
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Normalized Price (Starting Price = 100)', fontsize=12)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Highlight selected ticker
    if selected_ticker:
        ax.text(0.02, 0.98, f"Training on: {selected_ticker}", 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               verticalalignment='top')
    
    plt.tight_layout()
    return fig

def search_ticker_info(ticker):
    """Get basic info about a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Currency': info.get('currency', 'N/A')
        }
    except:
        return None

def create_ticker_search_widget():
    """Create a ticker search widget"""
    with st.expander("🔍 Ticker Search & Info"):
        search_ticker = st.text_input("Search for ticker information:", placeholder="e.g., AAPL")
        
        if search_ticker and st.button("Search"):
            ticker_info = search_ticker_info(search_ticker.upper())
            if ticker_info:
                st.success(f"Found information for {search_ticker.upper()}:")
                for key, value in ticker_info.items():
                    if value != 'N/A':
                        if key == 'Market Cap' and isinstance(value, (int, float)):
                            value = f"${value:,.0f}"
                        st.write(f"**{key}:** {value}")
            else:
                st.error(f"Could not find information for {search_ticker.upper()}")

def get_industry_etf_mapping():
    """Map industries to their corresponding ETFs"""
    return {
        # Technology
        'Software': 'XLK',
        'Semiconductors': 'SOXX',
        'Technology Hardware': 'XLK',
        'Computer Hardware': 'XLK',
        'Internet Content & Information': 'XLK',
        'Electronic Gaming & Multimedia': 'XLK',
        'Software - Application': 'XLK',
        'Software - Infrastructure': 'XLK',
        
        # Healthcare
        'Biotechnology': 'XBI',
        'Drug Manufacturers': 'XLV',
        'Medical Devices': 'XLV',
        'Healthcare': 'XLV',
        'Pharmaceuticals': 'XLV',
        'Medical Instruments & Supplies': 'XLV',
        'Health Information Services': 'XLV',
        
        # Financial
        'Banks': 'XLF',
        'Financial Services': 'XLF',
        'Insurance': 'XLF',
        'Asset Management': 'XLF',
        'Credit Services': 'XLF',
        'Capital Markets': 'XLF',
        
        # Energy
        'Oil & Gas': 'XLE',
        'Oil & Gas E&P': 'XLE',
        'Oil & Gas Integrated': 'XLE',
        'Oil & Gas Refining & Marketing': 'XLE',
        'Oil & Gas Equipment & Services': 'XLE',
        'Renewable Energy': 'ICLN',
        
        # Consumer
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Retail': 'XRT',
        'Automotive': 'CARZ',
        'Restaurants': 'XLY',
        'Apparel Manufacturing': 'XLY',
        'Home Improvement Retail': 'XLY',
        'Internet Retail': 'XLY',
        
        # Industrial
        'Industrials': 'XLI',
        'Aerospace & Defense': 'ITA',
        'Airlines': 'JETS',
        'Transportation': 'XTN',
        'Construction': 'XLI',
        'Machinery': 'XLI',
        
        # Materials
        'Materials': 'XLB',
        'Steel': 'SLX',
        'Mining': 'XME',
        'Chemicals': 'XLB',
        'Construction Materials': 'XLB',
        
        # Utilities
        'Utilities': 'XLU',
        'Electric Utilities': 'XLU',
        'Gas Utilities': 'XLU',
        'Water Utilities': 'XLU',
        
        # Real Estate
        'Real Estate': 'XLRE',
        'REIT': 'VNQ',
        
        # Communication
        'Communication Services': 'XLC',
        'Telecommunications': 'XLC',
        'Media': 'XLC',
        'Entertainment': 'XLC',
        
        # Default fallback
        'Other': 'SPY'  # Use SPY as default
    }

def get_ticker_industry_info(ticker):
    """Get detailed industry information for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract industry information
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Map to ETF
        industry_etf_map = get_industry_etf_mapping()
        
        # Try to find the best matching ETF
        industry_etf = None
        for key, etf in industry_etf_map.items():
            if key.lower() in industry.lower() or key.lower() in sector.lower():
                industry_etf = etf
                break
        
        # Fallback to sector-based mapping
        if not industry_etf:
            sector_etf_map = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Energy': 'XLE',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }
            industry_etf = sector_etf_map.get(sector, 'SPY')
        
        return {
            'ticker': ticker,
            'company_name': info.get('longName', 'N/A'),
            'sector': sector,
            'industry': industry,
            'industry_etf': industry_etf,
            'market_cap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'N/A')
        }
        
    except Exception as e:
        st.warning(f"Could not fetch industry info for {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'company_name': 'N/A',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'industry_etf': 'SPY',  # Default to SPY
            'market_cap': 'N/A',
            'currency': 'N/A'
        }

def fetch_industry_etf_data(industry_etf, start_date, end_date):
    """Fetch industry-specific ETF data"""
    try:
        st.info(f"Fetching industry ETF data for {industry_etf}...")
        
        # Add buffer days
        start_date_buffer = start_date - pd.Timedelta(days=10)
        end_date_buffer = end_date + pd.Timedelta(days=5)
        
        # Fetch ETF data
        etf_data = yf.download(industry_etf, start=start_date_buffer, end=end_date_buffer, progress=False)
        
        if etf_data.empty:
            st.warning(f"No data available for industry ETF {industry_etf}")
            return None
        
        # Reset index and rename columns
        etf_data = etf_data.reset_index()
        etf_data = etf_data.rename(columns={
            'Close': f'{industry_etf}_Close',
            'Volume': f'{industry_etf}_Volume'
        })
        
        # Select relevant columns
        etf_data = etf_data[['Date', f'{industry_etf}_Close', f'{industry_etf}_Volume']]
        
        st.success(f"✅ Successfully fetched industry ETF data for {industry_etf}")
        return etf_data
        
    except Exception as e:
        st.error(f"Error fetching industry ETF data for {industry_etf}: {str(e)}")
        return None

def add_industry_context_features(df, ticker_info, include_industry_etf=True):
    """Add industry-specific ETF features to the dataset"""
    enhanced_df = df.copy()
    
    if not include_industry_etf or not ticker_info:
        return enhanced_df
    
    if 'Date' in enhanced_df.columns:
        enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date'])
        start_date = enhanced_df['Date'].min()
        end_date = enhanced_df['Date'].max()
        
        industry_etf = ticker_info.get('industry_etf', 'SPY')
        
        # Fetch industry ETF data
        etf_data = fetch_industry_etf_data(industry_etf, start_date, end_date)
        
        if etf_data is not None:
            # Merge with main dataframe
            enhanced_df = enhanced_df.merge(etf_data, on='Date', how='left')
            
            etf_close_col = f'{industry_etf}_Close'
            etf_volume_col = f'{industry_etf}_Volume'
            
            # Forward fill missing values
            enhanced_df[etf_close_col] = enhanced_df[etf_close_col].fillna(method='ffill').fillna(enhanced_df[etf_close_col].bfill())
            enhanced_df[etf_volume_col] = enhanced_df[etf_volume_col].fillna(method='ffill').fillna(enhanced_df[etf_volume_col].bfill())
            
            # Create industry-relative features
            enhanced_df[f'Stock_vs_{industry_etf}'] = enhanced_df['Close'] / enhanced_df[etf_close_col]
            enhanced_df[f'{industry_etf}_Returns'] = enhanced_df[etf_close_col].pct_change()
            enhanced_df['Stock_Returns'] = enhanced_df['Close'].pct_change()
            
            # Industry correlation features
            enhanced_df[f'{industry_etf}_Correlation'] = enhanced_df['Stock_Returns'].rolling(window=20).corr(enhanced_df[f'{industry_etf}_Returns'])
            
            # Industry momentum features
            enhanced_df[f'{industry_etf}_Momentum'] = enhanced_df[etf_close_col].pct_change(periods=5)
            enhanced_df['Stock_Industry_Beta'] = enhanced_df['Stock_Returns'].rolling(window=30).cov(enhanced_df[f'{industry_etf}_Returns']) / enhanced_df[f'{industry_etf}_Returns'].rolling(window=30).var()
            
            # Volume comparison with industry
            enhanced_df[f'Volume_vs_{industry_etf}'] = enhanced_df['Volume'] / enhanced_df[etf_volume_col]
            
    return enhanced_df
