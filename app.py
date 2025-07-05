import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import math

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
        season_output = tf.reduce_mean(stacked_seasonal_outputs, axis=-1)
        
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

def build_galformer_model(input_shape, d_model=512, n_heads=8, e_layers=2, d_layers=1, 
                         d_ff=2048, factor=1, dropout=0.1, label_len=48, pred_len=24):
    """Build complete Galformer model for stock prediction"""
    
    # Input layers
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    decoder_inputs = keras.Input(shape=(label_len + pred_len, input_shape[-1]), name='decoder_input')
    
    # Data embedding for encoder
    enc_embedding = DataEmbedding(input_shape[-1], d_model, dropout)
    enc_out = enc_embedding(encoder_inputs)
    
    # Data embedding for decoder  
    dec_embedding = DataEmbedding(input_shape[-1], d_model, dropout)
    dec_out = dec_embedding(decoder_inputs)
    
    # Encoder layers
    encoder_layers = []
    for _ in range(e_layers):
        encoder_layers.append(EncoderLayer(d_model, n_heads, d_ff, dropout, factor=factor))
    
    encoder = Encoder(encoder_layers, norm_layer=layers.LayerNormalization(epsilon=1e-6))
    enc_out = encoder(enc_out)
    
    # Decoder layers
    decoder_layers = []
    for _ in range(d_layers):
        decoder_layers.append(DecoderLayer(d_model, n_heads, d_ff, dropout, factor=factor))
    
    decoder = Decoder(decoder_layers, norm_layer=layers.LayerNormalization(epsilon=1e-6))
    dec_out = decoder(dec_out, enc_out)
    
    # Final projection layer
    outputs = layers.Dense(1, name='price_prediction')(dec_out[:, -pred_len:, :])
    outputs = tf.squeeze(outputs, axis=-1)  # Remove last dimension
    
    return keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

def create_sequences_for_galformer(data, sequence_length, label_len=48, pred_len=24, target_column_index=3):
    """Create sequences for Galformer (Encoder-Decoder) model"""
    encoder_inputs, decoder_inputs, targets = [], [], []
    
    total_len = sequence_length + pred_len
    
    for i in range(sequence_length, len(data) - pred_len + 1):
        # Encoder input: historical sequence
        enc_input = data[i - sequence_length:i]
        encoder_inputs.append(enc_input)
        
        # Decoder input: label_len from history + pred_len zeros (teacher forcing)
        dec_input_hist = data[i - label_len:i]  # Historical part
        dec_input_pred = np.zeros((pred_len, data.shape[1]))  # Future part (will be predicted)
        dec_input = np.concatenate([dec_input_hist, dec_input_pred], axis=0)
        decoder_inputs.append(dec_input)
        
        # Target: next pred_len close prices
        target = data[i:i + pred_len, target_column_index]
        targets.append(target)
    
    return np.array(encoder_inputs), np.array(decoder_inputs), np.array(targets)

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

def plot_galformer_predictions(y_true, y_pred, title="Galformer: Actual vs Predicted Prices"):
    """Plot predictions from Galformer model"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot each sequence prediction
    for i in range(min(5, len(y_true))):  # Plot first 5 sequences
        start_idx = i * len(y_true[i])
        true_seq = y_true[i]
        pred_seq = y_pred[i]
        
        x_range = range(start_idx, start_idx + len(true_seq))
        
        if i == 0:
            ax.plot(x_range, true_seq, 'b-', alpha=0.7, label='Actual', linewidth=2)
            ax.plot(x_range, pred_seq, 'r--', alpha=0.7, label='Predicted', linewidth=2)
        else:
            ax.plot(x_range, true_seq, 'b-', alpha=0.7, linewidth=2)
            ax.plot(x_range, pred_seq, 'r--', alpha=0.7, linewidth=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
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
            "CRM", "INTC", "CSCO", "PFE", "VZ", "KO", "PEP", "T", "XOM", "CVX",
            "SOFI", "PLTR", "COIN", "RBLX", "HOOD", "RIVN", "LCID", "F", "GM"
        ],
        "Growth Stocks": [
            "SOFI", "PLTR", "COIN", "RBLX", "HOOD", "RIVN", "LCID", "UPST",
            "AFRM", "SQ", "SHOP", "ROKU", "ZOOM", "SNOW", "CRWD", "NET"
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
            "BND", "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLV",
            "ARKK", "ARKQ", "ARKG", "SOXL", "TQQQ", "SPXL"
        ],
        "Crypto": [
            "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "MATIC-USD"
        ]
    }

def create_ticker_selection_interface():
    """Create interface for ticker selection"""
    st.subheader("📊 Fetch Market Data")
    st.info("Select tickers to fetch historical stock data from Yahoo Finance")
    
    # Create tabs for different selection methods
    tab1, tab2, tab3 = st.tabs(["📈 Popular Lists", "✏️ Custom Entry", "📁 File Upload"])
    
    with tab1:
        st.write("**Select from popular categories:**")
        popular_tickers = get_popular_tickers()
        
        selected_category = st.selectbox("Choose Category:", list(popular_tickers.keys()))
        selected_tickers = st.multiselect(
            f"Select {selected_category}:",
            popular_tickers[selected_category],
            default=popular_tickers[selected_category][:3] if selected_category == "Popular Stocks" else []
        )
        
        return selected_tickers
    
    with tab2:
        st.write("**Enter ticker symbols manually:**")
        custom_input = st.text_input(
            "Ticker Symbols (comma-separated):",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter one or more ticker symbols separated by commas"
        )
        
        selected_tickers = []
        if custom_input:
            selected_tickers = [ticker.strip().upper() for ticker in custom_input.split(",") if ticker.strip()]
        
        return selected_tickers
    
    with tab3:
        st.write("**Upload a text file with ticker symbols:**")
        uploaded_file = st.file_uploader(
            "Choose a text file", 
            type=['txt'],
            help="Upload a .txt file with one ticker symbol per line"
        )
        
        selected_tickers = []
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                selected_tickers = [line.strip().upper() for line in content.split('\n') if line.strip()]
                st.success(f"Loaded {len(selected_tickers)} tickers from file")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        return selected_tickers

def display_ticker_data_interface():
    """Main interface for ticker data selection and fetching"""
    
    # Add ticker search widget
    create_ticker_search_widget()
    
    # Date range selection
    st.subheader("📅 Select Date Range")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=1)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=datetime.now() - timedelta(days=1),
            min_value=start_date,
            max_value=datetime.now()
        )
    
    # Ticker selection
    selected_tickers = create_ticker_selection_interface()
    
    if not selected_tickers:
        st.warning("Please select at least one ticker symbol.")
        return False
    
    st.write(f"**Selected tickers:** {', '.join(selected_tickers)}")
    
    # Fetch data button
    if st.button("📊 Fetch Market Data", type="primary"):
        with st.spinner(f"Fetching data for {len(selected_tickers)} ticker(s)..."):
            
            if len(selected_tickers) == 1:
                # Single ticker
                ticker = selected_tickers[0]
                ticker_data = fetch_ticker_data(ticker, start_date, end_date)
                
                if ticker_data is not None:
                    st.session_state.df = ticker_data
                    st.session_state.current_ticker = ticker
                    st.success(f"✅ Data loaded for {ticker}")
                    return True
                else:
                    st.error(f"Failed to fetch data for {ticker}")
                    return False
            
            else:
                # Multiple tickers
                ticker_dataframes = fetch_multiple_tickers(selected_tickers, start_date, end_date)
                
                if ticker_dataframes and len(ticker_dataframes) > 0:
                    st.session_state.all_ticker_data = ticker_dataframes
                    
                    # Let user choose which ticker to use for training
                    st.subheader("📊 Select Ticker for Model Training")
                    
                    # Show comparison chart first
                    comparison_fig = plot_ticker_comparison(ticker_dataframes)
                    if comparison_fig:
                        st.pyplot(comparison_fig)
                    
                    # Ticker selection for training
                    available_tickers = list(ticker_dataframes.keys())
                    selected_ticker = st.selectbox(
                        "Choose ticker for model training:",
                        available_tickers,
                        help="This ticker will be used to train the prediction model"
                    )
                    
                    if st.button("Use Selected Ticker", type="secondary"):
                        st.session_state.df = ticker_dataframes[selected_ticker]
                        st.session_state.current_ticker = selected_ticker
                        st.success(f"✅ Using {selected_ticker} for model training")
                        return True
                    
                    return True  # Data is available, even if not selected yet
                else:
                    st.error("Failed to fetch data for any of the selected tickers")
                    return False
    
    # Check if data is already loaded
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        return True
    
    return False

def plot_ticker_comparison(ticker_dataframes, highlight_ticker=None):
    """Plot comparison of multiple tickers"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize prices to start at 100 for comparison
        for ticker, data in ticker_dataframes.items():
            if len(data) > 0 and 'Close' in data.columns:
                normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
                
                if ticker == highlight_ticker:
                    ax.plot(normalized_prices.index, normalized_prices, 
                           label=ticker, linewidth=3, alpha=0.9)
                else:
                    ax.plot(normalized_prices.index, normalized_prices, 
                           label=ticker, linewidth=2, alpha=0.7)
        
        ax.set_title("Ticker Performance Comparison (Normalized to 100)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel("Normalized Price", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def train_test_split_temporal(X, y, test_size=0.2):
    """Split time series data maintaining temporal order"""
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_test_split_temporal_galformer(X_enc, X_dec, y, test_size=0.2):
    """Split Galformer data maintaining temporal order"""
    split_idx = int(len(X_enc) * (1 - test_size))
    
    X_enc_train = X_enc[:split_idx]
    X_enc_test = X_enc[split_idx:]
    X_dec_train = X_dec[:split_idx]
    X_dec_test = X_dec[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_enc_train, X_enc_test, X_dec_train, X_dec_test, y_train, y_test

# Industry ETF functions
def get_industry_etf_mapping():
    """Return mapping of industries to their corresponding ETFs"""
    return {
        'Technology': 'XLK',
        'Software': 'XLK',
        'Semiconductors': 'SOXX',
        'Internet': 'FDN',
        'Communication Services': 'XLC',
        'Healthcare': 'XLV',
        'Biotechnology': 'XBI',
        'Pharmaceuticals': 'XLV',
        'Financial Services': 'XLF',
        'Banks': 'KBE',
        'Insurance': 'KIE',
        'Real Estate': 'XLRE',
        'Energy': 'XLE',
        'Oil & Gas': 'XLE',
        'Utilities': 'XLU',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Retail': 'XRT',
        'Materials': 'XLB',
        'Industrial': 'XLI',
        'Transportation': 'IYT',
        'Aerospace': 'ITA',
        'Defense': 'ITA',
        'Gold': 'GLD',
        'Silver': 'SLV'
    }

def get_ticker_industry_info(ticker):
    """Get industry information for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        industry = info.get('industry', 'Unknown')
        sector = info.get('sector', 'Unknown')
        
        # Map to ETF
        etf_mapping = get_industry_etf_mapping()
        industry_etf = None
        
        # Try to find ETF by industry first, then by sector
        for key, etf in etf_mapping.items():
            if key.lower() in industry.lower() or key.lower() in sector.lower():
                industry_etf = etf
                break
        
        # Default to SPY if no specific industry ETF found
        if not industry_etf:
            industry_etf = 'SPY'
        
        return {
            'industry': industry,
            'sector': sector,
            'industry_etf': industry_etf
        }
    except:
        return {
            'industry': 'Unknown',
            'sector': 'Unknown', 
            'industry_etf': 'SPY'
        }

def fetch_industry_etf_data(etf_ticker, start_date, end_date):
    """Fetch industry ETF data"""
    try:
        etf_data = yf.download(etf_ticker, start=start_date, end=end_date, progress=False)
        if not etf_data.empty:
            etf_data = etf_data.reset_index()
            return etf_data[['Date', 'Close', 'Volume']].rename(columns={
                'Close': f'Industry_ETF_Close',
                'Volume': f'Industry_ETF_Volume'
            })
    except:
        pass
    return None

def add_industry_context_features(df, ticker_info, include_industry_etf=True):
    """Add industry ETF context features"""
    if not include_industry_etf:
        return df
    
    enhanced_df = df.copy()
    
    # Get date range
    start_date = enhanced_df['Date'].min()
    end_date = enhanced_df['Date'].max()
    
    # Fetch industry ETF data
    industry_etf = ticker_info['industry_etf']
    etf_data = fetch_industry_etf_data(industry_etf, start_date, end_date)
    
    if etf_data is not None:
        # Merge with main dataframe
        enhanced_df = enhanced_df.merge(etf_data, on='Date', how='left')
        
        # Forward fill missing values
        enhanced_df['Industry_ETF_Close'] = enhanced_df['Industry_ETF_Close'].fillna(method='ffill').fillna(method='bfill')
        enhanced_df['Industry_ETF_Volume'] = enhanced_df['Industry_ETF_Volume'].fillna(method='ffill').fillna(method='bfill')
        
        # Create relative performance features
        enhanced_df['Stock_vs_Industry_ETF'] = enhanced_df['Close'] / enhanced_df['Industry_ETF_Close']
        
        # Industry ETF momentum features
        enhanced_df['Industry_ETF_Returns'] = enhanced_df['Industry_ETF_Close'].pct_change()
        enhanced_df['Stock_Returns'] = enhanced_df['Close'].pct_change()
        
        # Rolling correlation with industry
        enhanced_df['Stock_Industry_Correlation'] = enhanced_df['Stock_Returns'].rolling(window=20).corr(enhanced_df['Industry_ETF_Returns'])
        
        # Beta calculation (simplified)
        enhanced_df['Stock_Industry_Beta'] = enhanced_df['Stock_Returns'].rolling(window=60).cov(enhanced_df['Industry_ETF_Returns']) / enhanced_df['Industry_ETF_Returns'].rolling(window=60).var()
        
        # Volume comparison
        enhanced_df['Industry_ETF_Relative_Volume'] = enhanced_df['Industry_ETF_Volume'] / enhanced_df['Industry_ETF_Volume'].rolling(window=20).mean()
        
        st.success(f"Added industry ETF features using {industry_etf} for {ticker_info['industry']}")
    else:
        st.warning(f"Could not fetch data for industry ETF {industry_etf}")
    
    return enhanced_df

# Add the complete main function
def main():
    st.set_page_config(
        page_title="🚀 Advanced Stock Price Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Advanced Stock Price Prediction with Transformers")
    st.markdown("**Predict stock prices using state-of-the-art Transformer and Galformer models**")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model architecture selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ Model Architecture")
    model_architecture = st.sidebar.radio(
        "Choose Model Architecture:",
        ["Standard Transformer", "Galformer (Advanced)"],
        help="Galformer uses autocorrelation and is better for capturing seasonal patterns"
    )
    
    # Data source selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose your data source:",
        ["Generate Synthetic Data", "Upload CSV File", "Fetch Market Data (yfinance)"]
    )
    
    # Enhanced features selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Enhanced Features")
    model_type = st.sidebar.radio(
        "Model Type:",
        ["Basic Model (OHLCV only)", "Enhanced Model (with selected features)"]
    )
    
    # Enhanced features options
    include_market_indices = False
    enhance_volume_features = False
    include_industry_etf = False
    
    if model_type == "Enhanced Model (with selected features)":
        include_market_indices = st.sidebar.checkbox(
            "Include SPX & NASDAQ indices",
            value=True,
            help="Add market context using S&P 500 and NASDAQ data"
        )
        enhance_volume_features = st.sidebar.checkbox(
            "Enhanced Volume Features",
            value=True,
            help="Add volume analysis features (moving averages, ratios, momentum)"
        )
        include_industry_etf = st.sidebar.checkbox(
            "Industry ETF Features",
            value=False,
            help="Add industry-specific ETF data for sector context"
        )
    
    # Model parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Model Parameters")
    
    sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 60, help="Number of days to look back")
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    if model_architecture == "Standard Transformer":
        num_heads = st.sidebar.selectbox("Number of Attention Heads", [4, 8, 16], index=1)
        num_transformer_blocks = st.sidebar.slider("Transformer Blocks", 1, 6, 2)
        ff_dim = st.sidebar.selectbox("Feed Forward Dimension", [64, 128, 256, 512], index=2)
    else:  # Galformer
        st.sidebar.subheader("Galformer Parameters")
        d_model = st.sidebar.selectbox("Model Dimension", [256, 512, 768], index=1)
        num_heads = st.sidebar.selectbox("Number of Attention Heads", [4, 8, 16], index=1)
        factor = st.sidebar.slider("Autocorrelation Factor", 1, 5, 3)
        label_len = st.sidebar.slider("Label Length", 24, 72, 48)
        pred_len = st.sidebar.slider("Prediction Length", 12, 48, 24)
        e_layers = st.sidebar.slider("Encoder Layers", 1, 4, 2)
        d_layers = st.sidebar.slider("Decoder Layers", 1, 2, 1)
        ff_dim = st.sidebar.selectbox("Feed Forward Dimension", [1024, 2048, 4096], index=1)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Status")
    if st.session_state.df is not None:
        st.sidebar.success("✅ Data loaded and ready!")
        if st.session_state.current_ticker:
            st.sidebar.info(f"Current ticker: {st.session_state.current_ticker}")
    else:
        st.sidebar.warning("⚠️ No data loaded")
    
    # Data loading section
    st.markdown("---")
    
    # Handle different data sources
    if data_source == "Generate Synthetic Data":
        st.subheader("🎲 Generate Synthetic Stock Data")
        
        col1, col2 = st.columns(2)
        with col1:
            num_days = st.number_input("Number of days", 100, 5000, 1000)
        with col2:
            start_price = st.number_input("Starting price ($)", 10, 1000, 100)
        
        if st.button("🎲 Generate Data", type="primary"):
            with st.spinner("Generating synthetic stock data..."):
                st.session_state.df = generate_synthetic_stock_data(num_days, start_price)
                st.session_state.current_ticker = "SYNTHETIC"
                st.success(f"✅ Generated {len(st.session_state.df)} days of synthetic stock data")
    
    elif data_source == "Upload CSV File":
        st.subheader("📁 Upload CSV File")
        st.info("Upload a CSV file with columns: Date, Open, High, Low, Close, Volume")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your CSV has the following columns: Date, Open, High, Low, Close, Volume")
                else:
                    st.session_state.df = df
                    st.session_state.current_ticker = "UPLOADED_CSV"
                    st.success(f"✅ Successfully loaded {len(df)} rows of data")
                    
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
    
    elif data_source == "Fetch Market Data (yfinance)":
        has_tickers = display_ticker_data_interface()
    
    # Show data preview and proceed with training if data is available
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Show basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(df))
        with col2:
            st.metric("Latest Price", f"${df['Close'].iloc[-1]:.2f}")
        with col3:
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            st.metric("Daily Change", f"${price_change:.2f}")
        with col4:
            st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
        
        # Show data sample
        st.write("**Data Sample:**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Show price chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'], label='Close Price')
        ax.set_title('Stock Price History')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Enhanced features information
        if model_type == "Enhanced Model (with selected features)":
            with st.expander("ℹ️ Enhanced Features Information"):
                feature_info = "**Enhanced model will include:**\n"
                feature_info += "- Base OHLCV features\n"
                
                if enhance_volume_features:
                    feature_info += "- Enhanced volume features (moving averages, ratios, momentum)\n"
                
                if include_market_indices and data_source == "Fetch Market Data (yfinance)":
                    feature_info += "- Market indices data (SPX, NASDAQ correlations)\n"
                elif include_market_indices:
                    feature_info += "- Market indices (not available for synthetic/uploaded data)\n"
                
                if include_industry_etf and data_source == "Fetch Market Data (yfinance)" and st.session_state.current_ticker:
                    feature_info += f"- Industry ETF features for {st.session_state.current_ticker}\n"
                elif include_industry_etf:
                    feature_info += "- Industry ETF features (only available for real tickers)\n"
                
                st.info(feature_info)
        
        # Training section
        st.markdown("---")
        st.subheader("🚀 Train Transformer Model")
        
        if st.button("🚀 Train Transformer Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                
                # Data preprocessing
                if model_type == "Enhanced Model (with selected features)":
                    # Only use enhanced features for real market data
                    use_indices = include_market_indices and data_source == "Fetch Market Data (yfinance)"
                    use_industry = include_industry_etf and data_source == "Fetch Market Data (yfinance)" and st.session_state.current_ticker
                    
                    scaled_data, scaler_features, scaler_target, enhanced_df, feature_names = load_and_preprocess_data_enhanced(
                        df, 
                        include_indices=use_indices, 
                        enhance_volume=enhance_volume_features,
                        include_industry_etf=use_industry,
                        ticker_symbol=st.session_state.current_ticker if st.session_state.current_ticker not in ['SYNTHETIC', 'UPLOADED_CSV'] else None
                    )
                    
                    st.success(f"Enhanced preprocessing complete! Using {len(feature_names)} features: {', '.join(feature_names)}")
                else:
                    scaled_data, scaler_features, scaler_target, enhanced_df = load_and_preprocess_data(df)
                    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
                    st.success("Basic preprocessing complete! Using standard OHLCV features.")
                
                # Create sequences based on model architecture
                if model_architecture == "Galformer (Advanced)":
                    # Galformer sequences
                    X_enc, X_dec, y = create_sequences_for_galformer(
                        scaled_data,
                        sequence_length,
                        label_len,
                        pred_len,
                        target_column_index=feature_names.index('Close')
                    )
                    # Split data for Galformer
                    X_enc_train, X_enc_test, X_dec_train, X_dec_test, y_train, y_test = \
                        train_test_split_temporal_galformer(X_enc, X_dec, y)
                else:
                    # Standard Transformer sequences
                    X, y = create_sequences(scaled_data, sequence_length)
                    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
                    
                    st.success(f"Standard Transformer data prepared! Training sequences: {len(X_train)}")
                    
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
                    
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae']
                    )
                    
                    st.success("Standard Transformer model built successfully!")
                    
                    # Train model
                    callbacks = [
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
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_test, verbose=0)
                    y_pred_original = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                st.success("Model training completed!")
                
                # Calculate metrics
                rmse, mae, mape = calculate_metrics(y_test_original, y_pred_original)
                
                # Display results
                st.subheader("📊 Training Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col2:
                    st.metric("MAE", f"{mae:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Plot results
                if model_architecture == "Galformer (Advanced)":
                    fig = plot_galformer_predictions(y_test_original.reshape(-1, 1), y_pred_original.reshape(-1, 1))
                else:
                    fig = plot_predictions(y_test_original, y_pred_original)
                
                st.pyplot(fig)
                
                # Future predictions
                st.subheader("🔮 Future Price Predictions")
                
                col1, col2 = st.columns(2)
                with col1:
                    future_days = st.slider("Days to Predict", 7, 90, 30)
                with col2:
                    historical_context_days = st.slider("Historical Context Days", 30, 200, 100)
                
                if st.button("🔮 Generate Future Predictions", type="secondary"):
                    with st.spinner(f"Predicting next {future_days} days..."):
                        if model_architecture == "Standard Transformer":
                            last_sequence_scaled = X_test[-1]
                            future_predictions = predict_future_prices(
                                model, last_sequence_scaled, scaler_target, num_days=future_days
                            )
                        else: # Galformer Future Prediction Logic
                            # For Galformer, use the last available sequence from the training data
                            last_enc_sequence = X_enc_test[-1:] 
                            
                            # The decoder input needs the last `label_len` part of the encoder sequence
                            # plus zeros for the part to be predicted.
                            dec_input_hist = last_enc_sequence[0, -label_len:, :]
                            dec_input_pred = np.zeros((pred_len, dec_input_hist.shape[1]))
                            last_dec_sequence = np.concatenate([dec_input_hist, dec_input_pred], axis=0)
                            last_dec_sequence = np.expand_dims(last_dec_sequence, axis=0)

                            # Predict `pred_len` steps at once
                            future_pred_scaled = model.predict([last_enc_sequence, last_dec_sequence], verbose=0)
                            
                            # Inverse transform the predictions
                            # We take the first `future_days` from the predicted sequence
                            num_predictions = min(future_days, pred_len)
                            future_predictions = scaler_target.inverse_transform(
                                future_pred_scaled[0][:num_predictions].reshape(-1, 1)
                            ).flatten()
                        
                        historical_prices = enhanced_df['Close'].values
                        
                        future_fig, stats = plot_future_predictions(
                            historical_prices, 
                            future_predictions,
                            num_historical_days=historical_context_days,
                            title=f"{model_architecture} - Next {future_days} Days Forecast"
                        )
                        
                        st.pyplot(future_fig)
                        
                        # Future prediction stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Predicted Min", f"${future_predictions.min():.2f}")
                        with col2:
                            st.metric("Predicted Max", f"${future_predictions.max():.2f}")
                        with col3:
                            st.metric("Predicted Mean", f"${stats['mean']:.2f}")
                        with col4:
                            st.metric("Volatility", f"${stats['volatility']:.2f}")
    
    else:
        st.info("👆 Please select a data source and load data to begin training.")

if __name__ == "__main__":
    main()
