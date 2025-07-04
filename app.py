import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PositionalEmbedding(layers.Layer):
    """Custom positional embedding layer for Transformer"""
    
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        
        # Create positional encoding matrix
        batch_size = tf.shape(inputs)[0]
        d_model = tf.shape(inputs)[-1]
        
        # Simple learned positional embedding
        position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.sequence_length, d_model),
            initializer="uniform",
            trainable=True
        )
        
        return inputs + position_embedding

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

# Streamlit App
def main():
    st.title("🔮 Stock Price Prediction with Transformer Model")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Upload CSV File", "Generate Synthetic Data"]
    )
    
    # Model hyperparameters
    st.sidebar.subheader("Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60)
    num_heads = st.sidebar.selectbox("Number of Attention Heads", [4, 8, 12], index=1)
    num_transformer_blocks = st.sidebar.slider("Transformer Blocks", 1, 4, 2)
    ff_dim = st.sidebar.selectbox("Feed Forward Dimension", [64, 128, 256], index=1)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    
    # Load data
    # Use session state to store the dataframe across reruns
    if 'df' not in st.session_state:
        st.session_state.df = None
    
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
                    return
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.df = None
                return
    
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
    
    if st.session_state.df is not None:
        # Assign the DataFrame from session state to a local variable for convenience
        df = st.session_state.df

        # Display data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10))
        
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
        ax.set_title("Historical Closing Prices")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Train model button
        if st.button("🚀 Train Transformer Model", type="primary"):
            
            with st.spinner("Preprocessing data..."):
                # Preprocess data
                scaled_data, scaler_features, scaler_target, original_df = load_and_preprocess_data(df)
                
                # Create sequences
                X, y = create_sequences(scaled_data, sequence_length)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
                
                st.success(f"Data preprocessed! Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
            
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
            
            current_price = df['Close'].iloc[-1]
            price_change = next_day_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${next_day_price:.2f}", f"${price_change:.2f}")
            with col3:
                st.metric("Predicted Change", f"{price_change_pct:.2f}%")
            
            # Model insights
            st.subheader("💡 Model Insights")
            
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
            """)
            
            if mape < 5:
                st.success("🎉 Excellent model performance! MAPE < 5%")
            elif mape < 10:
                st.warning("⚠️ Good model performance, but could be improved. MAPE < 10%")
            else:
                st.error("❌ Model performance needs improvement. Consider tuning hyperparameters.")

if __name__ == "__main__":
    main()
