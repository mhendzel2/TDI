import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

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
    
    # Create technical indicators
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Select features for model
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Price_Change', 'Volume_Change']
    
    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
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

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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

# Simple Transformer-like model using ensemble methods
class SimpleTransformerModel:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            RandomForestRegressor(n_estimators=150, max_depth=10, random_state=43),
            LinearRegression()
        ]
        self.weights = [0.4, 0.4, 0.2]  # Ensemble weights
    
    def prepare_features(self, X):
        """Prepare features from sequences for traditional ML models"""
        # Flatten sequences and add statistical features
        batch_size = X.shape[0]
        
        # Flatten the sequence
        X_flat = X.reshape(batch_size, -1)
        
        # Add statistical features for each sequence
        X_stats = np.column_stack([
            np.mean(X, axis=1),      # Mean of each feature across time
            np.std(X, axis=1),       # Std of each feature across time
            np.max(X, axis=1),       # Max of each feature across time
            np.min(X, axis=1),       # Min of each feature across time
            X[:, -1, :],             # Last timestep features
            X[:, 0, :],              # First timestep features
        ])
        
        return np.column_stack([X_flat, X_stats])
    
    def fit(self, X, y, progress_callback=None):
        """Train the ensemble model"""
        X_features = self.prepare_features(X)
        
        # Train each model
        for i, model in enumerate(self.models):
            if progress_callback:
                progress_callback(i, len(self.models))
            model.fit(X_features, y)
    
    def predict(self, X):
        """Make predictions using ensemble"""
        X_features = self.prepare_features(X)
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X_features)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)

# Streamlit App
def main():
    st.title("Stock Price Prediction with Advanced ML Models")
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
    model_type = st.sidebar.selectbox("Model Type", ["Ensemble (RF + Linear)", "Random Forest", "Linear Regression"])
    
    # Load data
    df = None
    
    if data_source == "Upload CSV File":
        st.subheader("Upload Stock Data")
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
        st.subheader("Generate Synthetic Data")
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.number_input("Number of Days", 500, 2000, 1000)
        with col2:
            start_price = st.number_input("Starting Price", 50, 500, 100)
        
        if st.button("Generate Data"):
            with st.spinner("Generating synthetic stock data..."):
                df = generate_synthetic_stock_data(num_days, start_price)
                st.success(f"Synthetic data generated! Shape: {df.shape}")
    
    if df is not None:
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Display basic statistics
        st.subheader("Data Statistics")
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
        st.subheader("Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'])
        ax.set_title("Historical Closing Prices")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Train model button
        if st.button("Train Prediction Model", type="primary"):
            
            with st.spinner("Preprocessing data..."):
                # Preprocess data
                scaled_data, scaler_features, scaler_target, original_df = load_and_preprocess_data(df)
                
                # Create sequences
                X, y = create_sequences(scaled_data, sequence_length)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
                
                st.success(f"Data preprocessed! Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
            
            with st.spinner("Training prediction model..."):
                # Initialize model
                if model_type == "Ensemble (RF + Linear)":
                    model = SimpleTransformerModel(sequence_length, X.shape[-1])
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                elif model_type == "Linear Regression":
                    model = LinearRegression()
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(step, total_steps):
                    progress = (step + 1) / total_steps
                    progress_bar.progress(progress)
                    status_text.text(f"Training step {step + 1}/{total_steps}")
                
                # Train model
                if hasattr(model, 'fit') and 'SimpleTransformerModel' in str(type(model)):
                    model.fit(X_train, y_train, progress_callback=update_progress)
                else:
                    # For sklearn models, prepare flattened features
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    model.fit(X_train_flat, y_train)
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                st.success("Model training completed!")
            
            # Make predictions
            st.subheader("Model Predictions")
            
            with st.spinner("Generating predictions..."):
                # Predict on test set
                if hasattr(model, 'predict') and 'SimpleTransformerModel' in str(type(model)):
                    y_pred_scaled = model.predict(X_test)
                else:
                    y_pred_scaled = model.predict(X_test_flat)
                
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
            st.subheader("Prediction Results")
            
            # Test set predictions
            fig = plot_predictions(y_test_original, y_pred, "Test Set: Actual vs Predicted Prices")
            st.pyplot(fig)
            
            # Training set predictions for comparison
            if st.checkbox("Show Training Set Predictions"):
                if hasattr(model, 'predict') and 'SimpleTransformerModel' in str(type(model)):
                    y_pred_train_scaled = model.predict(X_train)
                else:
                    y_pred_train_scaled = model.predict(X_train_flat)
                    
                y_pred_train = scaler_target.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
                y_train_original = scaler_target.inverse_transform(y_train.reshape(-1, 1)).flatten()
                
                fig = plot_predictions(y_train_original, y_pred_train, "Training Set: Actual vs Predicted Prices")
                st.pyplot(fig)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance = model.feature_importances_
                feature_names = [f"Feature_{i}" for i in range(len(importance))]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                indices = np.argsort(importance)[::-1][:20]  # Top 20 features
                ax.bar(range(len(indices)), importance[indices])
                ax.set_title("Top 20 Feature Importance")
                ax.set_xlabel("Features")
                ax.set_ylabel("Importance")
                st.pyplot(fig)

if __name__ == "__main__":
    main()