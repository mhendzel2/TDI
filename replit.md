# Stock Price Prediction with Transformer Model

## Overview

This is a Streamlit-based web application that implements a true Transformer neural network for stock price prediction using PyTorch. The application allows users to upload stock data and train a deep learning model with multi-head attention mechanism to predict next-day closing prices based on historical market data and technical indicators.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Deployment**: Autoscale deployment on Replit with port 5000
- **UI Components**: Interactive web interface for data upload, model training, and visualization
- **Visualization**: Matplotlib for plotting actual vs predicted stock prices

### Backend Architecture
- **Runtime**: Python 3.11 with Nix package management
- **ML Framework**: PyTorch for deep learning with native transformer implementation
- **Data Processing**: Pandas for data manipulation, NumPy for numerical operations
- **Model Architecture**: Multi-layer Transformer encoder with positional encoding and multi-head attention

### Core Components
1. **Transformer Model Architecture**: 
   - Custom PyTorch Transformer with configurable layers and attention heads
   - Positional encoding for sequence understanding
   - Multi-head self-attention mechanism for temporal pattern recognition
   - Feed-forward networks with residual connections and dropout regularization

2. **Data Pipeline**:
   - CSV data loading and preprocessing
   - MinMaxScaler for feature normalization
   - Sliding window sequence creation for time series modeling

3. **Model Training**:
   - Configurable sequence length (lookback window)
   - Chronological train-test split to prevent data leakage
   - Adam optimizer with learning rate scheduling and gradient clipping
   - Real-time training progress with loss visualization

## Key Components

### Model Architecture
- **Input Processing**: Accepts historical stock data with comprehensive technical indicators (MA, EMA, MACD, RSI, Bollinger Bands)
- **Sequence Modeling**: Transformer encoder with positional encoding for temporal understanding
- **Attention Mechanism**: Multi-head self-attention for capturing complex temporal dependencies
- **Output Layer**: Fully connected layers with dropout for robust price prediction

### Data Requirements
- CSV format with columns: Date, Open, High, Low, Close, Volume
- Chronological ordering required for proper time series modeling
- Automatic feature scaling and normalization
- Advanced technical indicators: Moving averages (5,10,20,50), EMA (12,26), MACD, RSI, Bollinger Bands, volatility measures

### Evaluation Metrics
- Mean Squared Error (MSE) for training loss
- Mean Absolute Error (MAE) for interpretable error measurement
- Root Mean Squared Error (RMSE) for scaled error assessment
- Directional accuracy for trend prediction assessment
- Visual comparison plots and error distribution analysis

## Data Flow

1. **Data Ingestion**: User uploads CSV file through Streamlit interface
2. **Preprocessing**: 
   - Feature selection and normalization using MinMaxScaler
   - Sequence generation with sliding window technique
   - Chronological train-test split (80/20 default)
3. **Model Training**:
   - PyTorch DataLoader for efficient batch processing
   - Transformer model compilation with Adam optimizer
   - Real-time loss monitoring with validation tracking
4. **Evaluation**:
   - Prediction generation on test set
   - Inverse transformation of scaled predictions
   - Metric calculation and visualization
5. **Results Display**: Interactive plots and performance metrics in Streamlit interface

## External Dependencies

### Python Packages
- **PyTorch**: Deep learning framework for Transformer implementation
- **Streamlit**: Web application framework for user interface  
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Data preprocessing (MinMaxScaler) and metrics
- **Matplotlib**: Plotting and visualization

### System Dependencies (Nix)
- Cairo, FFmpeg, FreeType: Graphics and media processing
- GTK3, GObject Introspection: GUI toolkit dependencies
- Ghostscript, pkg-config: Additional system utilities
- QHull, TCL/TK: Mathematical and interface libraries

## Deployment Strategy

### Replit Configuration
- **Target**: Autoscale deployment for automatic resource management
- **Port**: 5000 for Streamlit server
- **Workflow**: Parallel execution with dedicated Streamlit server task
- **Environment**: Nix-based package management with stable-24_05 channel

### Runtime Configuration
- Streamlit server configured for headless operation
- Bound to 0.0.0.0:5000 for external accessibility
- Automatic port waiting for deployment readiness

### Scalability Considerations
- Stateless application design for horizontal scaling
- Model training occurs per session (no persistent storage)
- Memory-efficient data processing with configurable batch sizes

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

```
Changelog:
- June 24, 2025: Initial setup with TensorFlow-based Transformer model
- June 24, 2025: Resolved TensorFlow compatibility issues by switching to ensemble approach using scikit-learn
- June 24, 2025: Implemented working stock prediction app with Random Forest + Linear Regression ensemble
- June 24, 2025: Upgraded to PyTorch implementation with true Transformer architecture including multi-head attention and positional encoding
```