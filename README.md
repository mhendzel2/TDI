# Stock Price Prediction with PyTorch Transformer

A Streamlit web application that implements a transformer neural network for stock price prediction using PyTorch. Features multi-head attention, positional encoding, and comprehensive technical indicators for time series forecasting.

## Features

- **True Transformer Architecture**: Multi-layer transformer encoder with self-attention mechanism
- **Technical Indicators**: Advanced indicators including MA, EMA, MACD, RSI, Bollinger Bands
- **Interactive Interface**: Streamlit web app for data upload and model training
- **Real-time Training**: Live progress tracking with loss visualization
- **Comprehensive Analysis**: Detailed evaluation metrics and prediction visualizations

## Quick Start

1. Install dependencies:
```bash
pip install torch torchvision torchaudio streamlit pandas numpy scikit-learn matplotlib
```

2. Run the application:
```bash
streamlit run app_pytorch.py --server.port 5000
```

3. Access the app at `http://localhost:5000`

## Project Structure

- `app_pytorch.py` - Main PyTorch transformer implementation
- `app_simple.py` - Ensemble model fallback (Random Forest + Linear Regression)
- `app.py` - Original TensorFlow implementation (compatibility issues)
- `replit.md` - Project documentation and architecture details
- `.streamlit/config.toml` - Streamlit server configuration

## Model Architecture

The transformer model includes:
- Input projection layer for feature dimension mapping
- Sinusoidal positional encoding for sequence understanding
- Multi-head self-attention layers for pattern recognition
- Feed-forward networks with residual connections
- Dropout regularization for generalization

## Data Requirements

Upload CSV files with columns:
- Date, Open, High, Low, Close, Volume

The system automatically generates technical indicators and scales features for optimal training.

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error) 
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (trend prediction accuracy)

## Technical Details

- **Framework**: PyTorch 2.7+ with CUDA support
- **Architecture**: Transformer encoder with configurable parameters
- **Training**: Adam optimizer with learning rate scheduling
- **Data**: MinMaxScaler normalization with chronological splitting

## Configuration Options

- Sequence length (30-120 days)
- Model dimension (64, 128, 256)
- Attention heads (4, 8, 16)
- Transformer layers (2-8)
- Training epochs (20-200)
- Batch size (16, 32, 64)

## Deployment

The application is configured for Replit deployment with autoscale support. The Streamlit server runs on port 5000 with proper headless configuration.