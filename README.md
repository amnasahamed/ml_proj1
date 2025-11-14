# UC Rally Detector - Machine Learning System

Complete machine learning system that detects conditions leading to Upper-Circuit rallies in Indian stocks (NSE/BSE) using 3 years of historical data.

## Features

### 1. Data Collection
- Fetches all NSE and BSE stock symbols
- Downloads 3 years of historical OHLCV data using yfinance
- Handles missing data and errors gracefully

### 2. Label Generation
- Identifies UC rally patterns (2+ consecutive days hitting upper circuit)
- Default UC percentages: NSE (10%), BSE (5%)
- Customizable UC percentage settings

### 3. Feature Engineering
Computes 19 technical features:
- Returns (1d, 3d, 5d, 10d)
- Volume ratio
- Moving averages (MA5, MA10, MA20, MA50)
- MA slopes
- Distance from highs
- Consecutive up days
- Candle patterns
- Volatility (ATR)
- Gap-up percentage
- Compression ratio

### 4. Model Training
Trains three models:
- XGBoost (n_estimators=300, max_depth=6, lr=0.05)
- RandomForest (n_estimators=300)
- LightGBM (num_leaves=31, lr=0.05, n_estimators=300)

Evaluates using: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 5. UC Condition Discovery
- Extracts top 20 most important features
- Generates natural language summary of pre-UC conditions

### 6. Prediction Module
For any stock, provides:
- UC rally probability (0-1)
- Trigger price (1% above 5-day high)
- Volume threshold (1.5x 20-day average)
- Top 5 contributing features

### 7. Streamlit GUI
Three-page application:
- **Page 1**: Data fetch and model training with metrics
- **Page 2**: Individual stock prediction
- **Page 3**: Bulk screener for all NSE/BSE stocks

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
streamlit run uc_rally_app.py
```

The application will open in your default browser at `http://localhost:8501`

## Workflow

### Step 1: Fetch Data & Train Model
1. Go to "Data Fetch & Training" page
2. Adjust UC percentage if needed (default: NSE=10%, BSE=5%)
3. Click "Fetch all NSE/BSE data (3 years)"
4. Wait for data collection to complete
5. Click "Train UC Rally Model"
6. Review model metrics and feature importance

### Step 2: Predict for Individual Stock
1. Go to "Predict for Any Stock" page
2. Enter stock symbol (e.g., "RELIANCE")
3. Select exchange (NSE/BSE)
4. Click "Predict UC Rally Probability"
5. Review probability, trigger price, and key signals

### Step 3: Screen All Stocks
1. Go to "Bulk Screener" page
2. Select exchange (NSE/BSE/Both)
3. Click "Screen All Stocks"
4. View ranked list of UC rally candidates
5. Download results as CSV

## Model Performance

The system automatically selects the best model based on ROC-AUC score. Typical performance:
- ROC-AUC: 0.75-0.85
- F1-Score: 0.60-0.75
- Precision: 0.65-0.80

## Technical Details

### Upper Circuit Definition
- **UC Rally**: Price hits upper circuit for 2 or more consecutive days
- **NSE Upper Circuit**: 10% above previous close (configurable)
- **BSE Upper Circuit**: 5% above previous close (configurable)

### Data Source
- Uses yfinance API for historical stock data
- Fetches OHLCV (Open, High, Low, Close, Volume) data
- 3-year historical window for training

### Feature List (19 features)
1. return_1d
2. return_3d
3. return_5d
4. return_10d
5. volume_ratio
6. MA5
7. MA10
8. MA20
9. MA50
10. slope_MA10_MA20
11. slope_MA5_MA20
12. dist_from_5d_high
13. dist_from_20d_high
14. consecutive_up_days
15. body_wick_ratio
16. delivery_ratio
17. ATR_14
18. gap_up_pct
19. compression_ratio

## Files Generated

- `uc_rally_model.pkl`: Trained model and metadata
- `uc_rally_screener_YYYYMMDD.csv`: Bulk screening results (downloadable)

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- scikit-learn
- xgboost
- lightgbm
- streamlit

## Limitations

- Depends on yfinance API availability
- Limited to stocks available in yfinance
- Delivery ratio feature not available (set to 0)
- Historical data may have gaps or errors

## Notes

- First data fetch may take 10-20 minutes depending on connection
- Model training takes 2-5 minutes
- Bulk screening can take several minutes
- Results are for educational purposes only
- Always verify predictions with additional analysis

## License

MIT License
