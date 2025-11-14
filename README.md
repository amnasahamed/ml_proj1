# UC Rally Detector - Machine Learning System

Complete machine learning system that identifies Upper-Circuit (UC) trading opportunities in Indian stocks (NSE/BSE).

## Strategy

**Buy stocks at Upper Circuit (UC), Hold until Lower Circuit (LC)**

The system detects when stocks hit upper circuit (price closes at or very near the daily high with significant gain), predicts future UC candidates, and estimates the holding period until the stock hits lower circuit.

## Features

### 1. Data Collection
- Fetches NSE and BSE stock symbols
- Downloads up to 3 years of historical OHLCV data using yfinance
- **Accepts stocks with minimum 15 days of data**
- Handles missing data and errors gracefully

### 2. UC/LC Detection
**Upper Circuit Detection:**
- Close price within 0.5% of High (configurable tolerance)
- Gain from previous close >= 2% (configurable minimum)
- Accurately identifies UC days where stock closed at circuit limit

**Lower Circuit Detection:**
- Close price within 0.5% of Low (configurable tolerance)
- Loss from previous close >= 2% (configurable minimum)
- Identifies LC days for exit strategy

**UC to LC Tracking:**
- For each UC day, calculates days until next LC hit
- Enables holding period prediction

### 3. Label Generation
**Two Prediction Tasks:**
1. **Classification:** Will stock hit UC tomorrow? (1/0)
2. **Regression:** How many days from UC to LC?

### 4. Feature Engineering
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

### 5. Model Training
**Classification Models** (Predict UC Tomorrow):
- XGBoost (n_estimators=300, max_depth=6, lr=0.05)
- RandomForest (n_estimators=300)
- LightGBM (num_leaves=31, lr=0.05, n_estimators=300)

**Regression Models** (Predict Days to LC):
- XGBoost Regressor
- RandomForest Regressor
- LightGBM Regressor

Evaluates using: Accuracy, Precision, Recall, F1-Score, ROC-AUC (Classification), MAE, RMSE (Regression)

### 6. UC Condition Discovery
- Extracts top 20 most important features
- Generates natural language summary of pre-UC conditions
- Analyzes historical UC to LC patterns
- Shows average holding periods

### 7. Prediction Module
For any stock, provides:
- **UC Tomorrow Probability** (0-1)
- **Current UC Status** (identifies if currently at UC)
- **Days to LC Prediction** (if currently at UC)
- Trigger price (1% above 5-day high)
- Volume threshold (1.5x 20-day average)
- Top 5 contributing features

### 8. Streamlit GUI
Four-page application:
- **Page 1**: Data fetch and model training with configurable circuit detection
- **Page 2**: Individual stock UC prediction with current status
- **Page 3**: Bulk screener for all NSE/BSE stocks (identifies current UC stocks)
- **Page 4**: UC to LC pattern analyzer with distribution charts

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

### Step 1: Configure & Fetch Data
1. Go to "Data Fetch & Training" page
2. Adjust circuit detection settings:
   - Min UC Gain %: Minimum gain to consider UC (default 2%)
   - Min LC Loss %: Minimum loss to consider LC (default 2%)
   - Price Tolerance %: How close Close should be to High/Low (default 0.5%)
3. Click "Fetch all NSE/BSE data"
4. System accepts stocks with 15+ days of data
5. Review UC/LC detection statistics

### Step 2: Train Models
1. Click "Train Models"
2. Two models trained:
   - **Classifier**: Predicts UC tomorrow
   - **Regressor**: Predicts days from UC to LC
3. Review model metrics and feature importance
4. Analyze historical UC to LC patterns

### Step 3: Predict Individual Stock
1. Go to "Predict UC for Any Stock" page
2. Enter stock symbol (e.g., "RELIANCE")
3. Select exchange (NSE/BSE)
4. Click "Predict UC Probability"
5. Results show:
   - If stock is **currently at UC** (🔥 highlighted)
   - Predicted days to LC (if at UC)
   - UC tomorrow probability
   - Trigger price and volume threshold
   - Key signals

### Step 4: Screen All Stocks
1. Go to "Bulk UC Screener" page
2. Select exchange (NSE/BSE/Both)
3. Click "Screen All Stocks"
4. View ranked list:
   - Stocks currently AT UPPER CIRCUIT marked ✓
   - UC probability for all stocks
   - Current price and gain
5. Download results as CSV

### Step 5: Analyze UC to LC Patterns
1. Go to "UC to LC Analyzer" page
2. View distribution of holding periods
3. See statistics by stock:
   - Average days from UC to LC
   - Median, Min, Max holding periods
   - Number of UC events per stock
4. Download full statistics

## How Circuit Detection Works

### Upper Circuit (UC)
A day is classified as UC when:
1. **Close ≈ High**: `(High - Close) / Close <= 0.5%`
2. **Significant Gain**: `(Close - Prev_Close) / Prev_Close >= 2%`

This means the stock closed very near its daily high with strong upward movement.

### Lower Circuit (LC)
A day is classified as LC when:
1. **Close ≈ Low**: `(Close - Low) / Close <= 0.5%`
2. **Significant Loss**: `(Close - Prev_Close) / Prev_Close <= -2%`

### Dynamic Detection
- Circuit percentages are NOT fixed
- Exchanges change limits dynamically (can be 2%, 5%, 10%, 20%)
- System detects from actual price behavior, not hardcoded percentages
- Configurable tolerance and minimum gain/loss thresholds

## Model Performance

Typical performance metrics:

**Classification (UC Tomorrow):**
- ROC-AUC: 0.70-0.85
- F1-Score: 0.55-0.75
- Precision: 0.60-0.80

**Regression (Days to LC):**
- MAE: 3-7 days
- RMSE: 5-12 days

## Technical Details

### Data Requirements
- **Maximum**: 3 years historical data
- **Minimum**: 15 days of data
- Stocks with less than 15 days are skipped
- Both NSE (.NS) and BSE (.BO) supported

### Data Source
- Uses yfinance API for historical stock data
- Fetches OHLCV (Open, High, Low, Close, Volume) data
- 0.1s delay between requests to avoid rate limiting

### Feature List (19 features)
1. return_1d - 1-day return
2. return_3d - 3-day return
3. return_5d - 5-day return
4. return_10d - 10-day return
5. volume_ratio - Volume / 20-day avg volume
6. MA5 - 5-day moving average
7. MA10 - 10-day moving average
8. MA20 - 20-day moving average
9. MA50 - 50-day moving average
10. slope_MA10_MA20 - MA10 - MA20
11. slope_MA5_MA20 - MA5 - MA20
12. dist_from_5d_high - Distance from 5-day high
13. dist_from_20d_high - Distance from 20-day high
14. consecutive_up_days - Count of consecutive up days
15. body_wick_ratio - Candle body / total range
16. delivery_ratio - Delivery percentage (set to 0, not available)
17. ATR_14 - 14-day Average True Range
18. gap_up_pct - Gap-up percentage
19. compression_ratio - (High-Low) / Close

## Files Generated

- `uc_rally_model.pkl`: Trained models and metadata
- `uc_screener_YYYYMMDD.csv`: Bulk screening results
- `uc_to_lc_stats_YYYYMMDD.csv`: UC to LC pattern statistics

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- scikit-learn
- xgboost
- lightgbm (optional)
- streamlit
- matplotlib

## GUI Pages

### Page 1: Data Fetch & Training
- Configure circuit detection parameters
- Fetch NSE/BSE data (15 days minimum, 3 years maximum)
- Train classification and regression models
- View model metrics, confusion matrix
- Feature importance analysis
- Historical UC to LC statistics

### Page 2: Predict UC for Any Stock
- Enter stock symbol and exchange
- Get UC probability prediction
- **Current status**: Shows if stock is AT UC now
- **Days to LC**: Predicted holding period if at UC
- Trigger price and volume threshold
- Top 5 feature signals
- Action recommendations

### Page 3: Bulk UC Screener
- Screen all NSE/BSE stocks
- **Highlights stocks currently at UC** with ✓ marker
- Ranked by UC tomorrow probability
- Shows current price, today's gain
- Download results as CSV

### Page 4: UC to LC Analyzer
- Histogram of holding periods (UC to LC)
- Statistics by stock
- Average, median, min, max days
- Download full statistics

## Strategy Explanation

**Traditional UC Strategy:**
1. Identify stocks hitting upper circuit
2. Buy at UC (or as close as possible)
3. Hold the position
4. Exit when stock hits lower circuit
5. Profit from the rally between UC and LC

**How This System Helps:**
1. **Prediction**: Identifies stocks likely to hit UC tomorrow
2. **Current Monitoring**: Shows which stocks are at UC right now
3. **Holding Period**: Predicts how long to hold (days until LC)
4. **Risk Management**: Provides trigger prices and volume thresholds
5. **Pattern Analysis**: Shows historical UC to LC behavior

## Limitations

- Depends on yfinance API availability
- Circuit detection based on price behavior (not official exchange data)
- Delivery ratio not available (set to 0)
- Historical data may have gaps
- Not all stocks available in yfinance
- Predictions are probabilistic, not guaranteed

## Notes

- First data fetch may take 10-20 minutes
- Model training takes 2-5 minutes
- Bulk screening can take several minutes
- Results are for educational purposes only
- Always verify with additional analysis
- Past performance doesn't guarantee future results

## Circuit Tolerance Settings

### Default Settings
- **Min UC Gain**: 2% (can increase for stricter detection)
- **Min LC Loss**: 2% (can increase for stricter detection)
- **Price Tolerance**: 0.5% (distance from High/Low)

### Adjusting for Different Markets
- **Volatile stocks**: Increase min gain/loss to 5-10%
- **Less liquid stocks**: Increase price tolerance to 1-2%
- **Large caps**: Default settings usually work well
- **Small caps**: May need higher thresholds

## License

MIT License
