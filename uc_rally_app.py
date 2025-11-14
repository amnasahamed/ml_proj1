"""
UC Rally Detector - Complete Machine Learning System
Detects conditions that lead to Upper-Circuit rallies in Indian stocks (NSE/BSE)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("LightGBM not installed. Only XGBoost and RandomForest will be used.")

import pickle
import os
from typing import List, Tuple, Dict
import time


# ====================================================
# 1. DATA COLLECTION MODULE
# ====================================================

class DataCollector:
    """Handles fetching NSE and BSE stock data"""

    @staticmethod
    def get_nse_symbols() -> List[str]:
        """Get list of NSE stock symbols"""
        # Major NSE stocks with .NS suffix for yfinance
        nse_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
            'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'WIPRO.NS', 'ULTRACEMCO.NS',
            'NESTLEIND.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
            'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'ADANIPORTS.NS', 'COALINDIA.NS',
            'DIVISLAB.NS', 'BAJAJFINSV.NS', 'DRREDDY.NS', 'GRASIM.NS', 'CIPLA.NS',
            'EICHERMOT.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS',
            'INDUSINDBK.NS', 'APOLLOHOSP.NS', 'BPCL.NS', 'ADANIENT.NS', 'SHREECEM.NS',
            'TATACONSUM.NS', 'UPL.NS', 'BAJAJ-AUTO.NS', 'SBILIFE.NS', 'HDFCLIFE.NS',
            'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS', 'PIDILITIND.NS', 'HAVELLS.NS',
            'VOLTAS.NS', 'BANDHANBNK.NS', 'IDEA.NS', 'VEDL.NS', 'SAIL.NS',
            'ZEEL.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'RBLBANK.NS',
            'FEDERALBNK.NS', 'PEL.NS', 'CONCOR.NS', 'LICHSGFIN.NS', 'NMDC.NS',
            'GAIL.NS', 'IOC.NS', 'MOTHERSON.NS', 'BOSCHLTD.NS', 'AMBUJACEM.NS',
            'ACC.NS', 'BERGEPAINT.NS', 'DLF.NS', 'JINDALSTEL.NS', 'TATAPOWER.NS',
            'AUROPHARMA.NS', 'LUPIN.NS', 'BIOCON.NS', 'TORNTPHARM.NS', 'MCDOWELL-N.NS',
            'SIEMENS.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'RECLTD.NS', 'PFC.NS',
            'HINDZINC.NS', 'PAGEIND.NS', 'COLPAL.NS', 'MUTHOOTFIN.NS', 'TRENT.NS',
            'INDIGO.NS', 'MPHASIS.NS', 'MINDTREE.NS', 'LTTS.NS', 'PERSISTENT.NS',
            'COFORGE.NS', 'LALPATHLAB.NS', 'STAR.NS', 'CHOLAFIN.NS', 'BATAINDIA.NS'
        ]
        return nse_symbols

    @staticmethod
    def get_bse_symbols() -> List[str]:
        """Get list of BSE stock symbols"""
        # Major BSE stocks with .BO suffix for yfinance
        bse_symbols = [
            'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO', 'HINDUNILVR.BO',
            'ICICIBANK.BO', 'SBIN.BO', 'BHARTIARTL.BO', 'ITC.BO', 'KOTAKBANK.BO',
            'LT.BO', 'AXISBANK.BO', 'ASIANPAINT.BO', 'MARUTI.BO', 'BAJFINANCE.BO',
            'HCLTECH.BO', 'SUNPHARMA.BO', 'TITAN.BO', 'WIPRO.BO', 'ULTRACEMCO.BO',
            'NESTLEIND.BO', 'ONGC.BO', 'NTPC.BO', 'POWERGRID.BO', 'M&M.BO',
            'TATAMOTORS.BO', 'TATASTEEL.BO', 'TECHM.BO', 'ADANIPORTS.BO', 'COALINDIA.BO',
            'DIVISLAB.BO', 'BAJAJFINSV.BO', 'DRREDDY.BO', 'GRASIM.BO', 'CIPLA.BO',
            'EICHERMOT.BO', 'HEROMOTOCO.BO', 'HINDALCO.BO', 'JSWSTEEL.BO', 'BRITANNIA.BO',
            'INDUSINDBK.BO', 'APOLLOHOSP.BO', 'BPCL.BO', 'ADANIENT.BO', 'SHREECEM.BO',
            'TATACONSUM.BO', 'UPL.BO', 'BAJAJ-AUTO.BO', 'SBILIFE.BO', 'HDFCLIFE.BO',
            'DABUR.BO', 'GODREJCP.BO', 'MARICO.BO', 'PIDILITIND.BO', 'HAVELLS.BO',
            'VOLTAS.BO', 'BANDHANBNK.BO', 'IDEA.BO', 'VEDL.BO', 'SAIL.BO',
            'ZEEL.BO', 'PNB.BO', 'BANKBARODA.BO', 'CANBK.BO', 'RBLBANK.BO',
            'FEDERALBNK.BO', 'PEL.BO', 'CONCOR.BO', 'LICHSGFIN.BO', 'NMDC.BO',
            'GAIL.BO', 'IOC.BO', 'MOTHERSON.BO', 'BOSCHLTD.BO', 'AMBUJACEM.BO',
            'ACC.BO', 'BERGEPAINT.BO', 'DLF.BO', 'JINDALSTEL.BO', 'TATAPOWER.BO',
            'AUROPHARMA.BO', 'LUPIN.BO', 'BIOCON.BO', 'TORNTPHARM.BO', 'MCDOWELL-N.BO',
            'SIEMENS.BO', 'ADANIGREEN.BO', 'ADANITRANS.BO', 'RECLTD.BO', 'PFC.BO',
            'HINDZINC.BO', 'PAGEIND.BO', 'COLPAL.BO', 'MUTHOOTFIN.BO', 'TRENT.BO'
        ]
        return bse_symbols

    @staticmethod
    def fetch_historical_data(symbols: List[str], years: int = 3) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for all symbols
        Returns combined dataframe with columns: Symbol, Date, Open, High, Low, Close, Volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)

        all_data = []
        total_symbols = len(symbols)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, symbol in enumerate(symbols):
            try:
                status_text.text(f"Fetching {symbol} ({idx+1}/{total_symbols})...")

                # Fetch data using yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty or len(df) < 100:  # Skip if insufficient data
                    continue

                # Reset index to get Date as column
                df = df.reset_index()
                df['Symbol'] = symbol

                # Select only required columns
                df = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                all_data.append(df)

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                # Skip stocks with errors
                continue

            # Update progress
            progress_bar.progress((idx + 1) / total_symbols)

        progress_bar.empty()
        status_text.empty()

        if not all_data:
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        return combined_df


# ====================================================
# 2. LABEL GENERATION MODULE
# ====================================================

class LabelGenerator:
    """Generates UC rally labels for each day"""

    @staticmethod
    def generate_labels(df: pd.DataFrame, uc_percent_nse: float = 10.0,
                       uc_percent_bse: float = 5.0) -> pd.DataFrame:
        """
        Label every day with uc_rally = 1 if next 2 days contain High >= upper_circuit_price

        UC Rally = price hits upper circuit for 2 or more consecutive days
        """
        df = df.copy()
        df['uc_rally'] = 0

        # Determine exchange based on symbol suffix
        df['exchange'] = df['Symbol'].apply(lambda x: 'NSE' if x.endswith('.NS') else 'BSE')

        # Calculate upper circuit price based on exchange
        df['uc_percent'] = df['exchange'].apply(
            lambda x: uc_percent_nse if x == 'NSE' else uc_percent_bse
        )

        # Group by symbol for rolling calculations
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            symbol_df = df[mask].copy()

            # Calculate upper circuit price (based on previous close)
            symbol_df['prev_close'] = symbol_df['Close'].shift(1)
            symbol_df['upper_circuit_price'] = symbol_df['prev_close'] * (1 + symbol_df['uc_percent'] / 100)

            # Check if next 2 days hit upper circuit
            symbol_df['hit_uc_day1'] = (symbol_df['High'].shift(-1) >= symbol_df['upper_circuit_price'].shift(-1)).astype(int)
            symbol_df['hit_uc_day2'] = (symbol_df['High'].shift(-2) >= symbol_df['upper_circuit_price'].shift(-2)).astype(int)

            # UC rally = both next 2 days hit upper circuit
            symbol_df['uc_rally'] = ((symbol_df['hit_uc_day1'] == 1) & (symbol_df['hit_uc_day2'] == 1)).astype(int)

            # Update main dataframe
            df.loc[mask, 'uc_rally'] = symbol_df['uc_rally'].values

        return df


# ====================================================
# 3. FEATURE ENGINEERING MODULE
# ====================================================

class FeatureEngineer:
    """Computes all required features"""

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ONLY the specified features:
        - 1-day, 3-day, 5-day, 10-day returns
        - Volume ratio (Volume / avg 20-day volume)
        - MA5, MA10, MA20, MA50
        - Slopes: MA10 - MA20, MA5 - MA20
        - Distance from 5-day high
        - Distance from 20-day high
        - Consecutive up days
        - Candle body/wick ratio
        - Delivery ratio (if available)
        - Volatility (ATR 14)
        - Gap-up percent
        - Compression ratio (High-Low)/Close
        """
        df = df.copy()

        # Group by symbol for calculations
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            symbol_df = df[mask].copy().sort_values('Date')

            # Returns
            symbol_df['return_1d'] = symbol_df['Close'].pct_change(1)
            symbol_df['return_3d'] = symbol_df['Close'].pct_change(3)
            symbol_df['return_5d'] = symbol_df['Close'].pct_change(5)
            symbol_df['return_10d'] = symbol_df['Close'].pct_change(10)

            # Volume ratio
            symbol_df['avg_volume_20d'] = symbol_df['Volume'].rolling(20).mean()
            symbol_df['volume_ratio'] = symbol_df['Volume'] / symbol_df['avg_volume_20d']

            # Moving averages
            symbol_df['MA5'] = symbol_df['Close'].rolling(5).mean()
            symbol_df['MA10'] = symbol_df['Close'].rolling(10).mean()
            symbol_df['MA20'] = symbol_df['Close'].rolling(20).mean()
            symbol_df['MA50'] = symbol_df['Close'].rolling(50).mean()

            # MA slopes
            symbol_df['slope_MA10_MA20'] = symbol_df['MA10'] - symbol_df['MA20']
            symbol_df['slope_MA5_MA20'] = symbol_df['MA5'] - symbol_df['MA20']

            # Distance from highs
            symbol_df['high_5d'] = symbol_df['High'].rolling(5).max()
            symbol_df['high_20d'] = symbol_df['High'].rolling(20).max()
            symbol_df['dist_from_5d_high'] = (symbol_df['Close'] - symbol_df['high_5d']) / symbol_df['high_5d']
            symbol_df['dist_from_20d_high'] = (symbol_df['Close'] - symbol_df['high_20d']) / symbol_df['high_20d']

            # Consecutive up days
            symbol_df['up_day'] = (symbol_df['Close'] > symbol_df['Close'].shift(1)).astype(int)
            symbol_df['consecutive_up_days'] = symbol_df['up_day'].groupby(
                (symbol_df['up_day'] != symbol_df['up_day'].shift()).cumsum()
            ).cumsum()
            symbol_df.loc[symbol_df['up_day'] == 0, 'consecutive_up_days'] = 0

            # Candle body/wick ratio
            symbol_df['candle_body'] = abs(symbol_df['Close'] - symbol_df['Open'])
            symbol_df['candle_range'] = symbol_df['High'] - symbol_df['Low']
            symbol_df['body_wick_ratio'] = symbol_df['candle_body'] / (symbol_df['candle_range'] + 1e-10)

            # Delivery ratio (not available in yfinance, set to 0)
            symbol_df['delivery_ratio'] = 0

            # ATR (Average True Range) for volatility
            symbol_df['h_l'] = symbol_df['High'] - symbol_df['Low']
            symbol_df['h_pc'] = abs(symbol_df['High'] - symbol_df['Close'].shift(1))
            symbol_df['l_pc'] = abs(symbol_df['Low'] - symbol_df['Close'].shift(1))
            symbol_df['tr'] = symbol_df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
            symbol_df['ATR_14'] = symbol_df['tr'].rolling(14).mean()

            # Gap-up percent
            symbol_df['gap_up_pct'] = ((symbol_df['Open'] - symbol_df['Close'].shift(1)) /
                                       symbol_df['Close'].shift(1))

            # Compression ratio
            symbol_df['compression_ratio'] = (symbol_df['High'] - symbol_df['Low']) / symbol_df['Close']

            # Update main dataframe
            df.loc[mask, symbol_df.columns] = symbol_df.values

        return df


# ====================================================
# 4. MODEL TRAINING MODULE
# ====================================================

class ModelTrainer:
    """Trains and evaluates ML models"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.metrics = {}
        self.feature_importance = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare features and labels for training"""
        # Feature columns (exact list from feature engineering)
        self.feature_columns = [
            'return_1d', 'return_3d', 'return_5d', 'return_10d',
            'volume_ratio', 'MA5', 'MA10', 'MA20', 'MA50',
            'slope_MA10_MA20', 'slope_MA5_MA20',
            'dist_from_5d_high', 'dist_from_20d_high',
            'consecutive_up_days', 'body_wick_ratio', 'delivery_ratio',
            'ATR_14', 'gap_up_pct', 'compression_ratio'
        ]

        # Remove rows with NaN in features or labels
        df_clean = df[self.feature_columns + ['uc_rally']].dropna()

        X = df_clean[self.feature_columns]
        y = df_clean['uc_rally']

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all three models"""

        # 1. XGBoost
        st.write("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model

        # 2. Random Forest
        st.write("Training RandomForest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model

        # 3. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            st.write("Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42
            )
            lgb_model.fit(X_train, y_train)
            self.models['LightGBM'] = lgb_model

        # Evaluate all models
        st.write("\nEvaluating models...")
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            self.metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

        # Select best model based on ROC-AUC
        best_roc_auc = 0
        for name, metrics in self.metrics.items():
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                self.best_model_name = name
                self.best_model = self.models[name]

        # Extract feature importance from best model
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

    def get_metrics_summary(self) -> pd.DataFrame:
        """Return metrics summary as dataframe"""
        metrics_data = []
        for name, metrics in self.metrics.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        return pd.DataFrame(metrics_data)

    def save_model(self, filepath: str):
        """Save best model and feature columns"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance
            }, f)

    @staticmethod
    def load_model(filepath: str):
        """Load saved model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        trainer = ModelTrainer()
        trainer.best_model = data['model']
        trainer.best_model_name = data['model_name']
        trainer.feature_columns = data['feature_columns']
        trainer.feature_importance = data['feature_importance']
        return trainer


# ====================================================
# 5. UC CONDITION DISCOVERY MODULE
# ====================================================

class UCConditionDiscovery:
    """Discovers and summarizes UC rally conditions"""

    @staticmethod
    def get_top_features(feature_importance: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        return feature_importance.head(top_n)

    @staticmethod
    def generate_summary(feature_importance: pd.DataFrame) -> str:
        """Generate natural language summary of UC conditions"""
        top_features = feature_importance.head(20)

        summary = "**The most common pre-UC conditions are:**\n\n"

        # Group features by category
        momentum_features = []
        volume_features = []
        ma_features = []
        volatility_features = []

        for _, row in top_features.iterrows():
            feature = row['feature']
            if 'return' in feature or 'gap_up' in feature or 'consecutive_up' in feature:
                momentum_features.append(feature)
            elif 'volume' in feature:
                volume_features.append(feature)
            elif 'MA' in feature or 'slope' in feature or 'dist_from' in feature:
                ma_features.append(feature)
            elif 'ATR' in feature or 'compression' in feature or 'body_wick' in feature:
                volatility_features.append(feature)

        if momentum_features:
            summary += f"1. **Strong Momentum**: {', '.join(momentum_features[:3])}\n"
        if volume_features:
            summary += f"2. **Volume Surge**: {', '.join(volume_features[:2])}\n"
        if ma_features:
            summary += f"3. **Moving Average Alignment**: {', '.join(ma_features[:3])}\n"
        if volatility_features:
            summary += f"4. **Volatility Patterns**: {', '.join(volatility_features[:2])}\n"

        summary += "\n**Key Insights:**\n"
        summary += "- UC rallies typically follow strong momentum in recent days\n"
        summary += "- Volume spikes are a critical indicator\n"
        summary += "- Stocks above key moving averages are more likely to rally\n"
        summary += "- Price compression often precedes breakouts\n"

        return summary


# ====================================================
# 6. PREDICTION MODULE
# ====================================================

class Predictor:
    """Predicts UC rally probability for any stock"""

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def predict_uc_rally(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Predict UC rally probability for a symbol
        Returns: probability, trigger_price, volume_threshold, reasoning
        """
        # Add exchange suffix
        if exchange == 'NSE':
            symbol_yf = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        else:
            symbol_yf = f"{symbol}.BO" if not symbol.endswith('.BO') else symbol

        # Fetch last 60 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Extra buffer

        try:
            ticker = yf.Ticker(symbol_yf)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty or len(df) < 60:
                return {'error': 'Insufficient data for prediction'}

            # Prepare data
            df = df.reset_index()
            df['Symbol'] = symbol_yf
            df = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Compute features
            feature_eng = FeatureEngineer()
            df = feature_eng.compute_features(df)

            # Get latest row with all features
            df_latest = df[self.trainer.feature_columns].dropna()
            if df_latest.empty:
                return {'error': 'Could not compute features'}

            latest_features = df_latest.iloc[-1:]

            # Predict
            probability = self.trainer.best_model.predict_proba(latest_features)[0, 1]

            # Calculate trigger price (1% above 5-day high)
            high_5d = df['High'].tail(5).max()
            trigger_price = high_5d * 1.01

            # Calculate volume threshold (1.5x 20-day average)
            avg_volume_20d = df['Volume'].tail(20).mean()
            volume_threshold = avg_volume_20d * 1.5

            # Get top 5 contributing features
            top_5_features = self.trainer.feature_importance.head(5)
            reasoning = []
            for _, row in top_5_features.iterrows():
                feature_name = row['feature']
                feature_value = latest_features[feature_name].values[0]
                reasoning.append(f"{feature_name}: {feature_value:.4f}")

            return {
                'symbol': symbol_yf,
                'probability': probability,
                'trigger_price': trigger_price,
                'volume_threshold': volume_threshold,
                'reasoning': reasoning,
                'current_price': df['Close'].iloc[-1]
            }

        except Exception as e:
            return {'error': str(e)}

    def bulk_screen(self, symbols: List[str]) -> pd.DataFrame:
        """Screen multiple symbols and return ranked by probability"""
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, symbol in enumerate(symbols):
            status_text.text(f"Screening {symbol} ({idx+1}/{len(symbols)})...")

            # Determine exchange from symbol
            exchange = 'NSE' if symbol.endswith('.NS') else 'BSE'
            pred = self.predict_uc_rally(symbol, exchange)

            if 'error' not in pred:
                results.append({
                    'Symbol': pred['symbol'],
                    'UC_Probability': f"{pred['probability']:.4f}",
                    'Current_Price': f"{pred['current_price']:.2f}",
                    'Trigger_Price': f"{pred['trigger_price']:.2f}",
                    'Volume_Threshold': f"{pred['volume_threshold']:.0f}"
                })

            progress_bar.progress((idx + 1) / len(symbols))
            time.sleep(0.1)  # Avoid rate limiting

        progress_bar.empty()
        status_text.empty()

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results['UC_Probability'] = df_results['UC_Probability'].astype(float)
            df_results = df_results.sort_values('UC_Probability', ascending=False)

        return df_results


# ====================================================
# 7. STREAMLIT GUI
# ====================================================

def main():
    st.set_page_config(page_title="UC Rally Detector", layout="wide")

    st.title("🚀 Upper-Circuit Rally Detection System")
    st.markdown("**Complete ML System for NSE/BSE Stocks**")

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Fetch & Training", "Predict for Any Stock", "Bulk Screener"]
    )

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None

    # ====================================================
    # PAGE 1: DATA FETCH & TRAINING
    # ====================================================
    if page == "Data Fetch & Training":
        st.header("📊 Data Collection & Model Training")

        # UC percent override
        st.subheader("Upper Circuit Settings")
        col1, col2 = st.columns(2)
        with col1:
            uc_percent_nse = st.number_input("NSE UC %", value=10.0, min_value=1.0, max_value=20.0)
        with col2:
            uc_percent_bse = st.number_input("BSE UC %", value=5.0, min_value=1.0, max_value=20.0)

        st.markdown("---")

        # Data fetching
        st.subheader("Step 1: Fetch Historical Data")
        if st.button("🔄 Fetch all NSE/BSE data (3 years)", type="primary"):
            with st.spinner("Fetching data for all NSE and BSE stocks..."):
                collector = DataCollector()

                # Get symbols
                nse_symbols = collector.get_nse_symbols()
                bse_symbols = collector.get_bse_symbols()
                all_symbols = nse_symbols + bse_symbols

                st.info(f"Fetching data for {len(all_symbols)} stocks ({len(nse_symbols)} NSE + {len(bse_symbols)} BSE)...")

                # Fetch data
                stock_data = collector.fetch_historical_data(all_symbols, years=3)

                if not stock_data.empty:
                    st.success(f"✅ Successfully fetched data for {stock_data['Symbol'].nunique()} stocks!")
                    st.write(f"Total records: {len(stock_data):,}")

                    # Generate labels
                    st.info("Generating UC rally labels...")
                    label_gen = LabelGenerator()
                    stock_data = label_gen.generate_labels(stock_data, uc_percent_nse, uc_percent_bse)

                    # Compute features
                    st.info("Computing features...")
                    feature_eng = FeatureEngineer()
                    stock_data = feature_eng.compute_features(stock_data)

                    st.session_state.stock_data = stock_data
                    st.session_state.data_loaded = True

                    st.success("✅ Data preparation complete!")
                    st.dataframe(stock_data.head(10))
                else:
                    st.error("Failed to fetch data. Please try again.")

        st.markdown("---")

        # Model training
        st.subheader("Step 2: Train UC Rally Model")
        if st.button("🤖 Train UC Rally Model", type="primary", disabled=not st.session_state.data_loaded):
            if st.session_state.stock_data is None:
                st.error("Please fetch data first!")
            else:
                with st.spinner("Training models..."):
                    trainer = ModelTrainer()

                    # Prepare data
                    st.info("Preparing training data...")
                    X_train, X_test, y_train, y_test = trainer.prepare_data(st.session_state.stock_data)

                    st.write(f"Training samples: {len(X_train):,}")
                    st.write(f"Test samples: {len(X_test):,}")
                    st.write(f"UC Rally cases: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")

                    # Train models
                    trainer.train_models(X_train, X_test, y_train, y_test)

                    st.session_state.trainer = trainer
                    st.session_state.model_trained = True

                    # Save model
                    trainer.save_model('uc_rally_model.pkl')

                    st.success(f"✅ Training complete! Best model: {trainer.best_model_name}")

        # Display metrics
        if st.session_state.model_trained and st.session_state.trainer:
            st.markdown("---")
            st.subheader("📈 Model Performance")

            trainer = st.session_state.trainer

            # Metrics table
            st.write("**Model Comparison:**")
            metrics_df = trainer.get_metrics_summary()
            st.dataframe(metrics_df)

            # Best model confusion matrix
            st.write(f"**Confusion Matrix ({trainer.best_model_name}):**")
            cm = trainer.metrics[trainer.best_model_name]['confusion_matrix']
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: No Rally', 'Actual: Rally'],
                columns=['Pred: No Rally', 'Pred: Rally']
            )
            st.dataframe(cm_df)

            # Feature importance
            st.markdown("---")
            st.subheader("🎯 Feature Importance (Top 20)")
            top_20_features = UCConditionDiscovery.get_top_features(trainer.feature_importance, 20)
            st.dataframe(top_20_features)

            # UC Conditions Summary
            st.markdown("---")
            st.subheader("💡 Discovered UC Rally Conditions")
            summary = UCConditionDiscovery.generate_summary(trainer.feature_importance)
            st.markdown(summary)

    # ====================================================
    # PAGE 2: PREDICT FOR ANY STOCK
    # ====================================================
    elif page == "Predict for Any Stock":
        st.header("🔮 Predict UC Rally for Individual Stock")

        # Load model if exists
        if not st.session_state.model_trained:
            if os.path.exists('uc_rally_model.pkl'):
                st.session_state.trainer = ModelTrainer.load_model('uc_rally_model.pkl')
                st.session_state.model_trained = True
            else:
                st.warning("⚠️ No trained model found. Please train model first in 'Data Fetch & Training' page.")
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            symbol_input = st.text_input("Enter Stock Symbol (without exchange suffix)", "RELIANCE")
        with col2:
            exchange = st.selectbox("Select Exchange", ["NSE", "BSE"])

        if st.button("🎯 Predict UC Rally Probability", type="primary"):
            predictor = Predictor(st.session_state.trainer)

            with st.spinner(f"Analyzing {symbol_input}..."):
                result = predictor.predict_uc_rally(symbol_input, exchange)

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("✅ Prediction Complete!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("UC Rally Probability", f"{result['probability']*100:.2f}%")
                with col2:
                    st.metric("Current Price", f"₹{result['current_price']:.2f}")
                with col3:
                    st.metric("Trigger Price", f"₹{result['trigger_price']:.2f}")

                st.metric("Volume Threshold", f"{result['volume_threshold']:,.0f}")

                # Reasoning
                st.markdown("---")
                st.subheader("🔍 Key Signals (Top 5 Features)")
                for reason in result['reasoning']:
                    st.write(f"• {reason}")

                # Interpretation
                st.markdown("---")
                st.subheader("📊 Interpretation")
                if result['probability'] > 0.7:
                    st.success("🟢 **HIGH PROBABILITY** - Strong UC rally signals detected!")
                elif result['probability'] > 0.4:
                    st.warning("🟡 **MODERATE PROBABILITY** - Some UC rally signals present.")
                else:
                    st.info("🔵 **LOW PROBABILITY** - Weak UC rally signals.")

    # ====================================================
    # PAGE 3: BULK SCREENER
    # ====================================================
    elif page == "Bulk Screener":
        st.header("📋 Bulk Stock Screener")

        # Load model if exists
        if not st.session_state.model_trained:
            if os.path.exists('uc_rally_model.pkl'):
                st.session_state.trainer = ModelTrainer.load_model('uc_rally_model.pkl')
                st.session_state.model_trained = True
            else:
                st.warning("⚠️ No trained model found. Please train model first in 'Data Fetch & Training' page.")
                st.stop()

        exchange_choice = st.radio("Select Exchange", ["NSE", "BSE", "Both"])

        if st.button("🔍 Screen All Stocks", type="primary"):
            collector = DataCollector()

            if exchange_choice == "NSE":
                symbols = collector.get_nse_symbols()
            elif exchange_choice == "BSE":
                symbols = collector.get_bse_symbols()
            else:
                symbols = collector.get_nse_symbols() + collector.get_bse_symbols()

            st.info(f"Screening {len(symbols)} stocks...")

            predictor = Predictor(st.session_state.trainer)
            results_df = predictor.bulk_screen(symbols)

            if not results_df.empty:
                st.success(f"✅ Screening complete! Found {len(results_df)} stocks with predictions.")

                st.subheader("🏆 Top UC Rally Candidates (Ranked by Probability)")
                st.dataframe(results_df, use_container_width=True, height=600)

                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"uc_rally_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found.")


if __name__ == "__main__":
    main()
