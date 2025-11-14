"""
UC Rally Detector - Complete Machine Learning System
Strategy: Buy stocks at Upper Circuit (UC), hold until Lower Circuit (LC)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error
import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import pickle
import os
from typing import List, Tuple, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import StringIO


# ====================================================
# 1. DATA COLLECTION MODULE
# ====================================================

class DataCollector:
    """Handles fetching NSE and BSE stock data"""

    @staticmethod
    def get_nse_symbols() -> List[str]:
        """Get list of NSE stock symbols (fast mode - 100 stocks)"""
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
        """Get list of BSE stock symbols (fast mode - 100 stocks)"""
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
    def load_all_nse_symbols() -> List[str]:
        """Load ALL NSE equity symbols from official NSE source"""
        try:
            st.info("Fetching complete NSE equity list from NSE India...")
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            # Extract symbol column (usually 'SYMBOL')
            if 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].dropna().unique().tolist()
            else:
                # Fallback to first column
                symbols = df.iloc[:, 0].dropna().unique().tolist()

            # Add .NS suffix for yfinance
            symbols_ns = [f"{symbol}.NS" for symbol in symbols if isinstance(symbol, str)]

            # Filter out index symbols and ETFs
            exclude_keywords = ['NIFTY', 'SENSEX', 'INDEX', 'ETF', '-EQ', 'BHARAT']
            symbols_filtered = [s for s in symbols_ns if not any(kw in s.upper() for kw in exclude_keywords)]

            st.success(f"✅ Loaded {len(symbols_filtered)} NSE equity symbols")
            return symbols_filtered

        except Exception as e:
            st.warning(f"⚠️ Could not fetch NSE symbol list: {str(e)}")
            st.info("Falling back to curated NSE symbol list...")
            return DataCollector._get_fallback_nse_symbols()

    @staticmethod
    def load_all_bse_symbols() -> List[str]:
        """Load ALL BSE equity symbols"""
        try:
            st.info("Loading BSE equity symbols...")

            # BSE symbol list (top 2000+ active stocks)
            # Since BSE API requires individual scrip codes, we use a curated list
            symbols_bo = DataCollector._get_fallback_bse_symbols()

            st.success(f"✅ Loaded {len(symbols_bo)} BSE equity symbols")
            return symbols_bo

        except Exception as e:
            st.warning(f"⚠️ Error loading BSE symbols: {str(e)}")
            return DataCollector._get_fallback_bse_symbols()

    @staticmethod
    def _get_fallback_nse_symbols() -> List[str]:
        """Fallback curated list of NSE symbols (top 500+)"""
        # Expanded NSE symbol list (500+ stocks)
        base_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
            'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'HCLTECH',
            'SUNPHARMA', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND', 'ONGC', 'NTPC', 'POWERGRID',
            'M&M', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'ADANIPORTS', 'COALINDIA', 'DIVISLAB',
            'BAJAJFINSV', 'DRREDDY', 'GRASIM', 'CIPLA', 'EICHERMOT', 'HEROMOTOCO', 'HINDALCO',
            'JSWSTEEL', 'BRITANNIA', 'INDUSINDBK', 'APOLLOHOSP', 'BPCL', 'ADANIENT', 'SHREECEM',
            'TATACONSUM', 'UPL', 'BAJAJ-AUTO', 'SBILIFE', 'HDFCLIFE', 'DABUR', 'GODREJCP', 'MARICO',
            'PIDILITIND', 'HAVELLS', 'VOLTAS', 'BANDHANBNK', 'IDEA', 'VEDL', 'SAIL', 'ZEEL', 'PNB',
            'BANKBARODA', 'CANBK', 'RBLBANK', 'FEDERALBNK', 'PEL', 'CONCOR', 'LICHSGFIN', 'NMDC',
            'GAIL', 'IOC', 'MOTHERSON', 'BOSCHLTD', 'AMBUJACEM', 'ACC', 'BERGEPAINT', 'DLF',
            'JINDALSTEL', 'TATAPOWER', 'AUROPHARMA', 'LUPIN', 'BIOCON', 'TORNTPHARM', 'MCDOWELL-N',
            'SIEMENS', 'ADANIGREEN', 'ADANITRANS', 'RECLTD', 'PFC', 'HINDZINC', 'PAGEIND', 'COLPAL',
            'MUTHOOTFIN', 'TRENT', 'INDIGO', 'MPHASIS', 'MINDTREE', 'LTTS', 'PERSISTENT', 'COFORGE',
            'LALPATHLAB', 'STAR', 'CHOLAFIN', 'BATAINDIA', 'ASHOKLEY', 'BEL', 'CUMMINSIND', 'ESCORTS',
            'EXIDEIND', 'GLENMARK', 'HDFCAMC', 'IBULHSGFIN', 'INDUSTOWER', 'IPCALAB', 'MRF', 'NATIONALUM',
            'NAVINFLUOR', 'PETRONET', 'PIIND', 'RAMCOCEM', 'SRF', 'SRTRANSFIN', 'TATACOMM', 'TORNTPOWER',
            'TVSMOTOR', 'ZYDUSLIFE', 'ABBOTT', 'ABCAPITAL', 'ABFRL', 'ALKEM', 'AMBUJACEM', 'APOLLOTYRE',
            'AUBANK', 'AUROPHARMA', 'BALKRISIND', 'BANDHANBNK', 'BANKBARODA', 'BHARATFORG', 'BHEL',
            'BIOCON', 'BOSCHLTD', 'BSOFT', 'CADILAHC', 'CANBK', 'CHAMBLFERT', 'CHOLAFIN', 'COROMANDEL',
            'CREDITACC', 'CROMPTON', 'CUB', 'CUMMINSIND', 'DABUR', 'DEEPAKNTR', 'DELTACORP', 'DIXON',
            'DLF', 'DMART', 'DRREDDY', 'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL',
            'GLENMARK', 'GMRINFRA', 'GNFC', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GSPL',
            'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFC', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'IBULHSGFIN', 'ICICIBANK',
            'ICICIGI', 'ICICIPRULI', 'IDEA', 'IDFCFIRSTB', 'IEX', 'IGL', 'INDHOTEL', 'INDIACEM',
            'INDIAMART', 'INDIANB', 'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 'INFY', 'IOC', 'IPCALAB',
            'IRCTC', 'ITC', 'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'KOTAKBANK', 'L&TFH',
            'LALPATHLAB', 'LAURUSLABS', 'LICHSGFIN', 'LT', 'LTI', 'LTTS', 'LUPIN', 'M&M', 'M&MFIN',
            'MANAPPURAM', 'MARICO', 'MARUTI', 'MCDOWELL-N', 'MCX', 'METROPOLIS', 'MFSL', 'MGL',
            'MINDTREE', 'MOTHERSON', 'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NAVINFLUOR',
            'NESTLEIND', 'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'OIL', 'ONGC', 'PAGEIND', 'PEL',
            'PERSISTENT', 'PETRONET', 'PFC', 'PIDILITIND', 'PIIND', 'PNB', 'POLYCAB', 'POWERGRID',
            'PVR', 'RAIN', 'RAJESHEXPO', 'RAMCOCEM', 'RBLBANK', 'RECLTD', 'RELIANCE', 'SAIL', 'SBICARD',
            'SBILIFE', 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF', 'SRTRANSFIN', 'STAR', 'SUNPHARMA', 'SUNTV',
            'SYNGENE', 'TATACHEM', 'TATACOMM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL',
            'TCS', 'TECHM', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TVSMOTOR', 'UBL', 'ULTRACEMCO',
            'UPL', 'VEDL', 'VOLTAS', 'WHIRLPOOL', 'WIPRO', 'ZEEL', 'ZYDUSLIFE'
        ]

        # Remove duplicates and add .NS suffix
        unique_symbols = list(set(base_symbols))
        return [f"{symbol}.NS" for symbol in unique_symbols]

    @staticmethod
    def _get_fallback_bse_symbols() -> List[str]:
        """Fallback curated list of BSE symbols (top 500+)"""
        # Expanded BSE symbol list using .BO suffix
        base_symbols = [
            '500325', '532540', '500180', '500209', '500696', '532174', '500112', '532454', '500875',
            '500247', '500510', '532215', '500820', '532500', '532977', '500182', '524715', '500114',
            '507685', '532538', '500790', '500312', '532555', '532898', '500520', '500570', '500400',
            '532755', '500410', '533278', '532281', '532134', '500124', '500300', '500087', '505200',
            '532454', '500440', '500228', '532712', '500676', '532921', '500547', '532454', '500387',
            '500696', '532522', '500034', '540376', '543066', '532715', '500084', '532478', '500010',
            '500104', '500101', '524715', '500490', '532281', '532712', '500425', '500570', '500302',
            '533098', '500425', '500413', '500408', '500087', '532281', '500031', '500295', '500188',
            '500477', '500387', '500570', '500096', '500820', '500440', '500790', '500550', '500124',
            '532281', '500209', '500696', '532174', '500112', '532454', '500875', '500247', '500510',
            '532215', '500820', '532500', '532977', '500182', '524715', '500114', '507685', '532538',
            '500790', '500312', '532555', '532898', '500520', '500570', '500400', '532755', '500410',
            '533278', '532281', '532134', '500124', '500300', '500087', '505200', '532454', '500440'
        ]

        # Remove duplicates and add .BO suffix
        unique_symbols = list(set(base_symbols))
        return [f"{symbol}.BO" for symbol in unique_symbols]

    @staticmethod
    def safe_fetch(symbol: str, start_date: datetime, end_date: datetime, min_days: int = 15) -> pd.DataFrame:
        """
        Safely fetch data for a single symbol with retry logic
        Returns None if fetch fails after retries or if data is insufficient
        """
        max_retries = 5
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                # Check if data is valid
                if df.empty or len(df) < min_days:
                    return None

                # Format data
                df = df.reset_index()
                df['Symbol'] = symbol
                df = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Silently skip on final failure
                    return None

        return None

    @staticmethod
    def fetch_full_market_data(symbols: List[str], max_years: int = 3, min_days: int = 15) -> pd.DataFrame:
        """
        Fetch data for ALL symbols using chunking and parallel processing with checkpoints
        - Splits symbols into chunks of 100
        - Uses ThreadPoolExecutor with 100 workers per chunk
        - Saves checkpoints as Parquet files
        - Automatically resumes from last checkpoint
        - Returns merged DataFrame
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max_years*365)

        # Create checkpoint directory
        checkpoint_dir = "data_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Split into chunks of 100
        chunk_size = 100
        chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
        total_chunks = len(chunks)

        st.info(f"📦 Processing {len(symbols)} symbols in {total_chunks} chunks of {chunk_size}")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            batch_file = os.path.join(checkpoint_dir, f"data_batch_{chunk_idx}.parquet")

            # Check if batch already exists
            if os.path.exists(batch_file):
                status_text.text(f"✓ Chunk {chunk_idx+1}/{total_chunks} already processed (skipping)")
                progress_bar.progress((chunk_idx + 1) / total_chunks)
                continue

            status_text.text(f"⏳ Processing chunk {chunk_idx+1}/{total_chunks} ({len(chunk)} symbols)...")

            # Parallel fetch within chunk
            chunk_data = []

            with ThreadPoolExecutor(max_workers=100) as executor:
                future_to_symbol = {
                    executor.submit(DataCollector.safe_fetch, symbol, start_date, end_date, min_days): symbol
                    for symbol in chunk
                }

                completed = 0
                for future in as_completed(future_to_symbol):
                    completed += 1
                    result = future.result()

                    if result is not None:
                        chunk_data.append(result)

                    # Update sub-progress
                    status_text.text(
                        f"⏳ Chunk {chunk_idx+1}/{total_chunks}: {completed}/{len(chunk)} symbols processed"
                    )

            # Save chunk to Parquet
            if chunk_data:
                chunk_df = pd.concat(chunk_data, ignore_index=True)
                chunk_df.to_parquet(batch_file, compression='gzip', index=False)
                status_text.text(f"✅ Chunk {chunk_idx+1}/{total_chunks} saved ({len(chunk_data)} stocks, {len(chunk_df)} records)")
            else:
                # Create empty marker file
                pd.DataFrame().to_parquet(batch_file, index=False)
                status_text.text(f"⚠️ Chunk {chunk_idx+1}/{total_chunks} had no valid data")

            # Update progress
            progress_bar.progress((chunk_idx + 1) / total_chunks)

            # Small delay between chunks
            time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        # Merge all batch files
        st.info("🔄 Merging all batch files...")

        all_data = []
        for chunk_idx in range(total_chunks):
            batch_file = os.path.join(checkpoint_dir, f"data_batch_{chunk_idx}.parquet")

            if os.path.exists(batch_file):
                try:
                    batch_df = pd.read_parquet(batch_file)
                    if not batch_df.empty:
                        all_data.append(batch_df)
                except Exception as e:
                    st.warning(f"⚠️ Could not read batch {chunk_idx}: {str(e)}")

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Save final merged file
        final_file = "full_india_market.parquet"
        combined_df.to_parquet(final_file, compression='gzip', index=False)

        st.success(f"✅ Saved full market data: {final_file} ({len(combined_df)} records, {combined_df['Symbol'].nunique()} stocks)")

        return combined_df

    @staticmethod
    def fetch_historical_data(symbols: List[str], max_years: int = 3, min_days: int = 15) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for all symbols (original fast mode)
        Max: 3 years, Min: 15 days
        Returns combined dataframe with columns: Symbol, Date, Open, High, Low, Close, Volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max_years*365)

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

                # Accept stocks with minimum 15 days of data
                if df.empty or len(df) < min_days:
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
# 2. UC/LC DETECTION MODULE
# ====================================================

class CircuitDetector:
    """Detects Upper Circuit and Lower Circuit days"""

    @staticmethod
    def detect_circuits(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect UC and LC days using strict 1% tolerance formulas.

        UC Day: (High - Close) / Close <= 0.01
        LC Day: (Close - Low) / Close <= 0.01

        These are the ONLY conditions used - no gain/loss requirements.
        """
        df = df.copy()

        # Group by symbol for calculations
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            symbol_df = df[mask].copy().sort_values('Date')

            # Calculate previous close for gain/loss tracking (used for features, not UC/LC detection)
            symbol_df['prev_close'] = symbol_df['Close'].shift(1)
            symbol_df['close_gain_pct'] = ((symbol_df['Close'] - symbol_df['prev_close']) /
                                          symbol_df['prev_close'] * 100)

            # Calculate distance from High and Low (as percentages for reference)
            symbol_df['dist_from_high_pct'] = ((symbol_df['High'] - symbol_df['Close']) /
                                               symbol_df['Close'] * 100)
            symbol_df['dist_from_low_pct'] = ((symbol_df['Close'] - symbol_df['Low']) /
                                              symbol_df['Close'] * 100)

            # Detect UC: (High - Close) / Close <= 0.01
            symbol_df['is_uc'] = (
                (symbol_df['High'] - symbol_df['Close']) / symbol_df['Close'] <= 0.01
            ).astype(int)

            # Detect LC: (Close - Low) / Close <= 0.01
            symbol_df['is_lc'] = (
                (symbol_df['Close'] - symbol_df['Low']) / symbol_df['Close'] <= 0.01
            ).astype(int)

            # For UC days, calculate days until next LC
            symbol_df['days_to_lc'] = np.nan

            uc_indices = symbol_df[symbol_df['is_uc'] == 1].index.tolist()
            for uc_idx in uc_indices:
                # Find next LC after this UC
                future_df = symbol_df.loc[uc_idx:].iloc[1:]  # Skip current day
                lc_days = future_df[future_df['is_lc'] == 1]

                if not lc_days.empty:
                    first_lc_idx = lc_days.index[0]
                    days_diff = (symbol_df.loc[first_lc_idx, 'Date'] -
                               symbol_df.loc[uc_idx, 'Date']).days
                    symbol_df.loc[uc_idx, 'days_to_lc'] = days_diff

            # Update main dataframe
            df.loc[mask, symbol_df.columns] = symbol_df.values

        return df


# ====================================================
# 3. LABEL GENERATION MODULE
# ====================================================

class LabelGenerator:
    """Generates labels for ML models"""

    @staticmethod
    def generate_uc_prediction_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels to predict: Will stock hit UC tomorrow?
        Label = 1 if next day is UC, else 0
        """
        df = df.copy()

        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            symbol_df = df[mask].copy().sort_values('Date')

            # Shift is_uc by -1 to get tomorrow's UC status
            symbol_df['will_hit_uc_tomorrow'] = symbol_df['is_uc'].shift(-1).fillna(0).astype(int)

            df.loc[mask, 'will_hit_uc_tomorrow'] = symbol_df['will_hit_uc_tomorrow'].values

        return df

    @staticmethod
    def generate_uc_to_lc_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        For stocks at UC, generate labels for days until LC
        This is for regression model
        """
        # Already computed in CircuitDetector.detect_circuits as 'days_to_lc'
        return df


# ====================================================
# 4. FEATURE ENGINEERING MODULE
# ====================================================

class FeatureEngineer:
    """Computes all required features"""

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for UC prediction:
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
        - RSI (14 period)
        - MACD (12/26/9)
        - OBV (On-Balance Volume)
        - Bollinger Bands (20 period, 2 std)
        - Support/Resistance levels (20 period)
        """
        df = df.copy()

        # Group by symbol for calculations
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            symbol_df = df[mask].copy().sort_values('Date')

            # Returns
            symbol_df['return_1d'] = symbol_df['Close'].pct_change(1) * 100
            symbol_df['return_3d'] = symbol_df['Close'].pct_change(3) * 100
            symbol_df['return_5d'] = symbol_df['Close'].pct_change(5) * 100
            symbol_df['return_10d'] = symbol_df['Close'].pct_change(10) * 100

            # Volume ratio
            symbol_df['avg_volume_20d'] = symbol_df['Volume'].rolling(20).mean()
            symbol_df['volume_ratio'] = symbol_df['Volume'] / (symbol_df['avg_volume_20d'] + 1)

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
            symbol_df['dist_from_5d_high'] = (symbol_df['Close'] - symbol_df['high_5d']) / symbol_df['high_5d'] * 100
            symbol_df['dist_from_20d_high'] = (symbol_df['Close'] - symbol_df['high_20d']) / symbol_df['high_20d'] * 100

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
                                       (symbol_df['Close'].shift(1) + 1e-10) * 100)

            # Compression ratio
            symbol_df['compression_ratio'] = (symbol_df['High'] - symbol_df['Low']) / (symbol_df['Close'] + 1e-10)

            # ====================================================
            # NEW TECHNICAL INDICATORS
            # ====================================================

            # 1. RSI (14 period)
            delta = symbol_df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            symbol_df['rsi_14'] = 100 - (100 / (1 + rs))

            # 2. MACD (12/26/9)
            ema_12 = symbol_df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = symbol_df['Close'].ewm(span=26, adjust=False).mean()
            symbol_df['macd_line'] = ema_12 - ema_26
            symbol_df['signal_line'] = symbol_df['macd_line'].ewm(span=9, adjust=False).mean()
            symbol_df['macd_histogram'] = symbol_df['macd_line'] - symbol_df['signal_line']

            # 3. OBV (On-Balance Volume)
            obv = np.zeros(len(symbol_df))
            obv[0] = symbol_df['Volume'].iloc[0]
            for i in range(1, len(symbol_df)):
                if symbol_df['Close'].iloc[i] > symbol_df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + symbol_df['Volume'].iloc[i]
                elif symbol_df['Close'].iloc[i] < symbol_df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - symbol_df['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            symbol_df['obv'] = obv

            # 4. Bollinger Bands (20 period, 2 std)
            bb_ma = symbol_df['Close'].rolling(window=20).mean()
            bb_std = symbol_df['Close'].rolling(window=20).std()
            symbol_df['bb_upper'] = bb_ma + (2 * bb_std)
            symbol_df['bb_lower'] = bb_ma - (2 * bb_std)
            symbol_df['bb_width'] = (symbol_df['bb_upper'] - symbol_df['bb_lower']) / (symbol_df['Close'] + 1e-10)

            # 5. Support/Resistance levels (20 period)
            symbol_df['support_20'] = symbol_df['Low'].rolling(window=20).min()
            symbol_df['resistance_20'] = symbol_df['High'].rolling(window=20).max()
            symbol_df['distance_from_resistance'] = symbol_df['Close'] / (symbol_df['resistance_20'] + 1e-10)
            symbol_df['distance_from_support'] = symbol_df['Close'] / (symbol_df['support_20'] + 1e-10)

            # Update main dataframe
            df.loc[mask, symbol_df.columns] = symbol_df.values

        return df


# ====================================================
# 5. MODEL TRAINING MODULE
# ====================================================

class ModelTrainer:
    """Trains and evaluates ML models"""

    def __init__(self):
        self.classification_models = {}
        self.regression_models = {}
        self.best_classifier = None
        self.best_classifier_name = None
        self.best_regressor = None
        self.best_regressor_name = None
        self.feature_columns = None
        self.classification_metrics = {}
        self.regression_metrics = {}
        self.feature_importance = None

    def prepare_classification_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare features and labels for UC prediction (classification)"""
        # Feature columns
        self.feature_columns = [
            'return_1d', 'return_3d', 'return_5d', 'return_10d',
            'volume_ratio', 'MA5', 'MA10', 'MA20', 'MA50',
            'slope_MA10_MA20', 'slope_MA5_MA20',
            'dist_from_5d_high', 'dist_from_20d_high',
            'consecutive_up_days', 'body_wick_ratio', 'delivery_ratio',
            'ATR_14', 'gap_up_pct', 'compression_ratio',
            'rsi_14', 'macd_line', 'signal_line', 'macd_histogram',
            'obv', 'bb_upper', 'bb_lower', 'bb_width',
            'support_20', 'resistance_20', 'distance_from_resistance', 'distance_from_support'
        ]

        # Remove rows with NaN in features or labels
        df_clean = df[self.feature_columns + ['will_hit_uc_tomorrow']].dropna()

        X = df_clean[self.feature_columns]
        y = df_clean['will_hit_uc_tomorrow']

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def prepare_regression_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare features and labels for UC to LC days prediction (regression)"""
        # Only use rows where stock hit UC and we know days_to_lc
        df_uc = df[(df['is_uc'] == 1) & (df['days_to_lc'].notna())].copy()

        if df_uc.empty:
            return None, None, None, None

        df_clean = df_uc[self.feature_columns + ['days_to_lc']].dropna()

        if len(df_clean) < 10:
            return None, None, None, None

        X = df_clean[self.feature_columns]
        y = df_clean['days_to_lc']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train models to predict UC tomorrow"""

        # 1. XGBoost
        st.write("Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.classification_models['XGBoost'] = xgb_model

        # 2. Random Forest
        st.write("Training RandomForest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.classification_models['RandomForest'] = rf_model

        # 3. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            st.write("Training LightGBM Classifier...")
            lgb_model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            self.classification_models['LightGBM'] = lgb_model

        # Evaluate all models
        st.write("\nEvaluating classification models...")
        best_roc_auc = 0

        for name, model in self.classification_models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            self.classification_metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            if self.classification_metrics[name]['roc_auc'] > best_roc_auc:
                best_roc_auc = self.classification_metrics[name]['roc_auc']
                self.best_classifier_name = name
                self.best_classifier = model

        # Extract feature importance from best model
        if hasattr(self.best_classifier, 'feature_importances_'):
            importance = self.best_classifier.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train models to predict days from UC to LC"""

        # 1. XGBoost Regressor
        st.write("Training XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        xgb_reg.fit(X_train, y_train)
        self.regression_models['XGBoost'] = xgb_reg

        # 2. Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        st.write("Training RandomForest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(X_train, y_train)
        self.regression_models['RandomForest'] = rf_reg

        # 3. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            st.write("Training LightGBM Regressor...")
            lgb_reg = lgb.LGBMRegressor(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
                verbose=-1
            )
            lgb_reg.fit(X_train, y_train)
            self.regression_models['LightGBM'] = lgb_reg

        # Evaluate all models
        st.write("\nEvaluating regression models...")
        best_mae = float('inf')

        for name, model in self.regression_models.items():
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            self.regression_metrics[name] = {
                'mae': mae,
                'rmse': rmse
            }

            if mae < best_mae:
                best_mae = mae
                self.best_regressor_name = name
                self.best_regressor = model

    def get_classification_metrics_summary(self) -> pd.DataFrame:
        """Return classification metrics summary as dataframe"""
        metrics_data = []
        for name, metrics in self.classification_metrics.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        return pd.DataFrame(metrics_data)

    def get_regression_metrics_summary(self) -> pd.DataFrame:
        """Return regression metrics summary as dataframe"""
        metrics_data = []
        for name, metrics in self.regression_metrics.items():
            metrics_data.append({
                'Model': name,
                'MAE (days)': f"{metrics['mae']:.2f}",
                'RMSE (days)': f"{metrics['rmse']:.2f}"
            })
        return pd.DataFrame(metrics_data)

    def save_model(self, filepath: str):
        """Save both models and feature columns"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.best_classifier,
                'classifier_name': self.best_classifier_name,
                'regressor': self.best_regressor,
                'regressor_name': self.best_regressor_name,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance
            }, f)

    @staticmethod
    def load_model(filepath: str):
        """Load saved models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        trainer = ModelTrainer()
        trainer.best_classifier = data['classifier']
        trainer.best_classifier_name = data['classifier_name']
        trainer.best_regressor = data.get('regressor')
        trainer.best_regressor_name = data.get('regressor_name')
        trainer.feature_columns = data['feature_columns']
        trainer.feature_importance = data['feature_importance']
        return trainer


# ====================================================
# 6. UC CONDITION DISCOVERY MODULE
# ====================================================

class UCConditionDiscovery:
    """Discovers and summarizes UC conditions"""

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
        summary += "- Stocks close to highs are more likely to hit UC\n"
        summary += "- Price compression often precedes breakouts\n"

        return summary

    @staticmethod
    def analyze_uc_to_lc_patterns(df: pd.DataFrame) -> Dict:
        """Analyze historical UC to LC patterns"""
        uc_data = df[df['is_uc'] == 1].copy()

        if uc_data.empty:
            return {
                'total_uc_events': 0,
                'uc_with_lc': 0,
                'avg_days_to_lc': 0,
                'median_days_to_lc': 0,
                'min_days_to_lc': 0,
                'max_days_to_lc': 0
            }

        uc_with_lc = uc_data[uc_data['days_to_lc'].notna()]

        return {
            'total_uc_events': len(uc_data),
            'uc_with_lc': len(uc_with_lc),
            'avg_days_to_lc': uc_with_lc['days_to_lc'].mean() if not uc_with_lc.empty else 0,
            'median_days_to_lc': uc_with_lc['days_to_lc'].median() if not uc_with_lc.empty else 0,
            'min_days_to_lc': uc_with_lc['days_to_lc'].min() if not uc_with_lc.empty else 0,
            'max_days_to_lc': uc_with_lc['days_to_lc'].max() if not uc_with_lc.empty else 0
        }


# ====================================================
# 7. PREDICTION MODULE
# ====================================================

class Predictor:
    """Predicts UC probability and UC to LC duration for any stock"""

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def predict_uc(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Predict UC probability for a symbol
        Returns: probability, current status, trigger levels, reasoning
        """
        # Add exchange suffix
        if exchange == 'NSE':
            symbol_yf = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        else:
            symbol_yf = f"{symbol}.BO" if not symbol.endswith('.BO') else symbol

        # Fetch last 90 days of data (buffer for feature calculation)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        try:
            ticker = yf.Ticker(symbol_yf)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty or len(df) < 15:
                return {'error': 'Insufficient data (minimum 15 days required)'}

            # Prepare data
            df = df.reset_index()
            df['Symbol'] = symbol_yf
            df = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Detect circuits
            circuit_detector = CircuitDetector()
            df = circuit_detector.detect_circuits(df)

            # Compute features
            feature_eng = FeatureEngineer()
            df = feature_eng.compute_features(df)

            # Get latest row with all features
            df_latest = df[self.trainer.feature_columns].dropna()
            if df_latest.empty:
                return {'error': 'Could not compute features'}

            latest_features = df_latest.iloc[-1:]
            latest_row = df.iloc[-1]

            # Predict UC probability
            uc_probability = self.trainer.best_classifier.predict_proba(latest_features)[0, 1]

            # Check if currently at UC
            current_uc_status = "AT UPPER CIRCUIT ✓" if latest_row['is_uc'] == 1 else "Not at UC"

            # If at UC, predict days to LC
            days_to_lc_pred = None
            if latest_row['is_uc'] == 1 and self.trainer.best_regressor:
                days_to_lc_pred = self.trainer.best_regressor.predict(latest_features)[0]

            # Calculate trigger price (5-day high + 1%)
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
                'uc_probability': uc_probability,
                'current_uc_status': current_uc_status,
                'is_currently_uc': int(latest_row['is_uc']),
                'days_to_lc_prediction': days_to_lc_pred,
                'trigger_price': trigger_price,
                'volume_threshold': volume_threshold,
                'reasoning': reasoning,
                'current_price': latest_row['Close'],
                'close_gain_pct': latest_row.get('close_gain_pct', 0)
            }

        except Exception as e:
            return {'error': str(e)}

    def bulk_screen(self, symbols: List[str]) -> pd.DataFrame:
        """Screen multiple symbols and return ranked by UC probability"""
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, symbol in enumerate(symbols):
            status_text.text(f"Screening {symbol} ({idx+1}/{len(symbols)})...")

            # Determine exchange from symbol
            exchange = 'NSE' if symbol.endswith('.NS') else 'BSE'
            pred = self.predict_uc(symbol, exchange)

            if 'error' not in pred:
                results.append({
                    'Symbol': pred['symbol'],
                    'UC_Probability': f"{pred['uc_probability']:.4f}",
                    'Current_Status': pred['current_uc_status'],
                    'Current_Price': f"{pred['current_price']:.2f}",
                    'Gain_Today_%': f"{pred['close_gain_pct']:.2f}",
                    'Trigger_Price': f"{pred['trigger_price']:.2f}",
                    'Volume_Threshold': f"{pred['volume_threshold']:.0f}"
                })

            progress_bar.progress((idx + 1) / len(symbols))
            time.sleep(0.1)

        progress_bar.empty()
        status_text.empty()

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results['UC_Probability'] = df_results['UC_Probability'].astype(float)
            df_results = df_results.sort_values('UC_Probability', ascending=False)

        return df_results


# ====================================================
# 8. STREAMLIT GUI
# ====================================================

def main():
    st.set_page_config(page_title="UC Rally Detector", layout="wide")

    st.title("🚀 Upper-Circuit Rally Detection System")
    st.markdown("**Strategy: Buy stocks at UC, Hold until LC**")

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Fetch & Training", "Predict UC for Any Stock", "Bulk UC Screener", "UC to LC Analyzer"]
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

        # Circuit detection info (fixed formulas, not user-configurable)
        st.subheader("Circuit Detection Rules")
        st.info("📌 UC Detection: Close nearly equals High → (High - Close) / Close ≤ 1%")
        st.info("📌 LC Detection: Close nearly equals Low → (Close - Low) / Close ≤ 1%")
        st.info("📌 Data Range: Maximum 3 years, Minimum 15 days")

        st.markdown("---")

        # Data fetching with mode selection
        st.subheader("Step 1: Fetch Historical Data")

        # Fetch mode toggle
        fetch_mode = st.radio(
            "Fetch Mode:",
            ["Fast Mode (200 stocks)", "Full India Market (~7000 stocks)"],
            help="Fast Mode: Quick fetch of 200 major stocks\nFull Market: Comprehensive fetch of all NSE+BSE stocks with checkpoints"
        )

        if st.button("🔄 Fetch data", type="primary"):
            with st.spinner("Fetching data..."):
                collector = DataCollector()

                if fetch_mode == "Fast Mode (200 stocks)":
                    # Fast mode - original implementation
                    nse_symbols = collector.get_nse_symbols()
                    bse_symbols = collector.get_bse_symbols()
                    all_symbols = nse_symbols + bse_symbols

                    st.info(f"Fast Mode: Fetching data for {len(all_symbols)} stocks ({len(nse_symbols)} NSE + {len(bse_symbols)} BSE)...")

                    # Fetch data (max 3 years, min 15 days)
                    stock_data = collector.fetch_historical_data(all_symbols, max_years=3, min_days=15)

                else:
                    # Full market mode - new implementation
                    nse_symbols = collector.load_all_nse_symbols()
                    bse_symbols = collector.load_all_bse_symbols()
                    all_symbols = nse_symbols + bse_symbols

                    st.info(f"Full Market Mode: Fetching data for {len(all_symbols)} stocks ({len(nse_symbols)} NSE + {len(bse_symbols)} BSE)...")

                    # Fetch data with chunking and checkpoints
                    stock_data = collector.fetch_full_market_data(all_symbols, max_years=3, min_days=15)

                if not stock_data.empty:
                    st.success(f"✅ Successfully fetched data for {stock_data['Symbol'].nunique()} stocks!")
                    st.write(f"Total records: {len(stock_data):,}")

                    # Detect circuits
                    st.info("Detecting UC and LC days...")
                    circuit_detector = CircuitDetector()
                    stock_data = circuit_detector.detect_circuits(stock_data)

                    # Generate labels
                    st.info("Generating prediction labels...")
                    label_gen = LabelGenerator()
                    stock_data = label_gen.generate_uc_prediction_labels(stock_data)

                    # Compute features
                    st.info("Computing features...")
                    feature_eng = FeatureEngineer()
                    stock_data = feature_eng.compute_features(stock_data)

                    st.session_state.stock_data = stock_data
                    st.session_state.data_loaded = True

                    st.success("✅ Data preparation complete!")

                    # Show statistics
                    total_uc = stock_data['is_uc'].sum()
                    total_lc = stock_data['is_lc'].sum()
                    st.write(f"**UC Days detected:** {total_uc:,}")
                    st.write(f"**LC Days detected:** {total_lc:,}")

                    st.dataframe(stock_data.head(10))
                else:
                    st.error("Failed to fetch data. Please try again.")

        st.markdown("---")

        # Model training
        st.subheader("Step 2: Train UC Prediction Models")
        if st.button("🤖 Train Models", type="primary", disabled=not st.session_state.data_loaded):
            if st.session_state.stock_data is None:
                st.error("Please fetch data first!")
            else:
                with st.spinner("Training models..."):
                    trainer = ModelTrainer()

                    # Train classification model (predict UC tomorrow)
                    st.info("Training UC Tomorrow Prediction (Classification)...")
                    X_train, X_test, y_train, y_test = trainer.prepare_classification_data(st.session_state.stock_data)

                    st.write(f"Training samples: {len(X_train):,}")
                    st.write(f"Test samples: {len(X_test):,}")
                    st.write(f"UC cases: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")

                    trainer.train_classification_models(X_train, X_test, y_train, y_test)

                    # Train regression model (predict days to LC)
                    st.info("Training UC to LC Days Prediction (Regression)...")
                    reg_data = trainer.prepare_regression_data(st.session_state.stock_data)

                    if reg_data[0] is not None:
                        X_train_reg, X_test_reg, y_train_reg, y_test_reg = reg_data
                        st.write(f"UC→LC Training samples: {len(X_train_reg):,}")
                        trainer.train_regression_models(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
                    else:
                        st.warning("⚠️ Insufficient UC→LC data for regression model")

                    st.session_state.trainer = trainer
                    st.session_state.model_trained = True

                    # Save model
                    trainer.save_model('uc_rally_model.pkl')

                    st.success(f"✅ Training complete!")
                    st.success(f"Best Classifier: {trainer.best_classifier_name}")
                    if trainer.best_regressor_name:
                        st.success(f"Best Regressor: {trainer.best_regressor_name}")

        # Display metrics
        if st.session_state.model_trained and st.session_state.trainer:
            st.markdown("---")
            st.subheader("📈 Model Performance")

            trainer = st.session_state.trainer

            # Classification metrics
            st.write("**UC Tomorrow Prediction (Classification):**")
            metrics_df = trainer.get_classification_metrics_summary()
            st.dataframe(metrics_df)

            # Best model confusion matrix
            if trainer.best_classifier_name in trainer.classification_metrics:
                st.write(f"**Confusion Matrix ({trainer.best_classifier_name}):**")
                cm = trainer.classification_metrics[trainer.best_classifier_name]['confusion_matrix']
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual: No UC', 'Actual: UC'],
                    columns=['Pred: No UC', 'Pred: UC']
                )
                st.dataframe(cm_df)

            # Regression metrics
            if trainer.regression_metrics:
                st.write("**UC to LC Days Prediction (Regression):**")
                reg_metrics_df = trainer.get_regression_metrics_summary()
                st.dataframe(reg_metrics_df)

            # Feature importance
            st.markdown("---")
            st.subheader("🎯 Feature Importance (Top 20)")
            top_20_features = UCConditionDiscovery.get_top_features(trainer.feature_importance, 20)
            st.dataframe(top_20_features)

            # UC Conditions Summary
            st.markdown("---")
            st.subheader("💡 Discovered UC Conditions")
            summary = UCConditionDiscovery.generate_summary(trainer.feature_importance)
            st.markdown(summary)

            # UC to LC Analysis
            st.markdown("---")
            st.subheader("📊 Historical UC to LC Patterns")
            patterns = UCConditionDiscovery.analyze_uc_to_lc_patterns(st.session_state.stock_data)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total UC Events", f"{patterns['total_uc_events']:,}")
            with col2:
                st.metric("UC→LC Events", f"{patterns['uc_with_lc']:,}")
            with col3:
                st.metric("Avg Days to LC", f"{patterns['avg_days_to_lc']:.1f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Median Days to LC", f"{patterns['median_days_to_lc']:.1f}")
            with col2:
                st.metric("Min Days to LC", f"{patterns['min_days_to_lc']:.0f}")
            with col3:
                st.metric("Max Days to LC", f"{patterns['max_days_to_lc']:.0f}")

    # ====================================================
    # PAGE 2: PREDICT UC FOR ANY STOCK
    # ====================================================
    elif page == "Predict UC for Any Stock":
        st.header("🔮 Predict UC for Individual Stock")

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

        if st.button("🎯 Predict UC Probability", type="primary"):
            predictor = Predictor(st.session_state.trainer)

            with st.spinner(f"Analyzing {symbol_input}..."):
                result = predictor.predict_uc(symbol_input, exchange)

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("✅ Prediction Complete!")

                # Current status
                if result['is_currently_uc'] == 1:
                    st.success(f"🔥 **{result['symbol']}** is currently AT UPPER CIRCUIT!")

                    if result['days_to_lc_prediction']:
                        st.info(f"📅 Predicted days to Lower Circuit: **{result['days_to_lc_prediction']:.1f} days**")
                else:
                    st.info(f"📊 **{result['symbol']}** is not currently at UC")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("UC Tomorrow Probability", f"{result['uc_probability']*100:.2f}%")
                with col2:
                    st.metric("Current Price", f"₹{result['current_price']:.2f}")
                with col3:
                    st.metric("Today's Gain", f"{result['close_gain_pct']:.2f}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Trigger Price", f"₹{result['trigger_price']:.2f}")
                with col2:
                    st.metric("Volume Threshold", f"{result['volume_threshold']:,.0f}")

                # Reasoning
                st.markdown("---")
                st.subheader("🔍 Key Signals (Top 5 Features)")
                for reason in result['reasoning']:
                    st.write(f"• {reason}")

                # Interpretation
                st.markdown("---")
                st.subheader("📊 Interpretation")
                if result['uc_probability'] > 0.7:
                    st.success("🟢 **HIGH PROBABILITY** - Strong UC signals detected!")
                    st.write("**Action**: Monitor for entry. Consider buying if volume crosses threshold.")
                elif result['uc_probability'] > 0.4:
                    st.warning("🟡 **MODERATE PROBABILITY** - Some UC signals present.")
                    st.write("**Action**: Watch closely. Wait for confirmation.")
                else:
                    st.info("🔵 **LOW PROBABILITY** - Weak UC signals.")
                    st.write("**Action**: Not recommended for UC strategy.")

    # ====================================================
    # PAGE 3: BULK UC SCREENER
    # ====================================================
    elif page == "Bulk UC Screener":
        st.header("📋 Bulk Stock Screener - UC Candidates")

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

                st.subheader("🏆 Top UC Candidates (Ranked by Probability)")

                # Highlight stocks currently at UC
                st.info("🔥 Stocks marked with 'AT UPPER CIRCUIT ✓' are currently at UC!")

                st.dataframe(results_df, use_container_width=True, height=600)

                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"uc_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found.")

    # ====================================================
    # PAGE 4: UC TO LC ANALYZER
    # ====================================================
    elif page == "UC to LC Analyzer":
        st.header("📊 UC to LC Pattern Analyzer")

        if not st.session_state.data_loaded or st.session_state.stock_data is None:
            st.warning("⚠️ No data loaded. Please fetch data first in 'Data Fetch & Training' page.")
            st.stop()

        st.info("Analyzing historical UC to LC patterns from loaded data...")

        data = st.session_state.stock_data

        # Filter UC days
        uc_days = data[data['is_uc'] == 1].copy()

        if uc_days.empty:
            st.warning("No UC days found in the data.")
            st.stop()

        st.write(f"**Total UC Days Found:** {len(uc_days):,}")

        # UC with LC data
        uc_with_lc = uc_days[uc_days['days_to_lc'].notna()]
        st.write(f"**UC Days with subsequent LC:** {len(uc_with_lc):,}")

        if not uc_with_lc.empty:
            # Distribution of days to LC
            st.subheader("📈 Distribution of Days from UC to LC")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(uc_with_lc['days_to_lc'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Days from UC to LC')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Holding Period (UC to LC)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Statistics by symbol
            st.subheader("📊 UC to LC Statistics by Stock")

            symbol_stats = uc_with_lc.groupby('Symbol')['days_to_lc'].agg([
                ('Count', 'count'),
                ('Avg_Days', 'mean'),
                ('Median_Days', 'median'),
                ('Min_Days', 'min'),
                ('Max_Days', 'max')
            ]).round(2).sort_values('Count', ascending=False).reset_index()

            st.dataframe(symbol_stats.head(20), use_container_width=True)

            # Download option
            csv = symbol_stats.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Statistics as CSV",
                data=csv,
                file_name=f"uc_to_lc_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
