import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np

def generate_pro_nasdaq_dataset():
    print("--- 1. INITIALIZING DOWNLOADER (FIXED SCALING) ---")
    
    # Target: NASDAQ Composite
    target_ticker = "^IXIC" 
    
    # Feature Tickers
    tickers = {
        # Global Markets (Prices -> Returns)
        'Japan_Nikkei': '^N225',      
        'UK_FTSE': '^FTSE',           
        'Germany_DAX': '^GDAXI',      
        
        # Macro Indicators (Keep Levels for VIX/Yield, Prices for others)
        'US_10Y_Yield': '^TNX',       # Yield (Level)
        'Volatility_VIX': '^VIX',     # Index (Level)
        'US_Dollar_Index': 'DX-Y.NYB', # Price -> Return
        'Crude_Oil': 'CL=F',          # Price -> Return
        'Gold': 'GC=F',               # Price -> Return
        
        # Risk Sentiment ETFs (Prices -> Returns)
        'Tech_ETF': 'XLK',            
        'Staples_ETF': 'XLP',         
        'Junk_Bonds': 'HYG',          
        'Treasuries': 'TLT'           
    }

    # Fetch MAX available history
    period = "max"
    
    # 1. Get Target Data
    print(f"Downloading Target: {target_ticker} (Max History)...")
    df = yf.download(target_ticker, period=period, progress=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Flatten MultiIndex if necessary
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except IndexError:
            pass # Already flat
            
    # Ensure columns are correct
    if 'Close' not in df.columns:
        # Fallback for some yfinance versions returning different structures
        df = df.xs(target_ticker, axis=1, level=1)
        
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 2. Get Feature Data & FIX SCALING
    print("Downloading External Market Signals & Converting to Returns...")
    external_data = pd.DataFrame()
    
    # List of assets that are PRICES and need conversion to RETURNS
    price_assets = [
        'Japan_Nikkei', 'UK_FTSE', 'Germany_DAX', 
        'US_Dollar_Index', 'Crude_Oil', 'Gold', 
        'Tech_ETF', 'Staples_ETF', 'Junk_Bonds', 'Treasuries'
    ]
    
    for name, ticker in tickers.items():
        print(f"  Fetching {name} ({ticker})...")
        try:
            # Fetch data
            data = yf.download(ticker, period=period, progress=False)
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # Try to get 'Close' for this specific ticker
                try:
                    data_close = data['Close'][ticker]
                except KeyError:
                    data_close = data['Close'].iloc[:, 0]
            else:
                data_close = data['Close']
            
            # ★ CRITICAL FIX: Convert Prices to Returns immediately ★
            if name in price_assets:
                # 30,000 points -> 0.01 (1%)
                external_data[name] = data_close.pct_change()
            else:
                # Keep VIX and Yield as levels (e.g., 20.0, 4.5)
                external_data[name] = data_close
                
        except Exception as e:
            print(f"  Warning: Could not download {name}: {e}")

    # 3. Merge and Align
    print("Merging and aligning data...")
    df = df.join(external_data)
    
    # Fill Forward: Handle different holidays/timezones
    df.ffill(inplace=True)
    
    # Drop NaNs (Cuts to the start of the youngest asset)
    df.dropna(inplace=True)
    
    print(f"History Cut: Data starts from {df.index[0].date()}")

    print("--- 2. ENGINEERING 50+ FEATURES ---")

    # --- BUCKET A: PRICE LAGS (Target) ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'Lag_Return_{lag}'] = df['Close'].pct_change(lag)

    # --- BUCKET B: TREND INDICATORS ---
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

    # --- BUCKET C: VOLATILITY & BANDS ---
    df['Roll_Vol_21'] = df['Close'].pct_change().rolling(21).std()
    
    ma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['Bollinger_Upper'] = ma_20 + (std_20 * 2)
    df['Bollinger_Lower'] = ma_20 - (std_20 * 2)
    df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / ma_20

    # --- BUCKET D: MOMENTUM ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # --- BUCKET E: ADVANCED MACRO RATIOS ---
    # Since Tech_ETF and Staples_ETF are now RETURNS, we can't divide them directly to get a price ratio.
    # Instead, we take the difference in returns (Relative Strength)
    df['Risk_On_Off_Diff'] = df['Tech_ETF'] - df['Staples_ETF']
    df['Credit_Stress_Diff'] = df['Junk_Bonds'] - df['Treasuries']
    
    # Global Lead Signal (Avg of Japan/Germany moves)
    df['Global_Avg_Change'] = (df['Japan_Nikkei'] + df['Germany_DAX']) / 2

    # --- BUCKET F: CALENDAR ---
    df['Day_of_Week'] = df.index.dayofweek
    df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)

    # Final Cleanup
    df.dropna(inplace=True)
    
    # Export
    filename = "nasdaq_new.csv"
    df.to_csv(filename)
    
    print("\n" + "="*50)
    print(f"SUCCESS! Dataset created: {filename}")
    print(f"Time Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total Trading Days: {len(df)}")
    print("="*50)

if __name__ == "__main__":
    generate_pro_nasdaq_dataset()