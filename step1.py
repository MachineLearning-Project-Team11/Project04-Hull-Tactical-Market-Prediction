import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.api.types
import polars as pl
from scipy.optimize import minimize

# --- Metric Implementation ---
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    """
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution.copy()
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        print(f"Warning: Position max {solution['position'].max()} > {MAX_INVESTMENT}")
    if solution['position'].min() < MIN_INVESTMENT:
        print(f"Warning: Position min {solution['position'].min()} < {MIN_INVESTMENT}")

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        return 0.0 
        
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        return 0.0

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

# --- Configuration & Environment Setup ---
IS_KAGGLE = Path('/kaggle').exists()

if IS_KAGGLE:
    INPUT_DIR = Path('/kaggle/input/hull-tactical-market-prediction')
    sys.path.append(str(INPUT_DIR))
else:
    INPUT_DIR = Path('.')
    sys.path.append(os.getcwd())

import kaggle_evaluation.default_inference_server

# --- Data Loading ---
TRAIN_PATH = INPUT_DIR / 'train.csv'
TEST_PATH = INPUT_DIR / 'test.csv'

def load_data(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if 'date_id' in df.columns:
        df = df.sort_values('date_id').reset_index(drop=True)
    return df

# --- Strategy Implementation ---
class MomentumStrategy:
    def __init__(self):
        self.prices = [1.0] # Start at 1.0
        self.signal_mean = 0.0
        self.signal_std = 1.0
        # Optimization parameters
        self.w_sma = 1.0
        self.w_mom = 1.0
        self.scale = 0.5
        self.bias = 1.0
        
    def fit(self, train_df):
        # 1. Reconstruct price history from training data
        # train_df has 'forward_returns'. 
        # We need 'lagged_forward_returns' to simulate the test environment.
        # lagged_ret[t] = forward_ret[t-1]
        lagged_returns = train_df['forward_returns'].shift(1).fillna(0.0).values
        
        # Reconstruct prices: P[t] = P[t-1] * (1 + lagged_ret[t])
        # We start with 1.0.
        # cumprod works: (1+r1)*(1+r2)...
        self.prices = np.cumprod(1 + lagged_returns).tolist()
        
        # 2. Calculate features on the whole history to get stats
        price_series = pd.Series(self.prices)
        
        sma_21 = price_series.rolling(21).mean()
        sma_63 = price_series.rolling(63).mean()
        mom_5 = price_series.pct_change(5)
        
        # Base Signals
        sma_ratio = (sma_21 / sma_63 - 1).fillna(0.0)
        mom_5 = mom_5.fillna(0.0)
        
        # --- Optimization ---
        print("Optimizing strategy parameters...")
        
        # Prepare data for optimization
        # We need to align with train_df for scoring
        # train_df has 'forward_returns', 'risk_free_rate'
        solution = train_df[['forward_returns', 'risk_free_rate']].copy()
        
        def objective(params):
            w_s, w_m, sc, bi = params
            
            # Combine signals
            raw = w_s * sma_ratio + w_m * mom_5
            
            # Normalize (using current mean/std of raw signal)
            # Note: This is slightly circular if we re-calc mean/std every time.
            # Let's just use raw signal directly and let scale/bias handle normalization implicitly.
            # weight = clip(bias + scale * raw, 0, 2)
            
            weights = np.clip(bi + sc * raw, 0.0, 2.0)
            
            submission = pd.DataFrame({'prediction': weights})
            
            try:
                # We want to MAXIMIZE score, so minimize NEGATIVE score
                s = score(solution, submission, 'date_id')
                return -s
            except:
                return 0.0
                
        # Initial guess: w_sma=1, w_mom=1, scale=50 (since raw is small), bias=1
        # raw signal is approx 0.01-0.05 range. 
        # z-score was (raw - mean)/std. std is approx 0.03. So raw/std ~ 30 * raw.
        # 0.5 * z_score ~ 15 * raw.
        # So scale around 10-20 might be good.
        x0 = [1.0, 1.0, 15.0, 1.0]
        
        # Bounds: weights can be negative (contrarian?), scale positive, bias around 1
        # Let's allow flexibility
        res = minimize(objective, x0, method='Nelder-Mead', tol=1e-4, options={'maxiter': 100})
        
        self.w_sma, self.w_mom, self.scale, self.bias = res.x
        print(f"Optimization result: {res.message}")
        print(f"Best Score: {-res.fun:.4f}")
        print(f"Params: w_sma={self.w_sma:.2f}, w_mom={self.w_mom:.2f}, scale={self.scale:.2f}, bias={self.bias:.2f}")
        
        # Keep only the last 63 prices to minimize state size, 
        # but enough to calculate SMA_63 for the next incoming point.
        self.prices = self.prices[-63:]

    def predict_one(self, lagged_return):
        # Update state
        current_price = self.prices[-1] * (1 + lagged_return)
        self.prices.append(current_price)
        if len(self.prices) > 64: # Keep enough history
            self.prices.pop(0)
            
        # Calculate features using current history
        # We need the last 63 prices to calculate SMA63
        # self.prices now has length up to 64 (if we popped)
        
        price_series = pd.Series(self.prices)
        
        # We want the feature for the *current* point (the last one)
        curr_sma_21 = price_series.iloc[-21:].mean()
        curr_sma_63 = price_series.iloc[-63:].mean()
        
        # Mom 5: Price(t) / Price(t-5) - 1
        if len(self.prices) >= 6:
            curr_mom_5 = self.prices[-1] / self.prices[-6] - 1
        else:
            curr_mom_5 = 0.0
            
        # Signal
        if curr_sma_63 == 0:
            sma_ratio = 0.0
        else:
            sma_ratio = curr_sma_21 / curr_sma_63 - 1
            
        # Apply optimized parameters
        raw_signal = self.w_sma * sma_ratio + self.w_mom * curr_mom_5
        weight = np.clip(self.bias + self.scale * raw_signal, 0.0, 2.0)
        
        return weight

# Initialize Global Model
model = MomentumStrategy()

# --- API Predict Function ---
def predict(test: pl.DataFrame) -> float:
    # The API passes a Polars DataFrame.
    # It contains 'lagged_forward_returns' (or equivalent) if provided by the environment.
    # If not, we assume 0.0 (which is bad, but safe).
    
    df = test.to_pandas()
    
    # Check for return column
    col_name = 'lagged_forward_returns'
    if col_name not in df.columns:
        # Fallback: check for 'return'
        if 'return' in df.columns:
            col_name = 'return'
        else:
            col_name = None
            
    # We expect a single row or batch. The example suggests returning a single float.
    # If batch, we might need to return a list? But the example returns float.
    # This implies the gateway calls predict() one row at a time or expects one value per call.
    # Let's assume single row for now based on the working example.
    
    lagged_ret = 0.0
    if not df.empty:
        row = df.iloc[0]
        if col_name:
            val = row.get(col_name, 0.0)
            if not pd.isna(val):
                lagged_ret = val
            
    w = model.predict_one(lagged_ret)
    return float(w)

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading training data...")
    train_df = load_data(TRAIN_PATH)
    
    if train_df is None:
        print("Train data not found. Exiting.")
        sys.exit(0)
    
    print("Fitting model...")
    model.fit(train_df)
    
    # Setup Inference Server
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

    if IS_KAGGLE:
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Starting inference server...")
            inference_server.serve()
        else:
            print("Running local gateway on Kaggle (Interactive)...")
            inference_server.run_local_gateway(
                (str(INPUT_DIR),)
            )
    else:
        print("Running locally...")
        try:
            inference_server.run_local_gateway(
                (str(INPUT_DIR),)
            )
        except Exception as e:
            print(f"Local gateway error: {e}")
            # Fallback: Manual generation
            if TEST_PATH.exists():
                print("Fallback: Generating submission.parquet manually...")
                test_df = load_data(TEST_PATH)
                if test_df is not None:
                    test_pl = pl.from_pandas(test_df)
                    
                    # Reset model state to end of train
                    model.fit(train_df)
                    
                    # Manual loop for local file generation
                    weights = []
                    ids = []
                    for i in range(len(test_pl)):
                        row_pl = test_pl[i]
                        w = predict(row_pl)
                        weights.append(w)
                        # Get ID
                        row_pd = row_pl.to_pandas()
                        id_val = row_pd['row_id'].iloc[0] if 'row_id' in row_pd.columns else row_pd['date_id'].iloc[0]
                        ids.append(id_val)
                        
                    submission_pl = pl.DataFrame({'date_id': ids, 'prediction': weights})
                    output_path = Path('submission.parquet')
                    submission_pl.write_parquet(output_path)
                    print(f"Saved {output_path}")
        
        if Path('submission.parquet').exists():
            print("Success: submission.parquet generated.")
        else:
            print("Warning: submission.parquet not found.")

