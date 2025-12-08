import pandas as pd
import numpy as np

class StockFeatureEngineer:
    """
    A class to generate technical indicators and features for stock market data.
    Expects a DataFrame with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """

    def __init__(self, df):
        # Create a copy to avoid SettingWithCopy warnings
        self.df = df.copy()
        
        # Ensure Date is datetime and set as index if not already
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            
        # Ensure standard column names (case-insensitive handling)
        self.df.columns = [x.lower() for x in self.df.columns]

    def add_moving_averages(self, windows=[20, 50, 200]):
        """Adds Simple and Exponential Moving Averages."""
        for w in windows:
            self.df[f'sma_{w}'] = self.df['close'].rolling(window=w).mean()
            self.df[f'ema_{w}'] = self.df['close'].ewm(span=w, adjust=False).mean()
        return self

    def add_rsi(self, window=14):
        """Adds Relative Strength Index (RSI)."""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN for the first 'window' rows
        self.df['rsi'] = self.df['rsi'].fillna(50)
        return self

    def add_macd(self, fast=12, slow=26, signal=9):
        """Adds MACD, Signal Line, and Histogram."""
        exp1 = self.df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = exp1 - exp2
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        return self

    def add_bollinger_bands(self, window=20, num_std=2):
        """Adds Upper, Middle, and Lower Bollinger Bands."""
        sma = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()
        
        self.df['bb_upper'] = sma + (std * num_std)
        self.df['bb_lower'] = sma - (std * num_std)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / sma
        return self

    def add_atr(self, window=14):
        """Adds Average True Range (ATR) for volatility."""
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        self.df['atr'] = true_range.rolling(window=window).mean()
        return self

    def add_volume_indicators(self):
        """Adds On-Balance Volume (OBV) and VWAP."""
        # OBV
        self.df['obv'] = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        
        # VWAP (Cumulative for simplicity, usually resets daily/session in strict trading)
        cum_vol = self.df['volume'].cumsum()
        cum_vol_price = (self.df['close'] * self.df['volume']).cumsum()
        self.df['vwap'] = cum_vol_price / cum_vol
        return self

    def add_returns(self):
        """Adds Daily Log Returns (preferred for ML)."""
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        return self

    def get_features(self):
        """Executes all methods and returns the cleaned DataFrame."""
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_volume_indicators()
        self.add_returns()
        
        # Drop rows with NaN values created by rolling windows (e.g., first 200 rows)
        return self.df.dropna()

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Generate Dummy Data
    dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
    data = {
        "Date": dates,
        "Open": np.random.uniform(100, 200, 500),
        "High": np.random.uniform(100, 200, 500),
        "Low": np.random.uniform(100, 200, 500),
        "Close": np.random.uniform(100, 200, 500),
        "Volume": np.random.randint(1000, 100000, 500)
    }
    
    # Fix High/Low logic for realism
    df_raw = pd.DataFrame(data)
    df_raw['High'] = df_raw[['Open', 'Close']].max(axis=1) + 2
    df_raw['Low'] = df_raw[['Open', 'Close']].min(axis=1) - 2

    # 2. Run Feature Engineering
    engineer = StockFeatureEngineer(df_raw)
    df_features = engineer.get_features()

    print("Features Generated Successfully:")
    print(df_features[['close', 'rsi', 'macd', 'bb_upper', 'atr']].tail())
    print(f"\nTotal Shape: {df_features.shape}")
