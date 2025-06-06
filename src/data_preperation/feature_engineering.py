import pandas as pd

class Data_Featuring:
    def __init__(self, data):
        """
        Initialize the Data_Featuring class.

        Parameters:
        - data (pd.DataFrame): The input time-series DataFrame with at least a 'close' column.
        """
        self.data = data

    def get_df(self, reset_index=False):
        """
        Returns the internal DataFrame with any added trend columns.

        Parameters:
            reset_index (bool): If True, resets the index so 'datetime' becomes a column again.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        if reset_index:
            return self.data.reset_index()
        return self.data


    def lag_features(self, lag_list=[1, 2]):
        """
        Create lag features for the 'close' column.

        Adds new columns:
        - close_lag_X: the close price X minutes/timesteps before.

        Affects:
        - Adds new columns to self.data representing past close prices.
        """
        for lag in lag_list:
            self.data[f'close_lag_{lag}'] = self.data['close'].shift(lag)

    def percentage_change_price(self, periods_list=[1, 3]):
        """
        Calculate percentage change in 'close' price over specified periods.

        Adds new columns:
        - return_X: % change in close over X periods.

        Affects:
        - Adds return features that capture price momentum.
        """
        for period in periods_list:
            self.data[f'return_{period}'] = self.data['close'].pct_change(periods=period)

    def moving_average(self, moving_window=[5, 10]):
        """
        Compute simple moving averages over specified window sizes.

        Adds new columns:
        - ma_X: moving average of close over X periods.

        Affects:
        - Adds trend-following features to self.data.
        """
        for window in moving_window:
            self.data[f'ma_{window}'] = self.data['close'].rolling(window=window).mean()

    def rolling_volatility(self, windows=[5, 10]):
        """
        Compute rolling standard deviation of 1-period returns to estimate volatility.

        Adds new columns:
        - volatility_X: standard deviation of 1-period returns over X periods.

        Affects:
        - Captures risk/volatility in short-term windows.
        """
        if 'return_1' not in self.data.columns:
            self.percentage_change_price(periods_list=[1])
        for window in windows:
            self.data[f'volatility_{window}'] = self.data['return_1'].rolling(window=window).std()

    def rolling_momentum(self, windows=[5]):
        """
        Compute rolling max/min close prices to measure local momentum.

        Adds new columns:
        - roll_max_X: max close price over X periods.
        - roll_min_X: min close price over X periods.

        Affects:
        - Highlights directional strength or overbought/oversold conditions.
        """
        for window in windows:
            self.data[f'roll_max_{window}'] = self.data['close'].rolling(window=window).max()
            self.data[f'roll_min_{window}'] = self.data['close'].rolling(window=window).min()

    def compute_rsi(self, window=14):
        """
        Calculate the Relative Strength Index (RSI) for the 'close' column.

        Adds:
        - rsi_X: momentum oscillator value between 0–100.

        Affects:
        - Identifies overbought (>70) or oversold (<30) conditions.
        """
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.data[f'rsi_{window}'] = rsi

    def compute_macd(self, fast=12, slow=26, signal=9):
        """
        Calculate MACD and Signal Line.

        Adds:
        - macd: difference between fast and slow EMAs.
        - macd_signal: signal line, EMA of MACD.

        Affects:
        - Detects trend reversals and momentum shifts.
        """
        ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        self.data['macd'] = macd
        self.data['macd_signal'] = macd_signal

    def bollinger_bands(self, window=20):
        """
        Compute Bollinger Bands using moving average and standard deviation.

        Adds:
        - bb_middle: mid-band (MA)
        - bb_upper: upper band = MA + 2*STD
        - bb_lower: lower band = MA - 2*STD

        Affects:
        - Helps detect price breakouts and volatility compression.
        """
        ma = self.data['close'].rolling(window=window).mean()
        std = self.data['close'].rolling(window=window).std()
        self.data['bb_middle'] = ma
        self.data['bb_upper'] = ma + 2 * std
        self.data['bb_lower'] = ma - 2 * std

    def exponential_moving_averages(self, spans=[10, 30]):
        """
        Calculate Exponential Moving Averages (EMA) of the close price.

        Adds:
        - ema_X: EMA over X periods.

        Affects:
        - Gives more weight to recent prices; useful for trend detection.
        """
        for span in spans:
            self.data[f'ema_{span}'] = self.data['close'].ewm(span=span, adjust=False).mean()

    def price_rate_of_change(self, windows=[10]):
        """
        Compute the rate of change in close price over the given windows.

        Adds:
        - roc_X: rate of change over X periods.

        Affects:
        - Measures speed and strength of price movement.
        """
        for window in windows:
            self.data[f'roc_{window}'] = self.data['close'].pct_change(periods=window)

    def crossover_signals(self):
        """
        Generate crossover signal between ma_5 and ma_10.

        Adds:
        - golden_cross: 1 if ma_5 > ma_10, else 0

        Affects:
        - Signals short-term bullish trend.
        """
        self.data['golden_cross'] = (self.data['ma_5'] > self.data['ma_10']).astype(int)

    def slope_of_ma(self, ma_column='ma_5'):
        """
        Compute slope (first derivative) of a moving average.

        Adds:
        - slope_ma_X: difference between current and previous MA value.

        Affects:
        - Helps measure acceleration in trend.
        """
        self.data[f'slope_{ma_column}'] = self.data[ma_column].diff()

    def volume_spike(self, window=10, threshold=1.5):
        """
        Detect sudden spikes in trading volume.

        Adds:
        - volume_ma: rolling average of volume
        - volume_spike: 1 if volume > threshold * volume_ma

        Affects:
        - Useful for breakout detection and event filtering.
        """
        self.data['volume_ma'] = self.data['volume'].rolling(window=window).mean()
        self.data['volume_spike'] = (self.data['volume'] > self.data['volume_ma'] * threshold).astype(int)

    def true_range_and_atr(self, window=14):
        """
        Compute True Range and Average True Range (ATR).

        Adds:
        - tr: true range
        - atr_X: average true range over X periods

        Affects:
        - Better volatility estimation than standard deviation alone.
        """
        high = self.data['high']
        low = self.data['low']
        close_prev = self.data['close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)

        self.data['tr'] = tr
        self.data[f'atr_{window}'] = self.data['tr'].rolling(window=window).mean()

    def spread_features(self):
        """
        Compute price range and difference between close and open.

        Adds:
        - hl_spread: high - low
        - co_spread: close - open

        Affects:
        - Captures intra-candle volatility and direction.
        """
        self.data['hl_spread'] = self.data['high'] - self.data['low']
        self.data['co_spread'] = self.data['close'] - self.data['open']

    def candle_structure_features(self):
        """
        Extract candle body and wick sizes.

        Adds:
        - body_size: absolute difference between open and close
        - upper_wick: high - max(open, close)
        - lower_wick: min(open, close) - low

        Affects:
        - Captures price action patterns and market sentiment.
        """
        self.data['body_size'] = abs(self.data['close'] - self.data['open'])
        self.data['upper_wick'] = self.data['high'] - self.data[['open', 'close']].max(axis=1)
        self.data['lower_wick'] = self.data[['open', 'close']].min(axis=1) - self.data['low']

    def time_features(self):
        """
        Extract time-related features from datetime.

        Adds:
        - hour: hour of the day
        - minute: minute of the hour
        - dayofweek: weekday (0 = Monday)

        Affects:
        - Helps model detect time-based trading patterns.
        """
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['minute'] = self.data['datetime'].dt.minute
        self.data['dayofweek'] = self.data['datetime'].dt.dayofweek
        self.data['date'] = self.data['datetime'].dt.date

    def all_features(self):
        """
        Run all feature creation methods with default parameters.

        Affects:
        - Extends self.data with all engineered features for modeling.
        """
        self.lag_features()
        self.percentage_change_price()
        self.moving_average()
        self.rolling_volatility()
        self.rolling_momentum()
        self.compute_rsi()
        self.compute_macd()
        self.bollinger_bands()
        self.exponential_moving_averages()
        self.price_rate_of_change()
        self.crossover_signals()
        self.slope_of_ma()
        self.volume_spike()
        self.true_range_and_atr()
        self.spread_features()
        self.candle_structure_features()
        self.time_features()
