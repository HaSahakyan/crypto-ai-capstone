import time
import pandas as pd
import matplotlib.pyplot as plt

class TargetCreatorSimple:
    def __init__(self, df: pd.DataFrame, price_type='close', date_col='datetime', pct_threshold=0.0035, min_length=3, smoothing_factor=0.001, grouping_interval = ''):
        """
        Simplified TargetCreator for detecting upward trends.
        
        Args:
            df (pd.DataFrame): DataFrame containing at least 'Date' and the price column.
            price_type (str): Column to use for price ('Close' is expected).
            pct_threshold (float): Minimum percentage increase (as a decimal) from trend start to end.
            smoothing_factor (float): Allowed fractional drop (e.g., 0.0001 for 0.01%) compared to the previous price
                                      to still consider the trend ongoing.
            min_length (int): Minimum number of points in a valid trend (inclusive).                                      
        """
        self.df = df.copy()
        self.price_type = price_type
        self.date_col = date_col
        self.pct_threshold = pct_threshold
        self.smoothing_factor = smoothing_factor
        self.min_length = min_length
        if grouping_interval != '':
            self.grouping_interval = f'[time interval : {grouping_interval}]'
        else:
            self.grouping_interval = ''
        self.prepare_data()

    def get_df(self, reset_index=False):
        if reset_index:
            return self.df.reset_index()
        return self.df
    
    def prepare_data(self):
        """Converts Date to datetime, sorts by date, and sets the Date as index."""
        # self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.sort_values('datetime', inplace=True)
        self.df.set_index('datetime', inplace=True)
        # Ensure the price column is available in a consistent format
        self.df[self.price_type] = self.df[self.price_type]


    def detect_trends(self, flactuation_length = 5, flactuation_threshold = 0.01):
        close = self.df[self.price_type].values
        n = len(close)
        up_trends = []
        down_trends = []

        def _process_trend(trends, start, end, direction):
            length = end - start + 1
            if length < self.min_length:
                return
            pct_change = (close[end] - close[start]) / close[start]
            if (direction == 'up' and pct_change < self.pct_threshold) or \
               (direction == 'down' and pct_change > -self.pct_threshold):
                return
            if trends:
                prev_start, prev_end, _ = trends[-1]
                gap = start - prev_end
                fluc = abs((close[start] - close[prev_end]) / close[prev_end])
                if gap <= flactuation_length and fluc <= flactuation_threshold:
                    pct_change = (close[end] - close[prev_start]) / close[prev_start]
                    trends[-1] = (prev_start, end, pct_change)
                    return
            trends.append((start, end, pct_change))

        up_start = down_start = 0
        for i in range(1, n):
            if close[i] < close[i - 1] * (1 - self.smoothing_factor):
                _process_trend(up_trends, up_start, i - 1, 'up')
                up_start = i
            else:
                _process_trend(down_trends, down_start, i - 1, 'down')
                down_start = i
        _process_trend(up_trends, up_start, n - 1, 'up')
        _process_trend(down_trends, down_start, n - 1, 'down')
        self.up_trends = up_trends
        self.down_trends = down_trends
        return up_trends, down_trends


    def add_trend_columns(self, direction=['Upward'], remap_for_classification=True):
        """
        Adds trend columns to the DataFrame to flag the start of each detected trend.
        Parameters:
            direction (list): List of directions to process. Should contain 'Upward', 'Downward', or both.
        """
        # Detect trends if not already done
        if not hasattr(self, 'up_trends') or not hasattr(self, 'down_trends'):
            self.detect_trends()

        for dir in direction:
            self.df[dir] = 0

            trends = self.up_trends if dir == 'Upward' else self.down_trends
            value = 1 if dir == 'Upward' else -1

            for start, _, _ in trends:
                self.df.iloc[start, self.df.columns.get_loc(dir)] = value
        if len(direction) != 1:
            if 'Upward' in self.df.columns and 'Downward' in self.df.columns:
                self.df['trend_target'] = self.df['Upward'] + self.df['Downward']
            if remap_for_classification:
                self.df['combined_target'] = self.df['trend_target'].map({-1: 0, 0: 1, 1: 2})

        # Add trend labels for all days within trends
        self.df['trend_label'] = 'none'
        for start, end, _ in self.up_trends:
            self.df.iloc[start:end+1, self.df.columns.get_loc('trend_label')] = 'up'
        for start, end, _ in self.down_trends:
            self.df.iloc[start:end+1, self.df.columns.get_loc('trend_label')] = 'down'

    def visualize_trends(self):
        """
        Visualizes the closing price along with detected upward and downward trends.
        Up trends are orange with green start markers; down trends are pink with red start markers.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.df.index, self.df[self.price_type], label=self.price_type, color='blue')

        # Plot up trends
        for i, (start, end, pct) in enumerate(getattr(self, 'up_trends', [])):
            ax.plot(self.df.index[start:end+1], self.df[self.price_type].values[start:end+1],
                    linewidth=3, color='orange', label='Up Trend' if i == 0 else "")
            ax.scatter(self.df.index[start], self.df[self.price_type].values[start],
                      color='green', s=100, marker='^', label='Up Start' if i == 0 else "")

        # Plot down trends
        for i, (start, end, pct) in enumerate(getattr(self, 'down_trends', [])):
            ax.plot(self.df.index[start:end+1], self.df[self.price_type].values[start:end+1],
                    linewidth=3, color='pink', label='Down Trend' if i == 0 else "")
            ax.scatter(self.df.index[start], self.df[self.price_type].values[start],
                      color='red', s=100, marker='v', label='Down Start' if i == 0 else "")

        # Legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Trend Segments in {self.price_type.capitalize()} Price {self.grouping_interval}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    file_path = 'Datasets/sol_binance.csv'
    sol_binance = pd.read_csv(file_path)
    sol_binance['datetime'] = pd.to_datetime(sol_binance['timestamp'], unit='s')
    sol_binance = sol_binance.sort_values('datetime')
    sol_binance.set_index('datetime', inplace=True)

    # target_creator = TargetCreatorSimple(sol_binance, price_type='close', min_length=3, pct_threshold=0.005, smoothing_factor=0.0000)
    
    # # Detect trends and print them.
    # target_creator.add_trend_columns()
    # # print(target_creator.up_trends)
    # # print(target_creator.down_trends)

    # Create an instance of class
    creator = TargetCreatorSimple(
        df=sol_binance,
        price_type='close',        
        pct_threshold=0.01,       
        min_length=3,             
        smoothing_factor=0.001     
    )

    creator.detect_trends(
        flactuation_length=5,       
        flactuation_threshold=0.01 
    )

    creator.add_trend_columns(direction=['Upward', 'Downward'])
