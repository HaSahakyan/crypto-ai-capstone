import pandas as pd

class CandleAggregator:
    def __init__(self, group_length: int):
        """
        :param group_length: Number of rows to aggregate into one candle.
        """
        self.group_length = group_length

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the DataFrame into larger time intervals.

        :param df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime']
        :return: Aggregated DataFrame with the same structure
        """
        df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)
        n_groups = len(df_sorted) // self.group_length

        ##adding
        column_list = list(df_sorted.columns)


        aggregated = []
        for i in range(n_groups):
            group = df_sorted.iloc[i * self.group_length : (i + 1) * self.group_length]
            row = {
                'timestamp': group['timestamp'].iloc[0],
                'open': group['open'].iloc[0],
                'high': group['high'].max(),
                'low': group['low'].min(),
                'close': group['close'].iloc[-1],
                'volume': group['volume'].sum(),
                'datetime': group['datetime'].iloc[0],
            }

            # Sentiment features handling
            for col in column_list:
                if col.endswith('_prev'):
                    row[col] = group[col].iloc[0]
                elif col.endswith('_avg'):
                    row[col] = group[col].mean()
                elif '_sentiment_score' in col:
                    row[col] = group[col].mean()

            aggregated.append(row)

        return pd.DataFrame(aggregated)


if __name__ == '__main__':
    file_path = 'Datasets/sol_binance.csv'
    sol_binance = pd.read_csv(file_path)
    sol_binance['datetime'] = pd.to_datetime(sol_binance['timestamp'], unit='s')
    sol_binance = sol_binance.sort_values('datetime')


    aggregator = CandleAggregator(group_length=3)
    aggregated_sol = aggregator.aggregate(sol_binance)
    print(aggregated_sol)