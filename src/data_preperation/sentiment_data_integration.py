# import os
# import sys
# import pandas as pd
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_time_grouping')))
# from data_time_grouping import CandleAggregator


# def sentiment_data_combined_with_price(crypto_path, sentiment_path, result_folder):
#     # If result already exists, load and return it
#     if os.path.exists(result_folder):
#         return pd.read_csv(result_folder)
    
#     group_length = 1 #window_config['group_length']
#     join_key = 'datetime' #window_config['join_key']
#     time_format = lambda x: x.dt.floor('T') #window_config['time_format']

#     # Load and prepare price data
#     price_data = pd.read_csv(crypto_path)
#     price_data['datetime'] = pd.to_datetime(price_data['timestamp'], unit='s')
#     price_data = price_data.sort_values('datetime')

#     aggregator = CandleAggregator(group_length=group_length)
#     aggregated_price = aggregator.aggregate(price_data)
#     aggregated_price[join_key] = time_format(aggregated_price['datetime'])

#     # Load and prepare sentiment data
#     sentiment_data = pd.read_csv(sentiment_path)
#     sentiment_data['datetime'] = pd.to_datetime(sentiment_data['time_published'])
#     sentiment_data = sentiment_data.sort_values('datetime')  # Sort by timestamp
#     sentiment_data[join_key] = time_format(sentiment_data['datetime'])
#     aggregated_sentiment = sentiment_data.groupby(join_key).agg({
#         'overall_sentiment_score': 'mean',
#         'sol_sentiment_score': 'mean'
#     }).reset_index()

#     # Merge datasets
#     merged_data = pd.merge(aggregated_price, aggregated_sentiment, on=join_key, how='left')
#     merged_data = merged_data.sort_values('datetime')  # Sort merged data by datetime
#     # Forward fill sentiment scores, then fill remaining NaNs with 0
#     merged_data['overall_sentiment_score'] = merged_data['overall_sentiment_score'].ffill().fillna(0)
#     merged_data['sol_sentiment_score'] = merged_data['sol_sentiment_score'].ffill().fillna(0)

#     aggregation_window_list = ['minutely', 'hourly', 'daily']

#     for aggregation_window in aggregation_window_list:
#         # Add rolling sentiment features
#         if aggregation_window == 'minutely':
#             windows = list(range(1, 15))
#             windows_min=windows
#             unit = 'min'
#         elif aggregation_window == 'hourly':
#             windows = list(range(1, 5))
#             windows_min = list(map(lambda x: x * 60, windows))
#             unit = 'hour'
#         elif aggregation_window == 'daily':
#             windows = list(range(1, 7))
#             windows_min = list(map(lambda x: x * 1440, windows))
#             unit = 'day'
#         for window in range(len(windows)):
#             merged_data[f'overall_sentiment_{window}{unit}_avg'] = merged_data['overall_sentiment_score'].rolling(window=windows_min[window]).mean().fillna(0)
#             merged_data[f'sol_sentiment_{window}{unit}_avg'] = merged_data['sol_sentiment_score'].rolling(window=windows_min[window]).mean().fillna(0)
#             merged_data[f'overall_sentiment_{window}{unit}_prev'] = merged_data['overall_sentiment_score'].shift(windows_min[window]).fillna(0)
#             merged_data[f'sol_sentiment_{window}{unit}_prev'] = merged_data['sol_sentiment_score'].shift(windows_min[window]).fillna(0)

#     # Reset index for trend detection
#     merged_data = merged_data.reset_index(drop=True)
#     # merged_data.fillna(0)

#     # Save result
#     merged_data.to_csv(result_folder, index=False)

#     return merged_data



# if __name__ == '__main__':
#     result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', f'minutely_aggrigated_sol_sentiment_price.csv')

#     # # Create results folder if it doesn't exist
#     # os.makedirs(result_folder, exist_ok=True)   
#     crypto_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'sol_binance.csv')
#     sentiment_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Sentiment_Scores', 'sol_sentiment.csv')
#     combined = sentiment_data_combined_with_price(crypto_path, sentiment_path, result_folder)
#     print(combined).head()


import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_time_grouping')))
from data_time_grouping import CandleAggregator


def sentiment_data_combined_with_price(crypto, price_data, sentiment_path, result_folder):
    # If result already exists, load and return it
    if os.path.exists(result_folder):
        return pd.read_csv(result_folder)
    
    group_length = 1 #window_config['group_length']
    join_key = 'datetime' #window_config['join_key']
    time_format = lambda x: x.dt.floor('T') #window_config['time_format']

    # Load and prepare price data
    price_data['datetime'] = pd.to_datetime(price_data['timestamp'], unit='s')
    price_data = price_data.sort_values('datetime')

    aggregator = CandleAggregator(group_length=group_length)
    aggregated_price = aggregator.aggregate(price_data)
    aggregated_price[join_key] = time_format(aggregated_price['datetime'])

    # Load and prepare sentiment data
    sentiment_data = pd.read_csv(sentiment_path)
    sentiment_data['datetime'] = pd.to_datetime(sentiment_data['time_published'])
    sentiment_data = sentiment_data.sort_values('datetime')  # Sort by timestamp
    sentiment_data[join_key] = time_format(sentiment_data['datetime'])
    aggregated_sentiment = sentiment_data.groupby(join_key).agg({
        'overall_sentiment_score': 'mean',
        f'{crypto.lower()}_sentiment_score': 'mean'
    }).reset_index()

    # Merge datasets
    merged_data = pd.merge(aggregated_price, aggregated_sentiment, on=join_key, how='left')
    merged_data = merged_data.sort_values('datetime')  # Sort merged data by datetime
    # Forward fill sentiment scores, then fill remaining NaNs with 0
    merged_data['overall_sentiment_score'] = merged_data['overall_sentiment_score'].ffill().fillna(0)
    merged_data[f'{crypto.lower()}_sentiment_score'] = merged_data[f'{crypto.lower()}_sentiment_score'].ffill().fillna(0)

    aggregation_window_list = ['minutely', 'hourly', 'daily']

    for aggregation_window in aggregation_window_list:
        # Add rolling sentiment features
        if aggregation_window == 'minutely':
            windows = list(range(1, 15))
            windows_min=windows
            unit = 'min'
        elif aggregation_window == 'hourly':
            windows = list(range(1, 5))
            windows_min = list(map(lambda x: x * 60, windows))
            unit = 'hour'
        elif aggregation_window == 'daily':
            windows = list(range(1, 7))
            windows_min = list(map(lambda x: x * 1440, windows))
            unit = 'day'
        for window in range(len(windows)):
            merged_data[f'overall_sentiment_{window}{unit}_avg'] = merged_data['overall_sentiment_score'].rolling(window=windows_min[window]).mean().fillna(0)
            merged_data[f'{crypto.lower()}_sentiment_{window}{unit}_avg'] = merged_data[f'{crypto.lower()}_sentiment_score'].rolling(window=windows_min[window]).mean().fillna(0)
            merged_data[f'overall_sentiment_{window}{unit}_prev'] = merged_data['overall_sentiment_score'].shift(windows_min[window]).fillna(0)
            merged_data[f'{crypto.lower()}_sentiment_{window}{unit}_prev'] = merged_data[f'{crypto.lower()}_sentiment_score'].shift(windows_min[window]).fillna(0)

    # Reset index for trend detection
    merged_data = merged_data.reset_index(drop=True)
    # merged_data.fillna(0)

    # Save result
    merged_data.to_csv(result_folder, index=False)

    return merged_data



if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets')
    result_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets')
    sentiment_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Sentiment_Scores')
    # Create results folder if it doesn't exist
    os.makedirs(result_folder, exist_ok=True)

    for crypto in ['SOL', 'BTC', 'ETH']:
        
        if crypto == 'SOL':
            save_result =  f'{result_folder}/minutely_aggrigated_{crypto.lower()}_sentiment_price.csv'
            if os.path.exists(save_result):
                print(f"File {save_result} already exists. Skipping...")
                continue
            crypto_path = f'{dataset_path}/sol_binance.csv'
            price_data = pd.read_csv(crypto_path)
        else:
            save_result =  f'{result_folder}/hourly_aggrigated_{crypto.lower()}_sentiment_price.csv'
            if os.path.exists(save_result):
                print(f"File {save_result} already exists. Skipping...")
                continue
            crypto_path = f'{dataset_path}/Currencies_Alpha_Vantage/{crypto}_data_hour.csv'
            price_data = pd.read_csv(crypto_path)

            price_data = price_data.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            })[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


        crypto_sentiment_path = f'{sentiment_path}/{crypto.lower()}_sentiment.csv'
        combined = sentiment_data_combined_with_price(crypto, price_data, crypto_sentiment_path, save_result)
        print(f"Combined data for {crypto} saved to {save_result}")

