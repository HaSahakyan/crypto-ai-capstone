import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_initiation')))
from dataset_initiation import prepare_dataset

import pandas as pd

def prepare_crypto_datasets(crypto_symbols):
    
    datasets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets')
    result_folder = os.path.join(os.path.dirname(__file__), f'{datasets_path}', 'Aggregated_Datasets')
    os.makedirs(result_folder, exist_ok=True)
    
    google_trends_path = os.path.join(os.path.dirname(__file__), f'{datasets_path}', 'Google_Search_Trends', 'google_search_trends.csv')
    google_trends = pd.read_csv(google_trends_path, sep=',', header=1)
    google_trends['date'] = pd.to_datetime(google_trends['Day'])
    google_trends.drop(columns=['Day'], inplace=True)

    for symbol in crypto_symbols:
        if symbol == 'SOL':
            crtpo_file = f'{datasets_path}/sol_binance.csv'
            period = 'minutely'
            crypto_data = pd.read_csv(crtpo_file, sep=',')
        else:
            crtpo_file = f'{datasets_path}/Currencies_Alpha_Vantage/{symbol}_data_hour.csv'
            period = 'hourly'
            crypto_data = pd.read_csv(crtpo_file, sep=',')
            crypto_data = crypto_data.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            })[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Load binance data
        crypto_data['datetime'] = pd.to_datetime(crypto_data['timestamp'], unit='s')
        crypto_data = crypto_data.sort_values('datetime')

        # Load sentiment and price data
        price_sentiment_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Aggregated_Datasets', f'{period}_aggrigated_{symbol}_sentiment_price.csv')
        price_sentiment = pd.read_csv(price_sentiment_path, sep=',')
        price_sentiment['datetime'] = pd.to_datetime(price_sentiment['timestamp'], unit='s')
        price_sentiment = price_sentiment.sort_values('datetime')

        print(f"Datasets preparation started for {symbol}...")

        # Prepare datasets with different aggregations
        if not crtpo_file.endswith('binance.csv'):
            datasets = {
                f'prep_{symbol}_google': prepare_dataset(df=crypto_data, aggregate=True, group_length=24, join_df=[google_trends]),
                f'prep_{symbol}_google_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=24, join_df=[google_trends]),
                f'prep_{symbol}_hourly': prepare_dataset(df=crypto_data, visualize_trends=False),
                f'prep_{symbol}_hourly_sentiment': prepare_dataset(df=price_sentiment, visualize_trends=False),
                f'prep_{symbol}_daily': prepare_dataset(df=crypto_data, aggregate=True, group_length=24, visualize_trends=False),
                f'prep_{symbol}_daily_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=24, visualize_trends=False)
            }
        else:
            datasets = {
                f'prep_{symbol}_google': prepare_dataset(df=crypto_data, aggregate=True, group_length=1440, join_df=[google_trends]),
                f'prep_{symbol}_google_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=1440, join_df=[google_trends]),
                f'prep_{symbol}': prepare_dataset(df=crypto_data, visualize_trends=False),
                f'prep_{symbol}_agg5': prepare_dataset(df=crypto_data, aggregate=True, group_length=5, visualize_trends=False),
                f'prep_{symbol}_agg10': prepare_dataset(df=crypto_data, aggregate=True, group_length=10, visualize_trends=False),
                f'prep_{symbol}_agg30': prepare_dataset(df=crypto_data, aggregate=True, group_length=30, visualize_trends=False),
                f'prep_{symbol}_agg30_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=30, visualize_trends=False),
                f'prep_{symbol}_hourly': prepare_dataset(df=crypto_data, aggregate=True, group_length=60, visualize_trends=False),
                f'prep_{symbol}_hourly_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=60, visualize_trends=False),
                f'prep_{symbol}_daily': prepare_dataset(df=crypto_data, aggregate=True, group_length=1440, visualize_trends=False),
                f'prep_{symbol}_daily_sentiment': prepare_dataset(df=price_sentiment, aggregate=True, group_length=1440, visualize_trends=False)
            }

        print(f"Aggregated {symbol} dataset prepared")

        # Save datasets
        print(f"Datasets saving started for {symbol}...")
        for name, df in datasets.items():
            df.to_csv(os.path.join(result_folder, f'{name}.csv'), index=False)
        print(f"Datasets saved successfully for {symbol}...")

if __name__ == "__main__":
    crypto_symbols = ['BTC', 'ETH', 'SOL']
    prepare_crypto_datasets(crypto_symbols)