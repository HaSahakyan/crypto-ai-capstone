from datetime import datetime
import pandas as pd
import requests
import time
import os


def timestamp_to_utc(data):
  return datetime.fromtimestamp(data)
  
def get_url(api_key,crypto,frequency,timestamp,records_each_call):
  api_key = '&api_key='+str(api_key)
  limit = '&limit='+str(records_each_call)
  cryptoHeader = 'fsym='+str(crypto)
  ouputCurrency = '&tsym=USD'

  if(timestamp == ""):
    url = 'https://min-api.cryptocompare.com/data/v2/histo' + frequency + '?' + cryptoHeader
  else:  
    url = 'https://min-api.cryptocompare.com/data/v2/histo' + frequency + '?' + cryptoHeader + '&toTs=' + str(timestamp)

  return url + ouputCurrency + api_key + limit  


def getCryptoData(apikey, crypto, frequency, start_date, end_date, records_each_call):
    data = pd.DataFrame()
    toTs = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp())
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp())

    while toTs > start_ts:
        url = get_url(apikey, crypto, frequency, toTs, records_each_call)
        time.sleep(1)
        response = requests.get(url).json()

        if response.get("Response") == "Error":
            print(f"Error from API: {response.get('Message')}")
            break

        partial_data = response.get('Data', {}).get('Data', [])
        if not partial_data:
            print(f"No data returned from API for timestamp {toTs}")
            break

        partial = pd.DataFrame(partial_data)

        if partial.empty:
            break

        earliest_time = partial['time'].min()
        toTs = earliest_time - 1
        data = pd.concat([data, partial])

        if earliest_time < start_ts:
            break

    data = data.sort_values(by='time', ignore_index=True)
    data['timeUTC'] = data['time'].apply(timestamp_to_utc)
    return data[data['time'] >= start_ts].reset_index(drop=True)

if __name__ == '__main__':
    # Replace api_key with your API key (https://developers.coindesk.com/)
    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Currencies_Alpha_Vantage' )
    api_key = "NHP2XOX7TEVKI115"
    cryptos = ["BTC", "ETH", "SOL"]
    frequency = "hour" # "day", "minute" also available (minute is available for last 7 days only)
    records_each_call = 2000

    start_date = "2024-01-01 00:00:00"
    end_date = "2025-01-12 00:00:00"

    for crypto in cryptos:
        print(f"Fetching data for {crypto}...")
        cryptoData = getCryptoData(api_key, crypto, frequency, start_date, end_date, records_each_call)
        cryptoData.to_csv(f'{result_path}/{crypto}_data_{frequency}.csv', index=False)
