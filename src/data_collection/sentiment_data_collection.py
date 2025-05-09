import requests
import json
import csv
import os

def get_crypto_sentiment(alpha_vantage_api_key, ticker, time_from, time_to=None, limit=1000):
    """
    Fetches historical news sentiment data for a given cryptocurrency from Alpha Vantage.
    
    Parameters:
    - alpha_vantage_api_key (str): Your Alpha Vantage API key.
    - ticker (str): The cryptocurrency ticker (e.g., 'SOL' for Solana, 'BTC' for Bitcoin).
    - time_from (str): Start time in YYYYMMDDTHHMM format (e.g., '20230101T0000').
    - time_to (str, optional): End time in YYYYMMDDTHHMM format.
    - limit (int, optional): Maximum number of articles to return (max 1000).
    
    Returns:
    - dict: JSON response from the API containing sentiment data.
    - None: If the API request fails.
    """
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': f'CRYPTO:{ticker}',
        'time_from': time_from,
        'apikey': alpha_vantage_api_key
    }
    if time_to:
        params['time_to'] = time_to
    params['limit'] = limit
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None
    
def separate_crypto_sentiment_data(tickers, result_path):
    """
    Separates the sentiment data for each cryptocurrency from the API response.
    """
    for ticker in  tickers:
        sentiment_data = get_crypto_sentiment(api_key, ticker, time_from, time_to)
        if sentiment_data and 'feed' in sentiment_data and sentiment_data['feed']:
            first_article = sentiment_data['feed'][0]
            ticker_sentiments = first_article.get('ticker_sentiment', [])
            for ts in ticker_sentiments:
                if ts.get('ticker') == f'CRYPTO:{ticker}':
                    print(f"Sentiment score for {ticker}: {ts.get('ticker_sentiment_score', 'N/A')}")
                    break
            else:
                print(f"No sentiment data found for {ticker} in the first article.")
        else:
            print("Failed to fetch data or no articles found.")

        data_file_name = f'{result_path}sentiment_data_{ticker.lower()}.json'
        with open(data_file_name, 'w') as f:
            json.dump(sentiment_data, f, indent=4)



        # Filtering and converting the JSON data to CSV
        with open(f'{result_path}/sentiment_data_{ticker.lower()}.json', 'r') as f:
            data = json.load(f)

        feed = data.get('feed', [])
        rows = []
        sol_ticker = f'CRYPTO:{ticker}'

        # Iterateing through each article in the feed
        for article in feed:
            for ts in article.get('ticker_sentiment', []):
                if ts.get('ticker') == sol_ticker:
                    row = {
                        'time_published': article.get('time_published', ''),
                        'source': article.get('source', ''),
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'overall_sentiment_score': article.get('overall_sentiment_score', ''),
                        'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                        f'{ticker.lower()}_relevance_score': ts.get('relevance_score', ''),
                        f'{ticker.lower()}_sentiment_score': ts.get('ticker_sentiment_score', ''),
                        f'{ticker.lower()}_sentiment_label': ts.get('ticker_sentiment_label', '')
                    }
                    rows.append(row)
                    break

        fieldnames = [
            'time_published', 'source', 'title', 'url',
            'overall_sentiment_score', 'overall_sentiment_label',
            f'{ticker.lower()}_relevance_score', f'{ticker.lower()}_sentiment_score', f'{ticker.lower()}_sentiment_label'
        ]

        # Writing the data to a CSV file
        with open(f'{result_path}/{ticker.lower()}_sentiment.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)



if __name__ == '__main__':
    # Replace api_key with your API key (https://developers.coindesk.com/)
    api_key = 'NHP2XOX7TEVKI115'
    time_from = '20240901T0000'
    time_to = '20250112T2359'
    tickers = ['BTC', 'ETH', 'SOL']
    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets', 'Sentiment_Scores' )
    separate_crypto_sentiment_data(tickers, result_path)