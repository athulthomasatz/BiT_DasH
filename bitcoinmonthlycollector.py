import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import random

def fetch_bitcoin_data_for_month(start_date):
    """
    Fetch Bitcoin data for a specific month using CryptoCompare API
    
    Args:
        start_date: datetime object for the first day of the month
    
    Returns:
        DataFrame with OHLCV data for that month
    """
    # Convert start_date to Unix timestamp
    timestamp = int(start_date.timestamp())
    
    # Calculate end date (roughly one month later)
    end_date = start_date + timedelta(days=30)
    end_timestamp = int(end_date.timestamp())
    
    # Format dates for display
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    print(f"Fetching data for period: {start_str} to {end_str}")
    
    # CryptoCompare API endpoint
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    
    # Parameters
    params = {
        "fsym": "BTC",               # From Symbol (Bitcoin)
        "tsym": "USD",               # To Symbol (US Dollar)
        "limit": 30,                 # Get ~1 month of data
        "toTs": end_timestamp,       # End timestamp
        "aggregate": 1               # 1 day aggregation
    }
    
    # Add random delay to avoid hitting rate limits
    delay = random.uniform(1.5, 3.0)
    print(f"Waiting {delay:.2f} seconds before API call...")
    time.sleep(delay)
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['Response'] == 'Success':
            candle_data = data['Data']['Data']
            
            # Process into dataframe
            records = []
            for candle in candle_data:
                # Skip empty data points
                if candle['time'] == 0:
                    continue
                    
                date = datetime.fromtimestamp(candle['time'])
                records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volumefrom']
                })
            
            df = pd.DataFrame(records)
            return df
        else:
            print(f"API error: {data.get('Message', 'Unknown error')}")
            return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def try_alternative_source_for_month(start_date):
    """Try an alternative source (Yahoo Finance) if CryptoCompare fails"""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance library not found. Installing...")
        os.system("pip install yfinance")
        import yfinance as yf
    
    # Calculate end date
    end_date = start_date + timedelta(days=30)
    
    # Format dates for yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Using Yahoo Finance for period: {start_str} to {end_str}")
    
    # Get Bitcoin data from Yahoo Finance
    btc_data = yf.download("BTC-USD", start=start_str, end=end_str, interval="1d")
    
    # Reset index to make Date a column
    btc_data = btc_data.reset_index()
    
    # Rename columns to match our format
    btc_data = btc_data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Convert date to string format
    btc_data['date'] = btc_data['date'].dt.strftime('%Y-%m-%d')
    
    # Select only the columns we need
    btc_data = btc_data[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    return btc_data

def save_monthly_data(df, month_num):
    """Save monthly data to CSV file"""
    if df.empty:
        print(f"No data to save for month {month_num}")
        return False
    
    filename = f"bitcoin_prices_month_{month_num}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    return True

def merge_csv_files():
    """Merge all monthly CSV files into one"""
    print("Merging monthly files...")
    all_data = []
    
    # Look for all monthly CSV files
    for i in range(1, 13):
        filename = f"bitcoin_prices_month_{i}.csv"
        if os.path.exists(filename):
            monthly_data = pd.read_csv(filename)
            all_data.append(monthly_data)
            print(f"Added {len(monthly_data)} records from {filename}")
    
    if not all_data:
        print("No monthly data files found to merge!")
        return
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['date'])
    
    # Sort by date
    merged_df = merged_df.sort_values('date')
    
    # Save the merged data
    merged_df.to_csv("bitcoin_prices_full_year.csv", index=False)
    merged_df.to_json("bitcoin_prices_full_year.json", orient="records", date_format="iso")
    
    print(f"Merged data saved with {len(merged_df)} total records")
    
    # Print sample
    print("\nSample of merged data:")
    print(merged_df.head())

def main():
    """Collect Bitcoin price data month by month for a year"""
    # Start date (one year ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate the 12 monthly start dates
    monthly_starts = []
    for i in range(12):
        month_start = start_date + timedelta(days=i*30)
        monthly_starts.append(month_start)
    
    # Process each month
    for i, month_start in enumerate(monthly_starts, 1):
        print(f"\nProcessing month {i} of 12...")
        
        # Try primary source
        df = fetch_bitcoin_data_for_month(month_start)
        
        # If primary source fails, try alternative
        if df.empty:
            print("Primary source failed, trying alternative...")
            df = try_alternative_source_for_month(month_start)
        
        # Save the monthly data
        save_monthly_data(df, i)
        
        # Add a longer delay between months to avoid rate limits
        if i < 12:
            delay = random.uniform(5, 10)
            print(f"Waiting {delay:.2f} seconds before next month...")
            time.sleep(delay)
    
    # Finally, merge all files
    merge_csv_files()

if __name__ == "__main__":
    main()
