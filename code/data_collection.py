# ---- Script to collect all data from the various sources ----
# -------------------------------------------------------------
# Loading in all necessary libraries
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from fredapi import Fred
from pytrends.request import TrendReq
import time
from typing import Optional
import requests
import io
from bs4 import BeautifulSoup 
from urllib.parse import urljoin 

# -------------------------------------------------------------
# Define Constants
START = '2024-01-02'
END = '2025-11-01'
# -------------------------------------------------------------

def collect_OHLVC(ticker):
    print(f"--- Collecting OHLVC data for {ticker} ---")
    df = yf.download(ticker, start=START, end=END)
    print(f"Collected {len(df)} days of data")
    return df

# -------------------------------------------------------------

def collect_Macro():
    load_dotenv()
    
    # Verify FRED API key
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY not set")
    
    print("--- Collecting Macro Data from FRED ---")
    fred = Fred(api_key=FRED_API_KEY)
    
    # Define the data we want to collect
    ids = {
        "FED_FUNDS": "FEDFUNDS",
        "MatRate_10Y": "DGS10",
        "MatRate_2Y": "DGS2",
        "Mat_Rate_CURVE": "T10Y2Y",
        "CPI": "CPIAUCSL",
        "CORE_CPI": "CPILFESL",
        "UNEMPLOYMENT": "UNRATE",
        "JOBLESS_CLAIMS": "ICSA",
        "GDP": "GDP",
        "CONSUMER_SENTIMENT": "UMCSENT",
        "EPU": "USEPUINDD",
        "VIX": "VIXCLS"
    }
    
    # Collect each series
    series_dict = {}
    for name, sid in ids.items():
        series_dict[name] = fred.get_series(
            sid, 
            observation_start=START, 
            observation_end=END
        )
    
    # Create DataFrame with daily frequency
    df = pd.DataFrame(index=pd.date_range(START, END, freq='D'))
    
    # Add each series and forward-fill
    for name, series in series_dict.items():
        df[name] = series
        df[name] = df[name].ffill()
    
    # Filter to business days only
    df = df[df.index.dayofweek < 5]
    
    print(f"Collected {len(df)} days of macro data")
    return df

# -------------------------------------------------------------

def collect_google_trends(keywords, start_date, end_date, geo="US", sleep=5, backoff=60, max_retries=5):    

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    pytrends = TrendReq(hl="en-US", tz=360)
    all_trends = []
    current_start = start.normalize()
    
    print(f"--- Starting Google Trends Collection for {start.date()} to {end.date()} ---")
    
    while current_start <= end:
        current_end = current_start + pd.Timedelta(days=240)
        if current_end > end:
            current_end = end
        
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        print(f"Fetching trends for period: {timeframe} ...")
        
        chunk_trends: Optional[pd.DataFrame] = None
        retries = 0
        
        # try to fetch data with retries
        while retries <= max_retries:
            try:
                pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
                chunk_trends = pytrends.interest_over_time()
                break # if the collection is successful, exit the retry loop
            
            except Exception as e:
                msg = str(e).lower() # store the error message
                
                is_429 = False
                try:
                    if hasattr(e, "response") and getattr(e.response, "status_code", None) == 429:
                        is_429 = True
                except Exception:
                    pass

                # if the error indicates rate limiting, retry after waiting
                if "429" in msg or "rate limit" in msg or "too many" in msg or is_429:
                    retries += 1
                    wait = backoff * retries
                    print(f"WARNING: Rate-limited (attempt {retries}/{max_retries}). Waiting {wait} seconds ...")
                    time.sleep(wait)
                    continue
                else: # for other errors, output and skip
                    print(f"ERROR: Unexpected error for timeframe {timeframe}. Skipping chunk. Error: {e}")
                    chunk_trends = None
                    break
        
        # Check if max retries exceeded and skip if so
        if retries > max_retries:
            print(f"ERROR: Exceeded max retries ({max_retries}) for timeframe {timeframe}. Skipping chunk.")
            chunk_trends = None
        
        # Process and store the chunk
        if chunk_trends is not None and not chunk_trends.empty:
            if "isPartial" in chunk_trends.columns:
                chunk_trends = chunk_trends.drop(columns=["isPartial"])
            all_trends.append(chunk_trends)
            print("Success.")
        else:
            print("No data returned.")
        
        current_start = current_end + pd.Timedelta(days=1)
        time.sleep(sleep)
    
    # if no data is collected return a blank DataFrame
    if not all_trends:
        return pd.DataFrame()
    
    result = pd.concat(all_trends).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    
    print(f"Collected {len(result)} days of Google Trends data")
    return result

# -------------------------------------------------------------

# Might not need this beacuse I have the NYT API collection, but leaving it for now
def get_daily_news_sentiment(start_date, end_date):
    print("--- Fetching SF Fed Daily News Sentiment Index ---")
    
    base_page_url = "https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/"
    
    try:
        print(f"Scraping main page: {base_page_url}")
        page_response = requests.get(base_page_url) # Get the main page
        page_response.raise_for_status() # Check for request errors
        
        soup = BeautifulSoup(page_response.content, 'lxml') # Parse HTML content
        download_link_tag = soup.find('a', string=lambda text: 'Daily News Sentiment data' in str(text)) # Find the download link
        
        if not download_link_tag or not download_link_tag.has_attr('href'): # If link not found, raise error
            raise ValueError("Could not find the download link on the page.")
        
        relative_url = download_link_tag['href'] # Extract the href attribute
        file_url = urljoin(base_page_url, relative_url) # Construct full URL
        
        print(f"Found download link: {file_url}")

        print("Downloading file content...")
        file_response = requests.get(file_url) # Download the file
        file_response.raise_for_status() # Check for request errors
        
        with io.BytesIO(file_response.content) as file_in_memory: # Read the file into memory
            df = pd.read_excel(file_in_memory, index_col=0, parse_dates=True, sheet_name='Data') # Load into DataFrame
        
        df.index = pd.to_datetime(df.index, errors='coerce') # Convert index to datetime
        df_filtered = df.loc[start_date:end_date] # Filter by date range
        
        print(f"Returned {len(df_filtered)} records.")
        return df_filtered
    
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------

def collect_nyt_top_daily_articles(start_date, end_date, articles_per_day=15):    

    # Load environment variables
    load_dotenv()
    
    # Verify NYT API key
    NYT_API_KEY = os.getenv("NEWS_API_KEY")
    if NYT_API_KEY is None:
        raise RuntimeError("NEWS_API_KEY not set in .env file")
    

    print(f"--- Collecting Top {articles_per_day} Articles Per Day: {start_date} to {end_date} ---")
    
    # define the date range
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    current = pd.to_datetime(start_date).replace(day=1)
    
    # Set up weights for sections, desks, keywords
    section_weights = {'Business Day': 20, 'Markets': 20, 'Economy': 18, 'Business': 15, 
                      'Financial': 15, 'Front Page': 12, 'Technology': 10, 'U.S.': 5}
    desk_weights = {'Business': 10, 'Financial': 10, 'Business Day': 10, 'National': 3}
    keywords = {'market': 3, 'stock': 3, 'recession': 5, 'inflation': 4, 'crash': 5, 
                'debt': 4, 'earnings': 3, 'volatility': 4, 'unemployment': 4}
    
    all_articles = []
    
    # Loop through each month in the date range
    while current <= pd.to_datetime(end_date):
        year, month = current.year, current.month # get year and month
        url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json" # construct API URL
        
        for attempt in range(3):  # Try 3 times per month
            try:
                # Make API request
                resp = requests.get(url, params={'api-key': NYT_API_KEY}, timeout=120)
                resp.raise_for_status()
                
                # Process articles for the current month
                daily_articles = {}

                # loop through every article in the response
                for article in resp.json()['response']['docs']:
                    pub_date = pd.to_datetime(article['pub_date']).date() # get publication date
                    if not (start <= pub_date <= end): # if the date is outside the range, skip
                        continue
                    
                    # Calculate relevance score
                    score = section_weights.get(article.get('section_name', ''), 0) # add section weight
                    score += desk_weights.get(article.get('news_desk', ''), 0) # add desk weight
                    wc = article.get('word_count', 0) # get the word count from the json
                    score += 10 if wc > 1500 else (8 if wc > 1000 else (5 if wc > 500 else 0)) # check the word count and increase the score accordingly
                    text = (article.get('headline', {}).get('main', '') + ' ' + 
                           article.get('abstract', '')).lower()
                    score += sum(w for k, w in keywords.items() if k in text) # add keyword weights
                    
                    # Minimum score filter
                    if score < 10:
                        continue
                    
                    # Group articles by publication date
                    if pub_date not in daily_articles:
                        daily_articles[pub_date] = []
                    
                    # Store article details with the score
                    daily_articles[pub_date].append({
                        'date': pub_date,
                        'headline': article['headline']['main'],
                        'abstract': article.get('abstract', ''),
                        'lead_paragraph': article.get('lead_paragraph', ''),
                        'section': article.get('section_name', ''),
                        'news_desk': article.get('news_desk', ''),
                        'word_count': wc,
                        'web_url': article.get('web_url', ''),
                        '_score': score
                    })
                
                # Get top N articles per day by score
                for _, arts in daily_articles.items():
                    all_articles.extend(sorted(arts, key=lambda x: x['_score'], reverse=True)[:articles_per_day]) # sort and take top N articles
                
                print(f"{year}-{month:02d}: {len(daily_articles)} days") 
                break  # Success - move to next month
                
            except Exception as e: # On error, retry after waiting
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))  # Wait 30s, then 60s
                else:
                    print(f"{year}-{month:02d}: Skipped after 3 attempts")
        
        current += pd.DateOffset(months=1) # move to next month
        time.sleep(15) # wait between months to avoid rate limiting
    
    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    if not df.empty: # if there are articles collected
        df = df.sort_values('date').reset_index(drop=True).drop(columns=['_score']) # sort by date and remove score column
    
    print(f"Complete: {len(df):,} articles")
    return df

# -------------------------------------------------------------

def save_data():    
    print("STARTING DATA COLLECTION")
    
    # Collect all data
    ohlvc_data = collect_OHLVC("SPY")
    macro_data = collect_Macro()
    
    trends_keywords = ["Debt", "Recession", "Stocks to buy", "Unemployment", "Market crash"]
    google_trends_data = collect_google_trends(trends_keywords, START, END)
    
    # sentiment_data = get_daily_news_sentiment(START, END)
    
    nyt_data = collect_nyt_top_daily_articles(START, END, articles_per_day=15)
    
    # Save all data
    print("SAVING DATA TO CSV FILES")
    
    ohlvc_data.to_csv("../data/raw/US_OHLVC_Data.csv")
    print("Saved: US_OHLVC_Data.csv")
    
    macro_data.to_csv("../data/raw/US_Macro_Data.csv")
    print("Saved: US_Macro_Data.csv")
    
    google_trends_data.to_csv("../data/raw/US_Google_Trends_Data.csv")
    print("Saved: US_Google_Trends_Data.csv")
    
    # sentiment_data.to_csv("../data/raw/US_Sentiment_Data.csv")
    # print("Saved: US_Sentiment_Data.csv")
    
    # Save NYT data
    nyt_data.to_csv("../data/raw/NYT_Top_Daily_Articles.csv", index=False)
    print("Saved: NYT_Top_Daily_Articles.csv")
    
    print("ALL DATA COLLECTION COMPLETE")

# -------------------------------------------------------------
# Main execution to collect and save all data
if __name__ == '__main__':
    save_data()
   
    