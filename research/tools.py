import yfinance as yf
from langchain.tools import tool

@tool
def yf(ticker):
    """ use this tool to get stocks data from yfinace """


# Replace 'AAPL' with your desired ticker symbol
    ticker = yf.Ticker(ticker)

# Get daily historical market data for the last 30 days
    daily_data = ticker.history(period="1d")  # Last 30 days
    return(daily_data)

    
