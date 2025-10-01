from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

# Allow CORS from your Namecheap site
origins = [
    "https://tickernorth.com",               # your main website
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com"  # add your Namecheap builder domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing API function
def calculate_metrics(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False)
        if data.empty or 'Adj Close' not in data.columns:
            raise ValueError("No data found or 'Adj Close' not available")

        data['Returns'] = data['Adj Close'].pct_change()
        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        return {
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ticker/{ticker_symbol}")
def get_ticker_metrics(ticker_symbol: str):
    return calculate_metrics(ticker_symbol)
