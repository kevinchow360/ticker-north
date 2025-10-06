from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

# CORS for your website
origins = [
    "https://tickernorth.com",
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com/fQ8ky/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_metrics(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for ticker '{ticker_symbol}'")

        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        data['Returns'] = data[price_col].pct_change().dropna()
        if len(data['Returns']) < 2:
            raise ValueError(f"Not enough data for '{ticker_symbol}'")

        # Free-tier metrics
        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        free_metrics = {
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }

        # Basic-tier metrics
        cagr = (cum_returns.iloc[-1])**(1/5) - 1  # 5-year CAGR
        sortino_ratio = (avg_return / (data['Returns'][data['Returns'] < 0].std() * np.sqrt(252))
                         if any(data['Returns'] < 0) else 0)
        rolling_3m = data['Returns'].rolling(63).apply(lambda x: (1 + x).prod() - 1).iloc[-1]  # ~63 trading days
        rolling_6m = data['Returns'].rolling(126).apply(lambda x: (1 + x).prod() - 1).iloc[-1]
        rolling_1y = data['Returns'].rolling(252).apply(lambda x: (1 + x).prod() - 1).iloc[-1]
        win_rate = (data['Returns'] > 0).sum() / len(data['Returns'])
        max_consec_gain = max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(data['Returns'] > 0) if k), default=0)
        max_consec_loss = max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(data['Returns'] < 0) if k), default=0)
        # Beta vs SPY example
        spy = yf.download('SPY', period="5y", interval="1d", progress=False, auto_adjust=True)['Adj Close'].pct_change().dropna()
        beta = np.cov(data['Returns'].iloc[-len(spy):], spy)[0,1] / np.var(spy) if len(spy) > 1 else 0

        basic_metrics = {
            "CAGR": round(cagr * 100, 2),
            "Sortino_ratio": round(sortino_ratio, 2),
            "Rolling_returns_3M": round(rolling_3m * 100, 2),
            "Rolling_returns_6M": round(rolling_6m * 100, 2),
            "Rolling_returns_1Y": round(rolling_1y * 100, 2),
            "Win_rate": round(win_rate * 100, 2),
            "Max_consecutive_gains": max_consec_gain,
            "Max_consecutive_losses": max_consec_loss,
            "Beta_vs_SPY": round(beta, 2)
        }

        return {
            "ticker": ticker_symbol.upper(),
            "free_metrics": free_metrics,
            "basic_metrics": basic_metrics
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ticker/{ticker_symbol}")
def get_ticker_metrics(ticker_symbol: str):
    return calculate_metrics(ticker_symbol)
