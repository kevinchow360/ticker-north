from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

# Allow CORS from your Namecheap site
origins = [
    "https://tickernorth.com",
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions for Basic metrics ---
def calc_cagr(data):
    start_price = data['Adj Close'].iloc[0]
    end_price = data['Adj Close'].iloc[-1]
    years = (data.index[-1] - data.index[0]).days / 365
    return (end_price / start_price) ** (1 / years) - 1

def calc_sortino(returns):
    avg_return = returns.mean() * 252
    downside = returns[returns < 0].std() * np.sqrt(252)
    return avg_return / downside if downside != 0 else 0

def calc_rolling_returns(data):
    price = data['Adj Close']
    returns = {}
    for label, days in [("3M", 63), ("6M", 126), ("1Y", 252)]:
        if len(price) > days:
            returns[label] = (price.iloc[-1] / price.iloc[-days] - 1) * 100
    return returns

def calc_win_rate(returns):
    return (returns > 0).sum() / len(returns) * 100

def calc_max_consecutive(returns):
    gains = losses = max_gains = max_losses = 0
    for r in returns:
        if r > 0:
            gains += 1
            losses = 0
        elif r < 0:
            losses += 1
            gains = 0
        max_gains = max(max_gains, gains)
        max_losses = max(max_losses, losses)
    return {"max_gains": max_gains, "max_losses": max_losses}

def calc_beta_vs_spy(data):
    spy = yf.download("SPY", start=data.index[0], end=data.index[-1], interval="1d", progress=False)
    if spy.empty:
        return None
    stock_ret = data['Adj Close'].pct_change().dropna()
    spy_ret = spy['Adj Close'].pct_change().dropna()
    df = pd.concat([stock_ret, spy_ret], axis=1).dropna()
    df.columns = ['stock', 'spy']
    cov = np.cov(df['stock'], df['spy'])[0, 1]
    var = np.var(df['spy'])
    return cov / var if var != 0 else None

# --- Core metrics ---
def calculate_metrics(ticker_symbol, tier="free"):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False)
        if data.empty or 'Adj Close' not in data.columns:
            raise ValueError("No data found or 'Adj Close' not available")

        data['Returns'] = data['Adj Close'].pct_change().dropna()

        # Free metrics
        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        result = {
            "tier": tier,
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }

        # Add Basic metrics if requested
        if tier == "basic":
            result.update({
                "cagr": round(calc_cagr(data) * 100, 2),
                "sortino_ratio": round(calc_sortino(data['Returns']), 2),
                "rolling_returns": calc_rolling_returns(data),
                "win_rate": round(calc_win_rate(data['Returns']), 2),
                "max_consecutive": calc_max_consecutive(data['Returns']),
                "beta_vs_spy": round(calc_beta_vs_spy(data), 3)
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ticker/{ticker_symbol}")
def get_ticker_metrics(ticker_symbol: str, tier: str = "free"):
    return calculate_metrics(ticker_symbol, tier)
