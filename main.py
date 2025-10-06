from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

# Allow CORS from your Namecheap site and any other origins
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

# --- Calculation function ---
def calculate_metrics(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty or 'Adj Close' not in data.columns:
            raise ValueError(f"No data found for '{ticker_symbol}'")

        data['Returns'] = data['Adj Close'].pct_change().dropna()
        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        # --- Basic metrics ---
        sortino_ratio = avg_return / data['Returns'][data['Returns'] < 0].std() * np.sqrt(252) if any(data['Returns'] < 0) else 0
        rolling_3m = (1 + data['Returns'].rolling(63).mean()).cumprod().iloc[-1] - 1
        rolling_6m = (1 + data['Returns'].rolling(126).mean()).cumprod().iloc[-1] - 1
        rolling_1y = (1 + data['Returns'].rolling(252).mean()).cumprod().iloc[-1] - 1
        win_rate = len(data[data['Returns'] > 0]) / len(data['Returns'])
        beta = np.cov(data['Returns'], data['Adj Close'])[0,1] / np.var(data['Adj Close']) if len(data) > 1 else 0

        return {
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return*100,2),
            "volatility": round(volatility*100,2),
            "max_drawdown": round(max_drawdown*100,2),
            "sharpe_ratio": round(sharpe_ratio,2),
            "sortino_ratio": round(sortino_ratio,2),
            "rolling_3m": round(rolling_3m*100,2),
            "rolling_6m": round(rolling_6m*100,2),
            "rolling_1y": round(rolling_1y*100,2),
            "win_rate": round(win_rate*100,2),
            "beta": round(beta,2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- API endpoint ---

from fastapi.responses import HTMLResponse

@app.get("/ticker-page/{ticker_symbol}", response_class=HTMLResponse)
def ticker_page(ticker_symbol: str):
    data = calculate_metrics(ticker_symbol)
    table_html = "<table><tr><th>Metric</th><th>Value</th></tr>"
    for key, value in data.items():
        if key == "ticker":
            continue
        cls = "positive" if value >= 0 and key != "max_drawdown" else "negative"
        table_html += f"<tr><td>{key.replace('_',' ').title()}</td><td class='{cls}'>{value}</td></tr>"
    table_html += "</table>"

    html = f"""
    <html>
    <head>
    <title>Metrics for {data['ticker']}</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        table {{ border-collapse: collapse; width: 60%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
    </style>
    </head>
    <body>
        <h2>Metrics for {data['ticker']}</h2>
        {table_html}
    </body>
    </html>
    """
    return HTMLResponse(content=html)

