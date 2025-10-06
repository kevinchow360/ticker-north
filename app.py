from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd


CORS(app, origins=[
    "https://tickernorth.com",
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com"
])

app = Flask(__name__)

# You can paste the same calc_* functions here as above (CAGR, Sortino, etc.)

# --- Core metrics function ---
def calculate_metrics(ticker_symbol, tier="free"):
    data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        return {"error": f"No data found for ticker '{ticker_symbol}'"}

    price = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data['Returns'] = data[price].pct_change().dropna()
    if len(data['Returns']) < 2:
        return {"error": f"Not enough data for '{ticker_symbol}'"}

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

@app.route("/api/ticker/<ticker_symbol>")
def api_ticker(ticker_symbol):
    tier = request.args.get("tier", "free")
    return jsonify(calculate_metrics(ticker_symbol, tier))


