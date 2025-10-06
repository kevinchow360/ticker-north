from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Enable CORS
CORS(app, origins=[
    "https://tickernorth.com",
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com"
])

logging.basicConfig(level=logging.INFO)

# --- Metrics calculation ---
def calculate_metrics(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return {"error": f"No data found for ticker '{ticker_symbol}'"}

        price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        data['Returns'] = data[price_column].pct_change().dropna()
        if len(data['Returns']) < 2:
            return {"error": f"Not enough data to calculate metrics for '{ticker_symbol}'"}

        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        # Basic tier metrics
        sortino_ratio = 0  # placeholder; implement if needed
        win_rate = (data['Returns'] > 0).mean()
        rolling_3m = data['Returns'].rolling(63).sum().iloc[-1] * 100  # ~63 trading days per 3 months

        return {
            "ticker": ticker_symbol.upper(),
            # Free tier
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            # Basic tier
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate": round(win_rate * 100, 2),
            "rolling_3m": round(rolling_3m, 2)
        }

    except Exception as e:
        logging.error(f"Error fetching metrics for {ticker_symbol}: {e}")
        return {"error": f"Unexpected error: {e}"}

# --- HTML Template ---
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TICKER-NORTH Calculator</title>
<style>
    body { font-family: Arial; padding: 20px; background: #f8f9fa; }
    input[type=text] { padding: 8px 12px; width: 200px; border-radius: 5px; border: 1px solid #ccc; }
    button { padding: 8px 16px; border-radius: 5px; background-color: #5b9cf6; color: #fff; border: none; cursor: pointer; }
    button:hover { background-color: #4a8de0; }
    table { border-collapse: collapse; margin-top: 20px; width: 60%; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
    th { background-color: #eee; }
    .positive { color: green; font-weight: bold; }
    .negative { color: red; font-weight: bold; }
    .error { color: red; font-weight: bold; margin-top: 20px; }
</style>
</head>
<body>
<h2>TICKER-NORTH Calculator</h2>
<form method="get">
    Ticker: <input type="text" name="ticker" placeholder="e.g., AAPL">
    <button type="submit">Calculate</button>
</form>

{% if result %}
    {% if result.error %}
        <div class="error">{{ result.error }}</div>
    {% else %}
        <h3>Results for {{ result['ticker'] }}</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Annual Return</td><td class="{{ 'positive' if result['average_annual_return']>=0 else 'negative' }}">{{ result['average_annual_return'] }}%</td></tr>
            <tr><td>Volatility</td><td>{{ result['volatility'] }}%</td></tr>
            <tr><td>Max Drawdown</td><td class="{{ 'negative' if result['max_drawdown']<0 else 'positive' }}">{{ result['max_drawdown'] }}%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{{ result['sharpe_ratio'] }}</td></tr>
            <tr><td>Sortino Ratio</td><td>{{ result['sortino_ratio'] }}</td></tr>
            <tr><td>Win Rate</td><td>{{ result['win_rate'] }}%</td></tr>
            <tr><td>Rolling 3M Return</td><td>{{ result['rolling_3m'] }}%</td></tr>
        </table>
    {% endif %}
{% endif %}
</body>
</html>
"""

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    ticker = request.args.get("ticker")
    result = None
    if ticker:
        result = calculate_metrics(ticker)
    return render_template_string(HTML_PAGE, result=result)

@app.route("/api/ticker/<ticker_symbol>", methods=["GET"])
def api_ticker(ticker_symbol):
    return jsonify(calculate_metrics(ticker_symbol))

# --- Run ---
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

