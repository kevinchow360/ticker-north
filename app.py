from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import os
from openai import OpenAI

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

        sortino_ratio = 0  # placeholder
        win_rate = (data['Returns'] > 0).mean()
        rolling_3m = data['Returns'].rolling(63).sum().iloc[-1] * 100

        return {
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate": round(win_rate * 100, 2),
            "rolling_3m": round(rolling_3m, 2)
        }

    except Exception as e:
        logging.error(f"Error fetching metrics for {ticker_symbol}: {e}")
        return {"error": f"Unexpected error: {e}"}

# --- HTML PAGE ---
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
    table { border-collapse: collapse; margin-top: 20px; width: 60%; max-width: 600px; }
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

# --- Avexineer Script Generator ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/generate", methods=["GET", "POST"])
def generate_script():
    if request.method == "GET":
        return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Avexineer - Pine Script Generator</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 30px; }
                .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                textarea, input[type=text] { width: 100%; padding: 10px; margin-top: 10px; border-radius: 5px; border: 1px solid #ccc; }
                button { margin-top: 10px; padding: 10px 15px; background: #5b9cf6; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #4a8de0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Avexineer 🧠 — Pine Script Generator</h2>
                <form method="POST">
                    <label for="description">What kind of script do you want?</label>
                    <textarea id="description" name="description" rows="4" placeholder="e.g., A Keltner Channel breakout strategy with alerts..."></textarea>
                    <button type="submit">Generate Script</button>
                </form>
            </div>
        </body>
        </html>
        """)

    user_input = request.form.get("description", "").strip()
    if not user_input:
        return "<p>Please describe the script you want.</p>"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert Pine Script v5 developer."},
            {"role": "user", "content": f"Write a complete TradingView Pine Script v5 code for: {user_input}"}
        ],
        temperature=0.4,
    )
    pine_code = response.choices[0].message.content

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Generated Pine Script</title>
        <style>
            body { font-family: monospace; background: #f8f9fa; padding: 30px; }
            .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            pre { background: #f0f0f0; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }
            button { margin-top: 10px; padding: 10px 15px; background: #5b9cf6; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Generated Pine Script</h2>
            <pre id="code">{{ code }}</pre>
            <button onclick="copyCode()">Copy to Clipboard</button>
        </div>
        <script>
            function copyCode() {
                const code = document.getElementById('code').innerText;
                navigator.clipboard.writeText(code);
                alert('Copied!');
            }
        </script>
    </body>
    </html>
    """, code=pine_code)

# --- Run ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

