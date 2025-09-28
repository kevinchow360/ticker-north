from flask import Flask, request, jsonify, render_template_string
import yfinance as yf
import numpy as np

app = Flask(__name__)

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

        return {
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

# --- HTML template ---
HTML_PAGE = """
<!doctype html>
<html>
<head>
<title>TICKER-NORTH Free Calculator</title>
<style>
    table { border-collapse: collapse; width: 50%; margin-top: 20px;}
    th, td { border: 1px solid #ccc; padding: 8px; text-align: center;}
    th { background-color: #f2f2f2;}
    .positive { color: green; font-weight: bold; }
    .negative { color: red; font-weight: bold; }
    .error { color: red; font-weight: bold; margin-top: 20px; }
</style>
</head>
<body>
<h2>TICKER-NORTH Free Calculator</h2>
<form method="get">
  Ticker Symbol: <input type="text" name="ticker">
  <input type="submit" value="Calculate">
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
        </table>
    {% endif %}
{% endif %}
</body>
</html>
"""

# --- Web page ---
@app.route("/", methods=["GET"])
def index():
    ticker = request.args.get("ticker")
    result = None
    if ticker:
        result = calculate_metrics(ticker)
    return render_template_string(HTML_PAGE, result=result)

# --- JSON API endpoint ---
@app.route("/api/ticker/<ticker_symbol>", methods=["GET"])
def api_ticker(ticker_symbol):
    return jsonify(calculate_metrics(ticker_symbol))

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


