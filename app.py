from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

# Enable CORS for your front-end domains
CORS(app, origins=[
    "https://tickernorth.com",
    "https://www.tickernorth.com",
    "https://sitebuilder1.web-hosting.com"
])

def calculate_metrics(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return {"error": f"No data found for ticker '{ticker_symbol}'"}

        price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        data['Returns'] = data[price_column].pct_change().dropna()
        if len(data['Returns']) < 2:
            return {"error": f"Not enough data to calculate metrics for '{ticker_symbol}'"}

        # Free Metrics
        avg_return = data['Returns'].mean() * 252
        volatility = data['Returns'].std() * np.sqrt(252)
        cum_returns = (1 + data['Returns']).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        # Basic Metrics
        cagr = (data[price_column][-1] / data[price_column][0]) ** (1/5) - 1  # 5-year CAGR
        negative_returns = data['Returns'][data['Returns'] < 0]
        sortino_ratio = avg_return / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0

        rolling_3m = data['Returns'].rolling(window=63).apply(lambda x: (1 + x).prod() - 1).iloc[-1]  # 3M
        rolling_6m = data['Returns'].rolling(window=126).apply(lambda x: (1 + x).prod() - 1).iloc[-1] # 6M
        rolling_1y = data['Returns'].rolling(window=252).apply(lambda x: (1 + x).prod() - 1).iloc[-1] # 1Y

        win_rate = (data['Returns'] > 0).sum() / len(data['Returns'])
        max_consec_gain = max((len(list(g)) for k,g in pd.Series(data['Returns']>0).groupby((data['Returns']>0).ne(pd.Series(data['Returns']>0)).cumsum()) if k), default=0)
        max_consec_loss = max((len(list(g)) for k,g in pd.Series(data['Returns']<0).groupby((data['Returns']<0).ne(pd.Series(data['Returns']<0)).cumsum()) if k), default=0)

        beta = np.nan  # placeholder for SPY beta (requires extra SPY data)

        return {
            "ticker": ticker_symbol.upper(),
            "free_metrics": {
                "average_annual_return": round(avg_return*100,2),
                "volatility": round(volatility*100,2),
                "max_drawdown": round(max_drawdown*100,2),
                "sharpe_ratio": round(sharpe_ratio,2)
            },
            "basic_metrics": {
                "cagr": round(cagr*100,2),
                "sortino_ratio": round(sortino_ratio,2),
                "rolling_returns": {
                    "3M": round(rolling_3m*100,2),
                    "6M": round(rolling_6m*100,2),
                    "1Y": round(rolling_1y*100,2)
                },
                "win_rate": round(win_rate*100,2),
                "max_consecutive_gain": max_consec_gain,
                "max_consecutive_loss": max_consec_loss,
                "beta_vs_spy": beta
            }
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

# --- HTML template ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TICKER-NORTH Metrics</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa;}
table { border-collapse: collapse; width: 60%; margin-top: 20px;}
th, td { border: 1px solid #ccc; padding: 8px; text-align: center;}
th { background-color: #f2f2f2;}
.positive { color: green; font-weight: bold; }
.negative { color: red; font-weight: bold; }
.error { color: red; font-weight: bold; margin-top: 20px; }
</style>
</head>
<body>
<h2>TICKER-NORTH Metrics</h2>
<form method="get">
  Enter Ticker: <input type="text" name="ticker" placeholder="e.g., AAPL">
  <input type="submit" value="Calculate">
</form>

{% if result %}
    {% if result.error %}
        <div class="error">{{ result.error }}</div>
    {% else %}
        <h3>Results for {{ result['ticker'] }}</h3>
        <table>
            <tr><th>Free Metric</th><th>Value</th></tr>
            {% for key, val in result['free_metrics'].items() %}
                <tr><td>{{ key.replace('_',' ').title() }}</td><td>{{ val }}</td></tr>
            {% endfor %}
        </table>

        <table>
            <tr><th>Basic Metric</th><th>Value</th></tr>
            {% for key, val in result['basic_metrics'].items() %}
                {% if val is mapping %}
                    {% for subkey, subval in val.items() %}
                        <tr><td>{{ key.replace('_',' ').title() }} - {{ subkey }}</td><td>{{ subval }}</td></tr>
                    {% endfor %}
                {% else %}
                    <tr><td>{{ key.replace('_',' ').title() }}</td><td>{{ val }}</td></tr>
                {% endif %}
            {% endfor %}
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
