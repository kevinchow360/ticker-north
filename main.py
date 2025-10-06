from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

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

        # Additional Basic Metrics
        sortino_ratio = avg_return / data['Returns'][data['Returns'] < 0].std() if len(data['Returns'][data['Returns'] < 0]) > 1 else 0
        win_rate = (data['Returns'] > 0).sum() / len(data['Returns']) * 100
        rolling_3m = data['Returns'].rolling(window=63).sum().mean() * 100  # approx 3 months
        rolling_6m = data['Returns'].rolling(window=126).sum().mean() * 100  # approx 6 months
        rolling_1y = data['Returns'].rolling(window=252).sum().mean() * 100  # approx 1 year

        return {
            "ticker": ticker_symbol.upper(),
            "average_annual_return": round(avg_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate": round(win_rate, 2),
            "rolling_3m": round(rolling_3m, 2),
            "rolling_6m": round(rolling_6m, 2),
            "rolling_1y": round(rolling_1y, 2)
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

@app.get("/api/ticker/{ticker_symbol}", response_class=HTMLResponse)
def get_ticker_table(ticker_symbol: str):
    result = calculate_metrics(ticker_symbol)
    if "error" in result:
        return f"<h3 style='color:red'>{result['error']}</h3>"

    html = f"""
    <html>
    <head>
        <title>TICKER-NORTH Metrics for {result['ticker']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa; }}
            table {{ border-collapse: collapse; width: 70%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Metrics for {result['ticker']}</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Annual Return</td><td class='{"positive" if result["average_annual_return"]>=0 else "negative"}'>{result["average_annual_return"]}%</td></tr>
            <tr><td>Volatility</td><td>{result["volatility"]}%</td></tr>
            <tr><td>Max Drawdown</td><td class='{"negative" if result["max_drawdown"]<0 else "positive"}'>{result["max_drawdown"]}%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{result["sharpe_ratio"]}</td></tr>
            <tr><td>Sortino Ratio</td><td>{result["sortino_ratio"]}</td></tr>
            <tr><td>Win Rate</td><td>{result["win_rate"]}%</td></tr>
            <tr><td>Rolling 3M</td><td>{result["rolling_3m"]}%</td></tr>
            <tr><td>Rolling 6M</td><td>{result["rolling_6m"]}%</td></tr>
            <tr><td>Rolling 1Y</td><td>{result["rolling_1y"]}%</td></tr>
        </table>
    </body>
    </html>
    """
    return html
