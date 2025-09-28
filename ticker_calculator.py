from flask import Flask, jsonify, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import os


def calculate_metrics(ticker_symbol):
    data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        return {"error": f"No data found for {ticker_symbol}"}
    
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data['Returns'] = data[price_column].pct_change().dropna()
    
    if len(data['Returns']) < 2:
        return {"error": f"Not enough data for {ticker_symbol}"}
    
    avg_return = data['Returns'].mean() * 252
    volatility = data['Returns'].std() * np.sqrt(252)
    cum_returns = (1 + data['Returns']).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    sharpe_ratio = avg_return / volatility if volatility != 0 else 0
    
    return {
        "ticker": ticker_symbol.upper(),
        "average_annual_return": round(avg_return*100,2),
        "volatility": round(volatility*100,2),
        "max_drawdown": round(max_drawdown*100,2),
        "sharpe_ratio": round(sharpe_ratio,2)
    }

# Example usage
if __name__ == "__main__":
    ticker = input("Enter ticker: ")
    metrics = calculate_metrics(ticker)
    print(metrics)

@app.route("/")
def home():
    # Serve your HTML front-end
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

