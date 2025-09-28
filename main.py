from fastapi import FastAPI
import yfinance as yf
import numpy as np

app = FastAPI(title="TICKER-NORTH API")

def calculate_metrics(ticker_symbol: str):
    try:
        data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return {"error": f"No data found for ticker '{ticker_symbol}'"}

        price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        if price_column not in data.columns:
            return {"error": f"No valid price data for ticker '{ticker_symbol}'"}

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

# --- API endpoint ---
@app.get("/api/ticker/{ticker_symbol}")
def get_ticker_metrics(ticker_symbol: str):
    return calculate_metrics(ticker_symbol)
