from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="TICKER-NORTH")

templates = Jinja2Templates(directory="templates")

def calculate_metrics(ticker_symbol):
    data = yf.download(ticker_symbol, period="5y", interval="1d", progress=False)
    if data.empty or "Adj Close" not in data.columns:
        raise HTTPException(status_code=404, detail=f"Data not available for {ticker_symbol}")

    data["Returns"] = data["Adj Close"].pct_change()
    avg_return = data["Returns"].mean() * 252
    volatility = data["Returns"].std() * np.sqrt(252)
    cum_returns = (1 + data["Returns"]).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    sharpe_ratio = avg_return / volatility if volatility != 0 else 0

    return {
        "ticker": ticker_symbol.upper(),
        "average_annual_return": round(avg_return * 100, 2),
        "volatility": round(volatility * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.get("/calculate", response_class=HTMLResponse)
def get_ticker(request: Request, ticker: str):
    try:
        result = calculate_metrics(ticker)
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
