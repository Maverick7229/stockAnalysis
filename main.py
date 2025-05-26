from fastapi import FastAPI
from claudeCode2 import run_enhanced_ticker_extraction

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Running"}

@app.get("/tickers")
def get_tickers():
    try:
        tickers = run_enhanced_ticker_extraction()
        return {"tickers": tickers}
    except Exception as e:
        return {"error": str(e)}
