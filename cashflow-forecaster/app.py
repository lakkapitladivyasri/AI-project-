"""
app.py
--------
FastAPI service exposing the cash-flow forecaster.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /run-forecast   -- runs a fresh forecast + evaluation, returns the summary
    GET  /daily-forecast  -- the full day-by-day forecast from the last run
    GET  /health
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from forecast_agent import run as run_forecast, REPORTS_DIR

app = FastAPI(title="Razorpay Buildathon — Cash-Flow Forecaster", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-forecast")
def run_forecast_endpoint():
    try:
        summary = run_forecast()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Source data not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return summary


@app.get("/daily-forecast")
def daily_forecast(flagged_only: bool = False):
    path = REPORTS_DIR / "daily_forecast.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No forecast run yet. POST /run-forecast first.")
    df = pd.read_csv(path)
    if flagged_only:
        df = df[df["cash_risk_flag"] == True]  # noqa: E712
    return {"count": len(df), "days": df.to_dict(orient="records")}
