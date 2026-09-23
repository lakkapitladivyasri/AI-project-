"""
forecast_agent.py
--------------------
The forecasting agent for the AI Finance Controller track (Cash-flow
forecaster direction). Predicts the daily cash position for the next 30
days and flags any day where the projected balance risks dropping below
a safety threshold — then validates itself against real held-out actuals
so the reported accuracy is honest, not asserted.

Design: a HYBRID forecast, not one black-box model guessing everything:
  - Organic inflow is genuinely uncertain -> modeled statistically
    (linear regression on day-of-week + trend, trained on history only).
  - Scheduled outflows (payroll, GST, rent, weekly vendor payments) are
    NOT uncertain — a finance team already knows these dates and amounts
    for certain. The agent uses the deterministic `known_schedule.csv`
    directly for these instead of trying to statistically predict
    something that's already known, which would only add error.

Bounded scope: this agent produces a forecast and a risk-flag list. It
never moves money, approves a payment, or delays a scheduled outflow —
those decisions are for a human, informed by this report.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

CASH_RISK_THRESHOLD = 1_150_000.0   # flag any day the lower confidence bound dips below this
                                      # (set relative to this business's typical operating
                                      # balance — see README for how to size this per business)
CONFIDENCE_Z = 1.28                  # ~80% one-sided confidence band on residual std


@dataclass
class DayForecast:
    date: str
    forecast_inflow: float
    known_outflow: float
    forecast_operational_outflow: float
    forecast_net_flow: float
    forecast_closing_balance: float
    lower_bound_balance: float
    upper_bound_balance: float
    cash_risk_flag: bool
    flag_reason: str


class CashFlowForecaster:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def _build_features(self, dates: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"date": dates})
        df["date"] = pd.to_datetime(df["date"])
        df["day_index"] = (df["date"] - pd.Timestamp("2026-01-01")).dt.days
        for wd in range(7):
            df[f"wd_{wd}"] = (df["date"].dt.dayofweek == wd).astype(int)
        return df

    def fit(self, history: pd.DataFrame):
        feats = self._build_features(history["date"])
        X = feats[["day_index"] + [f"wd_{i}" for i in range(7)]]

        self.inflow_model = LinearRegression().fit(X, history["inflow"])
        inflow_resid = history["inflow"] - self.inflow_model.predict(X)
        self.inflow_resid_std = float(inflow_resid.std())

        self.opflow_model = LinearRegression().fit(X, history["operational_outflow"])
        opflow_resid = history["operational_outflow"] - self.opflow_model.predict(X)
        self.opflow_resid_std = float(opflow_resid.std())

    def forecast(self, forecast_dates: pd.Series, known_schedule: pd.DataFrame,
                 opening_balance: float) -> list[DayForecast]:
        feats = self._build_features(forecast_dates)
        X = feats[["day_index"] + [f"wd_{i}" for i in range(7)]]

        pred_inflow = np.clip(self.inflow_model.predict(X), 0, None)
        pred_opflow = np.clip(self.opflow_model.predict(X), 0, None)

        known_map = dict(zip(known_schedule["date"], known_schedule["known_outflow"]))
        known_label = dict(zip(known_schedule["date"], known_schedule["label"]))

        results = []
        balance = opening_balance
        # Uncertainty compounds day over day (independent daily errors),
        # so the confidence band widens the further out the forecast goes.
        cumulative_variance = 0.0

        for i, date in enumerate(forecast_dates):
            known_out = float(known_map.get(date, 0.0))
            net_flow = round(float(pred_inflow[i] - pred_opflow[i] - known_out), 2)
            balance = round(balance + net_flow, 2)

            cumulative_variance += self.inflow_resid_std**2 + self.opflow_resid_std**2
            band = CONFIDENCE_Z * (cumulative_variance ** 0.5)
            lower = round(balance - band, 2)
            upper = round(balance + band, 2)

            flag = lower < CASH_RISK_THRESHOLD
            reason = (f"Lower confidence bound Rs {lower:,.2f} is below the Rs "
                      f"{CASH_RISK_THRESHOLD:,.2f} safety threshold"
                      + (f" (driven partly by known {known_label.get(date,'')} outflow)"
                         if known_out > 0 else "")) if flag else "Within safe range"

            results.append(DayForecast(
                date=date,
                forecast_inflow=round(float(pred_inflow[i]), 2),
                known_outflow=round(known_out, 2),
                forecast_operational_outflow=round(float(pred_opflow[i]), 2),
                forecast_net_flow=net_flow,
                forecast_closing_balance=balance,
                lower_bound_balance=lower,
                upper_bound_balance=upper,
                cash_risk_flag=flag,
                flag_reason=reason,
            ))

        return results

    def evaluate_against_actuals(self, forecasts: list[DayForecast], actual_future: pd.DataFrame) -> dict:
        actual_map = dict(zip(actual_future["date"], actual_future["closing_balance"]))
        errors, pct_errors, within_band = [], [], 0

        for f in forecasts:
            actual = actual_map.get(f.date)
            if actual is None:
                continue
            err = f.forecast_closing_balance - actual
            errors.append(err)
            pct_errors.append(abs(err) / actual if actual else 0)
            if f.lower_bound_balance <= actual <= f.upper_bound_balance:
                within_band += 1

        errors = np.array(errors)
        n = len(errors)
        return {
            "days_evaluated": n,
            "mae": round(float(np.mean(np.abs(errors))), 2),
            "mape_pct": round(float(np.mean(pct_errors)) * 100, 2),
            "bias": round(float(np.mean(errors)), 2),
            "final_day_forecast_balance": forecasts[-1].forecast_closing_balance,
            "final_day_actual_balance": float(actual_map.get(forecasts[-1].date, float("nan"))),
            "pct_days_within_confidence_band": round(within_band / n * 100, 1) if n else 0.0,
        }


def run(data_dir: Path = DATA_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    history = pd.read_csv(data_dir / "cash_flow_history.csv")
    actual_future = pd.read_csv(data_dir / "actual_future.csv")
    known_schedule = pd.read_csv(data_dir / "known_schedule.csv")

    model = CashFlowForecaster(data_dir)
    model.fit(history)

    opening_balance = float(history["closing_balance"].iloc[-1])
    forecasts = model.forecast(actual_future["date"], known_schedule, opening_balance)

    eval_metrics = model.evaluate_against_actuals(forecasts, actual_future)
    flagged_days = [f for f in forecasts if f.cash_risk_flag]

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_days": len(forecasts),
        "opening_balance": round(opening_balance, 2),
        "cash_risk_threshold": CASH_RISK_THRESHOLD,
        "flagged_days_count": len(flagged_days),
        "flagged_dates": [f.date for f in flagged_days],
        "evaluation_against_held_out_actuals": eval_metrics,
    }

    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "forecast_report.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([asdict(f) for f in forecasts]).to_csv(reports_dir / "daily_forecast.csv", index=False)

    with open(reports_dir / "audit_log.jsonl", "w") as f:
        for day in forecasts:
            f.write(json.dumps(asdict(day)) + "\n")

    md = [
        "# Cash-Flow Forecast Report\n",
        f"Run: {summary['run_timestamp']}\n",
        f"- Forecast horizon: {summary['forecast_horizon_days']} days",
        f"- Opening balance: Rs {summary['opening_balance']:,.2f}",
        f"- Cash risk threshold: Rs {summary['cash_risk_threshold']:,.2f}",
        f"- **Days flagged as cash-risk: {summary['flagged_days_count']}** ({', '.join(summary['flagged_dates']) or 'none'})\n",
        "## Accuracy against held-out actuals (never seen during training)",
        f"- Mean Absolute Error (MAE): Rs {eval_metrics['mae']:,.2f}",
        f"- Mean Absolute Percentage Error (MAPE): {eval_metrics['mape_pct']}%",
        f"- Bias (forecast \u2212 actual, average): Rs {eval_metrics['bias']:,.2f}",
        f"- Final-day forecast balance: Rs {eval_metrics['final_day_forecast_balance']:,.2f} "
        f"vs actual Rs {eval_metrics['final_day_actual_balance']:,.2f}",
        f"- % of days the actual balance fell within the forecast's confidence band: "
        f"{eval_metrics['pct_days_within_confidence_band']}%\n",
        "See `daily_forecast.csv` for the full day-by-day forecast, including which known scheduled",
        "outflows (payroll/GST/rent/vendor) were factored in deterministically for each day.",
    ]
    (reports_dir / "forecast_report.md").write_text("\n".join(md))

    return summary


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))
