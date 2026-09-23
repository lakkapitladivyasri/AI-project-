"""
generate_data.py
-----------------
Generates synthetic daily cash-flow data for the AI Finance Controller
track (Cash-flow forecaster direction):

  1. data/cash_flow_history.csv  -- 180 days of "known" history the
     forecaster trains on.
  2. data/actual_future.csv       -- the NEXT 30 days, generated with the
     same underlying process but held out and NEVER shown to the
     forecaster at training time. This is what lets forecast accuracy be
     measured honestly against real held-out outcomes instead of just
     eyeballing a chart.
  3. data/known_schedule.csv       -- deterministic, already-known
     scheduled outflows for the forecast window (payroll, tax, rent) —
     the kind of thing a finance team already knows for certain and
     which a forecaster should use directly rather than trying to predict
     statistically.

Cash-flow composition modeled:
  - Organic daily settlement inflow: weekly seasonality (weekdays higher
    than weekends) + a mild upward growth trend + noise.
  - Small daily operational outflows (noisy, no strong pattern).
  - Scheduled outflows: payroll (1st & 16th), vendor payments (every
    Friday), GST/tax payment (20th of the month), rent (1st of the month).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(23)

HISTORY_DAYS = 180
FORECAST_DAYS = 30
START_DATE = pd.Timestamp("2026-01-01")
OPENING_BALANCE_DAY0 = 500_000.0

PAYROLL_AMOUNT = 180_000.0
VENDOR_BASE_AMOUNT = 45_000.0
GST_AMOUNT = 95_000.0
RENT_AMOUNT = 60_000.0


def organic_inflow(day_index: int, date: pd.Timestamp) -> float:
    weekday = date.dayofweek  # 0=Mon ... 6=Sun
    weekend_factor = 0.55 if weekday >= 5 else 1.0
    trend = 1.0 + 0.0015 * day_index          # mild organic growth over time
    base = 32_000 * weekend_factor * trend
    noise = RNG.normal(0, 4_500)
    return max(0.0, base + noise)


def operational_outflow(date: pd.Timestamp) -> float:
    weekday = date.dayofweek
    weekend_factor = 0.4 if weekday >= 5 else 1.0
    base = 6_000 * weekend_factor
    noise = RNG.normal(0, 1_200)
    return max(0.0, base + noise)


def scheduled_outflow(date: pd.Timestamp) -> tuple[float, str]:
    """Returns (amount, label) for any deterministic scheduled outflow on this date."""
    d = date.day
    if d in (1, 16):
        return PAYROLL_AMOUNT + RNG.normal(0, 2_000), "payroll"
    if date.dayofweek == 4:  # Friday
        return VENDOR_BASE_AMOUNT + RNG.normal(0, 6_000), "vendor_payment"
    if d == 20:
        return GST_AMOUNT + RNG.normal(0, 1_500), "gst_payment"
    return 0.0, ""


def build_series(start_date: pd.Timestamp, n_days: int, day_offset: int, opening_balance: float):
    rows = []
    balance = opening_balance
    for i in range(n_days):
        date = start_date + pd.Timedelta(days=i)
        day_index = day_offset + i

        inflow = round(organic_inflow(day_index, date), 2)
        op_out = round(operational_outflow(date), 2)
        sched_amt, sched_label = scheduled_outflow(date)
        sched_amt = round(max(0.0, sched_amt), 2)

        # rent separately, always 1st of month, additive to whatever scheduled_outflow returned
        extra_rent = 0.0
        if date.day == 1:
            extra_rent = round(RENT_AMOUNT + RNG.normal(0, 1000), 2)

        total_outflow = round(op_out + sched_amt + extra_rent, 2)
        net = round(inflow - total_outflow, 2)
        opening = round(balance, 2)
        balance = round(balance + net, 2)

        rows.append({
            "date": date.date().isoformat(),
            "opening_balance": opening,
            "inflow": inflow,
            "operational_outflow": op_out,
            "scheduled_outflow": sched_amt,
            "scheduled_label": sched_label if sched_amt > 0 else ("rent" if extra_rent > 0 else ""),
            "rent_outflow": extra_rent,
            "total_outflow": total_outflow,
            "net_cash_flow": net,
            "closing_balance": balance,
        })
    return pd.DataFrame(rows), balance


def main():
    history_df, balance_after_history = build_series(START_DATE, HISTORY_DAYS, 0, OPENING_BALANCE_DAY0)
    future_start = START_DATE + pd.Timedelta(days=HISTORY_DAYS)
    actual_future_df, _ = build_series(future_start, FORECAST_DAYS, HISTORY_DAYS, balance_after_history)

    # known_schedule.csv: the deterministic part of the future window ONLY
    # (what a finance team would already know for certain going into the
    # forecast — no organic inflow, no noise-driven operational spend).
    known_rows = []
    for i in range(FORECAST_DAYS):
        date = future_start + pd.Timedelta(days=i)
        sched_amt, sched_label = scheduled_outflow(date)
        rent = RENT_AMOUNT if date.day == 1 else 0.0
        total_known = round(max(0.0, sched_amt) + rent, 2)
        label = sched_label if sched_amt > 0 else ("rent" if rent > 0 else "")
        known_rows.append({"date": date.date().isoformat(), "known_outflow": total_known, "label": label})
    known_df = pd.DataFrame(known_rows)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    history_df.to_csv(out_dir / "cash_flow_history.csv", index=False)
    actual_future_df.to_csv(out_dir / "actual_future.csv", index=False)
    known_df.to_csv(out_dir / "known_schedule.csv", index=False)

    print(f"History: {len(history_df)} days -> {out_dir/'cash_flow_history.csv'}")
    print(f"Held-out actual future: {len(actual_future_df)} days -> {out_dir/'actual_future.csv'}")
    print(f"Known scheduled outflows for forecast window -> {out_dir/'known_schedule.csv'}")
    print(f"\nOpening balance day 0: Rs {OPENING_BALANCE_DAY0:,.2f}")
    print(f"Balance at end of history (day {HISTORY_DAYS}): Rs {balance_after_history:,.2f}")
    print(f"Actual balance at end of forecast window (ground truth): Rs {actual_future_df['closing_balance'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
