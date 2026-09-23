# Architecture

```
┌────────────────────────┐
│  generate_data.py         │
│  - cash_flow_history.csv    │  (180 days, model TRAINS on this)
│  - actual_future.csv          │  (30 days, held out, EVAL ONLY)
│  - known_schedule.csv          │  (deterministic outflows for the
│                                    forecast window)
└────────────┬──────────────┘
             ▼
┌──────────────────────────────────────────────────┐
│  CashFlowForecaster.fit(history)                     │
│  (forecast_agent.py)                                    │
│                                                            │
│  - LinearRegression: inflow ~ day_index + weekday dummies  │
│  - LinearRegression: operational_outflow ~ same features     │
│  - Residual std stored from EACH model (for confidence bands) │
└────────────┬─────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────┐
│  .forecast(dates, known_schedule, opening_balance)    │
│                                                          │
│  For each of the 30 forecast days:                        │
│    predicted_inflow      <- statistical model               │
│    predicted_op_outflow  <- statistical model                 │
│    known_outflow          <- looked up directly from            │
│                               known_schedule.csv (NOT predicted)  │
│    net_flow = inflow − op_outflow − known_outflow                  │
│    balance += net_flow                                               │
│    confidence band widens with sqrt(days elapsed)                     │
│    flag if lower bound < CASH_RISK_THRESHOLD                            │
└────────────┬─────────────────────────────────────────┘
             │
   ┌─────────┴──────────┐
   ▼                      ▼
┌─────────────────┐   ┌──────────────────────────┐
│ .evaluate_against_ │   │ daily_forecast.csv /         │
│ actuals(actual_     │   │ audit_log.jsonl                │
│ future.csv)           │   │ (day-by-day forecast, bounds,   │
│ -> MAE, MAPE, bias,    │   │  flags with reasons)             │
│    band coverage %      │   └──────────────────────────┘
└─────────────────┘
             ▲
             │ exposed over HTTP by
┌────────────┴───────────┐
│  FastAPI app.py            │
│  POST /run-forecast          │
│  GET  /daily-forecast          │
└───────────────────────┘
```

## Design decisions worth calling out in the pitch

- **Hybrid model, not one black box.** Scheduled outflows (payroll, GST,
  rent, vendor payments) are facts a finance team already has, not
  unknowns to be predicted. Feeding them into the forecast as exact
  numbers rather than letting a generic time-series model try to "learn"
  weekly/monthly patterns from noisy history is both more accurate and
  more explainable — you can point to exactly which line in
  `known_schedule.csv` produced a given day's known outflow.

- **Confidence bands that widen with the horizon.** Day-1 forecast error
  and day-30 forecast error are not the same kind of uncertain — errors
  compound. Growing the band with `sqrt(cumulative variance)` instead of
  using a single flat band is what makes the risk-flagging meaningful at
  the far end of the horizon rather than either constantly over-flagging
  near-term days or under-flagging far-term ones.

- **The model is graded on calibration, not just point accuracy.** MAPE
  alone can look great while the confidence bands are meaningless. Reporting
  the % of actual days that fell inside the model's own band (63.3% here,
  against an ~80% design target) surfaces that the bands are currently a
  bit too tight — an honest self-assessment that a "look, low error!"
  headline number alone would hide. This is exactly the kind of thing a
  reviewer will ask about, so it's better to have already found and
  reported it.

- **Held-out evaluation is structural, not optional.** `actual_future.csv`
  is generated with the same process as history but independent noise —
  the model architecturally cannot see it during `.fit()`. This is what
  makes the accuracy numbers trustworthy rather than an in-sample fit
  score dressed up as forecast accuracy.

- **Bounded scope: forecast and flag, never act.** The agent has no path
  to delay a payment, reject a scheduled outflow, or move money based on
  its own forecast — a cash-risk flag is a prompt for a human decision,
  not an automated one, which matters most exactly on the days the
  forecast is least certain.
