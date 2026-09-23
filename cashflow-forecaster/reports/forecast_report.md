# Cash-Flow Forecast Report

Run: 2026-08-27T06:36:16.183107+00:00

- Forecast horizon: 30 days
- Opening balance: Rs 1,244,290.11
- Cash risk threshold: Rs 1,150,000.00
- **Days flagged as cash-risk: 5** (2026-07-01, 2026-07-02, 2026-07-03, 2026-07-04, 2026-07-05)

## Accuracy against held-out actuals (never seen during training)
- Mean Absolute Error (MAE): Rs 12,555.45
- Mean Absolute Percentage Error (MAPE): 1.03%
- Bias (forecast − actual, average): Rs 10,865.26
- Final-day forecast balance: Rs 1,496,478.84 vs actual Rs 1,492,678.37
- % of days the actual balance fell within the forecast's confidence band: 63.3%

See `daily_forecast.csv` for the full day-by-day forecast, including which known scheduled
outflows (payroll/GST/rent/vendor) were factored in deterministically for each day.