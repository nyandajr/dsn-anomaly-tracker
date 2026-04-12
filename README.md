# DSN Anomaly Tracker

DSN Anomaly Tracker polls NASA's live DSN XML feed every 10 minutes, engineers telemetry features, scores anomalies with a statistical z-score baseline per spacecraft, and generates a monitoring chart.

## What It Does

- Fetches live DSN telemetry from the NASA feed.
- Extracts per-antenna contact details and normalizes numeric fields.
- Engineers features (`is_active`, `rate_log`, elevation band, complex encoding).
- Scores anomalies using per-spacecraft baseline mean/std and z-score thresholding.
- Appends each run to `data/history.csv`.
- Renders `outputs/signal_report.png` for recent signal and anomaly trends.
- Runs automatically on GitHub Actions every 10 minutes and commits artifacts.

## Run Locally

1. Create and activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python main.py
```

4. Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard provides an operations-console view of the latest DSN snapshot, anomaly activity, complex status, spacecraft drill-downs, and pipeline health using the locally stored `data/history.csv` and `models/baselines.json` files.

## Output Chart

The report in `outputs/signal_report.png` contains two panels over the latest ~24 hours (last 144 rows):

- Top panel: downlink rate line chart split by spacecraft.
- Bottom panel: z-score scatter plot where red points are flagged anomalies and steel-blue points are normal.
