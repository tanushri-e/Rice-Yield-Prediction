# preprocess.py
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------------
# NASA POWER DATA FUNCTIONS
# -----------------------------
def fetch_nasa_power(lat, lon, start_date, end_date, out_csv="climate_daily.csv"):
    """
    Fetches daily T2M, RH2M, PRECTOTCORR from NASA POWER API and saves CSV.
    Dates: YYYYMMDD
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M,RH2M,PRECTOTCORR"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=CSV"
    )
    r = requests.get(url)
    r.raise_for_status()
    Path(out_csv).write_bytes(r.content)

    # skip metadata header rows
    df = pd.read_csv(out_csv, skiprows=12)
    df.columns = [c.strip() for c in df.columns]

    # parse date
    if 'YYYYMMDD' in df.columns:
        df['date'] = pd.to_datetime(df['YYYYMMDD'], format="%Y%m%d")
    elif 'DATE' in df.columns:
        df['date'] = pd.to_datetime(df['DATE'])
    else:
        df['date'] = pd.to_datetime(df.iloc[:,0])
    
    df = df.set_index('date').sort_index()
    return df

def daily_to_weekly(df, precip_col='PRECTOTCORR', temp_col='T2M', rh_col='RH2M', week_freq='W-MON'):
    w = pd.DataFrame()
    if precip_col in df.columns:
        w['rainfall_mm'] = df[precip_col].resample(week_freq).sum()
    if temp_col in df.columns:
        w['temp_mean_C'] = df[temp_col].resample(week_freq).mean()
    if rh_col in df.columns:
        w['rh_mean_pct'] = df[rh_col].resample(week_freq).mean()
    w = w.dropna(how='all')
    return w

# -----------------------------
# NDVI FUNCTIONS
# -----------------------------
def ndvi_to_wide(csv_path, time_steps=12, week_freq='W-MON', single_roi=True):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    if single_roi:
        df = df.sort_values('date').set_index('date')
        daily = df['NDVI'].resample('D').mean().dropna()
        weekly = daily.resample(week_freq).mean().dropna()
        vals = weekly.values
        if vals.size < time_steps:
            pad = [float(vals[-1])]*(time_steps - vals.size) if vals.size > 0 else [0.0]*time_steps
            vals = list(vals) + pad
        else:
            vals = list(vals[:time_steps])
        row = {f'ndvi_{i+1}': float(vals[i]) for i in range(time_steps)}
        wide = pd.DataFrame([row])
    else:
        # Multiple points / features
        id_col = 'system:index' if 'system:index' in df.columns else ('id' if 'id' in df.columns else None)
        if id_col is None:
            raise ValueError("No feature id column found. Expected 'system:index' or 'id'.")
        df = df[[id_col, 'NDVI', 'date']].dropna(subset=['NDVI','date'])
        rows = []
        for fid, group in df.groupby(id_col):
            g = group.set_index('date').sort_index()
            daily = g['NDVI'].resample('D').mean()
            weekly = daily.resample(week_freq).mean().dropna()
            vals = weekly.values
            if vals.size < time_steps:
                pad = [float(vals[-1])]*(time_steps - vals.size) if vals.size > 0 else [0.0]*time_steps
                vals = list(vals) + pad
            else:
                vals = list(vals[:time_steps])
            row = {'feature_id': fid}
            row.update({f'ndvi_{i+1}': float(vals[i]) for i in range(time_steps)})
            rows.append(row)
        wide = pd.DataFrame(rows)
    return wide

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # --- 1. Fetch & preprocess climate data ---
    lat, lon = 10.7867, 79.1378  # Thanjavur coordinates
    start, end = "20230601", "20231031"
    daily_climate = fetch_nasa_power(lat, lon, start, end, out_csv="data/climate_daily_thanjavur.csv")
    weekly_climate = daily_to_weekly(daily_climate)
    weekly_climate.to_csv("data/weekly_climate_thanjavur.csv")
    print("Saved weekly_climate_thanjavur.csv")

    # --- 2. Convert NDVI CSV to wide ---
    ndvi_wide = ndvi_to_wide("data/sample_ndvi_wide.csv", time_steps=12, week_freq='W-MON', single_roi=True)
    ndvi_wide.to_csv("data/ndvi_wide.csv", index=False)
    print("Saved ndvi_wide.csv")

    # --- 3. Merge NDVI + weekly climate for training ---
    # ensure same number of time-steps
    weekly_climate_12 = weekly_climate.head(12).reset_index(drop=True)
    training_ready = pd.concat([ndvi_wide.reset_index(drop=True), weekly_climate_12], axis=1)
    training_ready.to_csv("data/training_ready.csv", index=False)
    print("Saved training_ready.csv")
