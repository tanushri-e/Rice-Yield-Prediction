# app.py
# Streamlit app: Rice Yield Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# map & geocoding
from streamlit_folium import st_folium
import folium
import requests

# optional Earth Engine import
try:
    import ee
    EE_AVAILABLE = True
except Exception:
    EE_AVAILABLE = False

st.set_page_config(page_title="Rice Yield Predictor", layout="centered")
st.title("Rice Yield Predictor")

TIME_STEPS = 12
N_FEATURES = 4
MODEL_PATH = "models/lstm_rice_yield_model.h5"
SCALER_PATH = "models/scaler.pkl"
LE_PATH = "models/labelencoder.pkl"

# load model & artifacts
model = None
scaler = None
labelenc = None
if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None
if os.path.exists(LE_PATH):
    try:
        labelenc = joblib.load(LE_PATH)
    except Exception:
        labelenc = None

rice_varieties = [
    # Traditional
    "Kullakar", "Mapillai Samba", "Seeraga Samba", "Karuppu Kavuni", "Iluppai Poo Samba",
    "Thooyamalli", "Kaatuyaanam", "Poongar", "Kichili Samba", "Samba Mosanam",
    "Navara", "Mappillai Samba Red Rice", "Neelam Samba", "Vaigunda Samba",
    # Modern / Released
    "ADT-36", "ADT-37", "ADT-38", "ADT-39", "ADT-43", "ADT-45", "ADT-46", "ADT-47",
    "ADT-48", "ADT-49", "ADT-50", "ADT-51",
    "CO-43", "CO-45", "CO-50", "CO-51",
    "ASD-16", "ASD-18", "ASD-19", "ASD-20",
    "TKM-9", "TKM-12",
    "Ponmani", "White Ponni", "BPT-5204 (Samba Mahsuri)"
]

# -------------------- Helpers --------------------

def geocode_address(address):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        r = requests.get(url, params=params, headers={"User-Agent": "rice-yield-app/1.0"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if len(data) == 0:
            return None
        d = data[0]
        return float(d['lat']), float(d['lon'])
    except Exception:
        return None

def fetch_nasa_power(lat, lon, start_date, end_date):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M,RH2M,PRECTOTCORR"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=CSV"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text), skiprows=12)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df[['YEAR','MO','DY']])
    df = df.set_index('date').sort_index()
    return df

def fetch_era5_open_meteo(lat, lon, start_date, end_date):
    s = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
    e = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}&start_date={s}&end_date={e}"
        "&daily=temperature_2m_mean,precipitation_sum&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
        daily = js['daily']
        dates = pd.to_datetime(daily['time'])
        df = pd.DataFrame({
            'date': dates,
            'T2M': daily.get('temperature_2m_mean', [np.nan]*len(dates)),
            'PRECTOTCORR': daily.get('precipitation_sum', [np.nan]*len(dates))
        })
        df = df.set_index('date').sort_index()
        return df
    except Exception as e:
        raise RuntimeError(f"Open-Meteo ERA5 fetch failed: {e}")

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

def fetch_ndvi_gee_point(lat, lon, start_date, end_date, time_steps=TIME_STEPS):
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine Python API not available.")
    try:
        try: ee.Initialize()
        except Exception: ee.Authenticate(); ee.Initialize()
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(ee.Date(start_date), ee.Date(end_date)) \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
        def add_ndvi(img): return img.addBands(img.normalizedDifference(['B8','B4']).rename('NDVI'))
        s2 = s2.map(add_ndvi).select('NDVI')
        sd = datetime.strptime(start_date, '%Y%m%d'); ed = datetime.strptime(end_date, '%Y%m%d')
        total_days = (ed - sd).days; days_per_step = max(1, total_days // time_steps)
        vals = []
        for i in range(time_steps):
            wstart = sd + pd.Timedelta(days=i*days_per_step)
            wend = wstart + pd.Timedelta(days=days_per_step)
            coll = s2.filterDate(ee.Date(wstart.strftime('%Y-%m-%d')), ee.Date(wend.strftime('%Y-%m-%d')))
            mean_img = coll.median()
            ndvi_val = mean_img.reduceRegion(ee.Reducer.mean(), ee.Geometry.Point(lon, lat), 500).get('NDVI')
            vals.append(ndvi_val)
        vals_client = [v.getInfo() if hasattr(v,'getInfo') else None for v in vals]
        vals_num = [float(v) if v is not None else np.nan for v in vals_client]
        arr = np.array(vals_num, dtype=float)
        if np.isnan(arr).any(): arr = np.nan_to_num(arr, nan=0.0)
        return arr
    except Exception as e:
        raise RuntimeError(f"Error fetching NDVI from GEE: {e}")

# -------------------- UI --------------------

# Location input
st.subheader("Input Location")
loc_option = st.radio("", ["Pick On Map", "Enter Address / Place"], horizontal=True)

# Initialize session state for lat/lon
if 'selected_lat' not in st.session_state:
    st.session_state['selected_lat'] = None
if 'selected_lon' not in st.session_state:
    st.session_state['selected_lon'] = None

if loc_option == "Pick On Map":
    m = folium.Map(location=[10.7867,79.1378], zoom_start=11, width='100%', height=600)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=600, returned_objects=["last_clicked"])
    if map_data and map_data.get('last_clicked'):
        clicked = map_data['last_clicked']
        st.session_state['selected_lat'] = clicked['lat']
        st.session_state['selected_lon'] = clicked['lng']
        st.success(f"Selected location: {st.session_state['selected_lat']:.6f}, {st.session_state['selected_lon']:.6f}")
    else:
        st.info("No Pin Detected")
else:
    addr = st.text_input("Address or place name (e.g., 'Thanjavur, India'):", value="")
    if st.button("Geocode address"):
        if addr.strip() == "":
            st.error("Please enter an address.")
        else:
            res = geocode_address(addr)
            if res is None:
                st.error("Could not geocode the address. Try a precise text.")
            else:
                st.session_state['selected_lat'], st.session_state['selected_lon'] = res
                st.success(f"Geocoded to: {st.session_state['selected_lat']:.6f}, {st.session_state['selected_lon']:.6f}")

selected_lat = st.session_state['selected_lat']
selected_lon = st.session_state['selected_lon']

# Season dates
st.subheader("Season Dates")
col1, col2 = st.columns([1,1])
with col1:
    start_date = st.date_input("Start date")
with col2:
    end_date = st.date_input("End date")

# Rice Variety
st.subheader("Rice Variety")
var_choice = st.selectbox("Select the rice variety", [""] + rice_varieties, index=0)

# NDVI
st.markdown("---")
st.subheader("NDVI Data Source")
ndvi_source = st.radio("", ["Use example NDVI time-series (demo)", "Fetch NDVI from Google Earth Engine (GEE)"])

# Predict
if st.button("Predict Yield"):
    if selected_lat is None or selected_lon is None:
        st.error("Error: No location selected.")
        st.stop()
    if var_choice == "":
        st.error("Please select a rice variety.")
        st.stop()
    if start_date is None or end_date is None:
        st.error("Start and End dates are required.")
        st.stop()
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    s_str = start_date.strftime('%Y%m%d'); e_str = end_date.strftime('%Y%m%d')

    # NDVI
    if ndvi_source == "Use example NDVI time-series (demo)":
        weeks = np.arange(TIME_STEPS)
        ndvi_vals = 0.25 + 0.45 * np.exp(-((weeks - TIME_STEPS/2)**2)/(2*(TIME_STEPS/4)**2))
        ndvi_vals = ndvi_vals / ndvi_vals.max() * 0.7
        ndvi_seq = ndvi_vals + np.random.normal(0,0.02, size=TIME_STEPS)
    else:
        if not EE_AVAILABLE:
            st.error("Earth Engine API not available.")
            st.stop()
        try: 
            ndvi_seq = fetch_ndvi_gee_point(selected_lat, selected_lon, s_str, e_str, TIME_STEPS)
            st.success("Fetched NDVI from GEE.")
        except Exception: 
            st.info("GEE NDVI fetch failed; using example NDVI fallback.")
            weeks = np.arange(TIME_STEPS)
            ndvi_vals = 0.25 + 0.45 * np.exp(-((weeks - TIME_STEPS/2)**2)/(2*(TIME_STEPS/4)**2))
            ndvi_vals = ndvi_vals / ndvi_vals.max() * 0.7
            ndvi_seq = ndvi_vals + np.random.normal(0,0.02, size=TIME_STEPS)

    # Climate
    weekly_climate = None
    try:
        daily = fetch_nasa_power(selected_lat, selected_lon, s_str, e_str)
        weekly_climate = daily_to_weekly(daily).reset_index()
        if weekly_climate.shape[0] == 0: raise RuntimeError('NASA POWER returned no rows')
    except:
        try:
            df_era = fetch_era5_open_meteo(selected_lat, selected_lon, s_str, e_str)
            weekly_climate = daily_to_weekly(df_era, precip_col='precipitation_sum', temp_col='temperature_2m_mean').reset_index()
            if weekly_climate.shape[0] == 0: raise RuntimeError('ERA5 returned no rows')
        except:
            weekly_climate = pd.DataFrame({'temp_mean_C':[28.0]*TIME_STEPS,'rainfall_mm':[10.0]*TIME_STEPS,'rh_mean_pct':[75.0]*TIME_STEPS})

    # pad climate
    if weekly_climate.shape[0]<TIME_STEPS:
        last = weekly_climate.iloc[-1]
        for _ in range(TIME_STEPS-weekly_climate.shape[0]):
            weekly_climate = weekly_climate.append(last,ignore_index=True)
    weekly_climate = weekly_climate.iloc[:TIME_STEPS]

    # build sequence
    seq = np.zeros((TIME_STEPS,N_FEATURES))
    seq[:,0]=ndvi_seq
    seq[:,1]=weekly_climate['temp_mean_C'].values
    seq[:,2]=weekly_climate['rainfall_mm'].values
    seq[:,3]=weekly_climate['rh_mean_pct'].values

    # scale
    if scaler is None:
        flat=seq.reshape(-1,N_FEATURES)
        scaler_local=StandardScaler().fit(flat)
        seq_scaled=scaler_local.transform(flat).reshape(TIME_STEPS,N_FEATURES)
    else:
        seq_scaled=scaler.transform(seq.reshape(-1,N_FEATURES)).reshape(TIME_STEPS,N_FEATURES)

    # variety encoding
    if labelenc is None:
        var_id=rice_varieties.index(var_choice) if var_choice in rice_varieties else 0
    else:
        var_id=int(np.where(labelenc.classes_==var_choice)[0][0]) if var_choice in labelenc.classes_ else 0

    X_seq=np.expand_dims(seq_scaled,axis=0)
    X_var=np.array([var_id])

    # preview
    st.subheader("Input Preview")
    st.write({"lat": round(selected_lat,6), "lon": round(selected_lon,6)})
    st.write("Selected variety:", var_choice)
    st.write("NDVI:", np.round(ndvi_seq,3))
    st.write("Weekly climate (first 3 rows):")
    st.write(weekly_climate.head(3))

    # predict
    if model is None: st.error("Model not found."); st.stop()
    try: pred = model.predict({'seq_in':X_seq,'var_in':X_var}).ravel()[0]
    except Exception as e: st.error(f"Model prediction failed: {e}"); st.stop()

    st.metric("Predicted yield (t/ha)", f"{pred:.2f}")

    # ---------------- Fixed Feature Importance ----------------
    base = pred
    feats = ['NDVI', 'Temp', 'Rain', 'Hum']
    imps = {}

    for f in range(N_FEATURES):
        vals = []
        for _ in range(8):
            perm = X_seq.copy()
            perm[0, :, f] = np.random.permutation(perm[0, :, f])
            try:
                p = model.predict({'seq_in': perm, 'var_in': X_var}).ravel()[0]
                vals.append(abs(p - base))
            except:
                vals.append(0)
        imps[feats[f]] = np.mean(vals)

    # scale for visibility
    imps_scaled = {k: v*100 for k, v in imps.items()}

    importances = pd.Series(imps_scaled).sort_values(ascending=False)
    st.subheader("Feature importance (approx.)")
    st.bar_chart(importances)
    # ---------------------------------------------------------

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # share x-axis
    ax[0].plot(range(1, TIME_STEPS + 1), ndvi_seq, marker='o', color='blue')
    ax[0].set_title('NDVI Time-Series')
    ax[0].set_ylabel('NDVI')
    ax[0].grid(True)

    ax[1].plot(range(1, TIME_STEPS + 1), seq[:, 2], marker='o', color='blue')
    ax[1].set_title('Weekly Rainfall (mm)')
    ax[1].set_ylabel('Rainfall (mm)')
    ax[1].set_xlabel('Week')
    ax[1].grid(True)

    fig.tight_layout()
    st.pyplot(fig)

    out=weekly_climate.copy(); out['ndvi']=ndvi_seq; out['predicted_yield']=pred
    csv=out.to_csv(index=False).encode('utf-8')
    st.download_button("Download result CSV", csv, "prediction_result.csv", "text/csv")
