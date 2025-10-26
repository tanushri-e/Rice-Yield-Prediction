# utils.py
"""
Helper utilities for the Rice Yield project.

Functions:
- parse_ndvi_csv: parse uploaded NDVI CSV (wide or tall) into a 1D NDVI array
- ensure_length: pad/trim sequence to TIME_STEPS
- merge_ndvi_climate_to_sequence: build (T, F) sequence from ndvi + weekly climate df
- build_sequences_from_dataframe: build X_seq, var_ids, y arrays from merged wide DataFrame
- encode_varieties / decode_varieties: label encode varieties and save/load encoder
- save_artifacts / load_artifacts: save/load scaler, labelencoder, model artifacts
- permutation_importance: approximate feature importance by permuting features
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from typing import Tuple, List, Dict

# -------------------------
# Parsing / sequence helpers
# -------------------------
def parse_ndvi_csv(csv_path_or_buffer, time_steps: int = 12) -> np.ndarray:
    """
    Parse NDVI CSV (wide or tall) and return a 1D numpy array of length `time_steps`.
    Accepts:
      - Wide: a single-row CSV containing columns ndvi_1..ndvi_T
      - Tall: columns 'week' and 'ndvi' (sorted by week)
    """
    df = pd.read_csv(csv_path_or_buffer)
    # Wide format
    wide_cols = [f'ndvi_{i+1}' for i in range(time_steps)]
    if all(c in df.columns for c in wide_cols):
        row = df.iloc[0]
        arr = np.array([float(row[c]) for c in wide_cols])
        return ensure_length(arr, time_steps)
    # Tall format (case-insensitive)
    cols_lower = [c.lower() for c in df.columns]
    if 'week' in cols_lower and 'ndvi' in cols_lower:
        df.columns = [c.lower() for c in df.columns]
        df_sorted = df.sort_values('week').reset_index(drop=True)
        arr = df_sorted['ndvi'].astype(float).values
        return ensure_length(arr, time_steps)
    raise ValueError("NDVI CSV format not recognized. Provide wide (ndvi_1..ndvi_T) or tall (week, ndvi) format.")

def ensure_length(arr: np.ndarray, time_steps: int) -> np.ndarray:
    """
    Pad with last value or trim to ensure `arr` has length `time_steps`.
    """
    arr = np.array(arr, dtype=float)
    if arr.size == time_steps:
        return arr
    if arr.size < time_steps:
        if arr.size == 0:
            return np.zeros(time_steps, dtype=float)
        pad = np.repeat(arr[-1], time_steps - arr.size)
        return np.concatenate([arr, pad])
    # arr.size > time_steps: trim the end
    return arr[:time_steps]

def merge_ndvi_climate_to_sequence(ndvi_seq: np.ndarray, weekly_climate_df: pd.DataFrame,
                                   time_steps: int = 12) -> np.ndarray:
    """
    Build a (time_steps, features) numpy array where features order is:
      [ndvi, temp_mean_C, rainfall_mm, rh_mean_pct]
    weekly_climate_df should have those columns or similar names.
    If columns missing, zeros are used for that feature.
    """
    ndvi_seq = ensure_length(ndvi_seq, time_steps)
    df = weekly_climate_df.reset_index(drop=True).copy()
    # pad/trim weekly df to time_steps
    if df.shape[0] < time_steps:
        last = df.iloc[-1].to_dict() if df.shape[0]>0 else {}
        for _ in range(time_steps - df.shape[0]):
            df = df.append(last, ignore_index=True)
    if df.shape[0] > time_steps:
        df = df.iloc[:time_steps]
    # safe column extraction
    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
    rain_col = next((c for c in df.columns if 'rain' in c.lower()), None)
    rh_col = next((c for c in df.columns if 'rh' in c.lower() or 'hum' in c.lower()), None)
    temp = df[temp_col].astype(float).values if temp_col is not None else np.zeros(time_steps)
    rain = df[rain_col].astype(float).values if rain_col is not None else np.zeros(time_steps)
    rh = df[rh_col].astype(float).values if rh_col is not None else np.zeros(time_steps)
    seq = np.stack([ndvi_seq, temp, rain, rh], axis=-1)  # shape (time_steps, 4)
    return seq

# -------------------------
# Building training arrays
# -------------------------
def build_sequences_from_dataframe(df_wide: pd.DataFrame, time_steps: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: df_wide where each row is one sample and columns:
      ndvi_1..ndvi_T, temp_1..temp_T, rain_1..rain_T, hum_1..hum_T, variety, yield
    Output:
      X_seq: (n_samples, time_steps, 4)
      var_ids: (n_samples,) integer labels
      y: (n_samples,) float targets
    """
    seq_cols = []
    for t in range(time_steps):
        seq_cols.append((f'ndvi_{t+1}', f'temp_{t+1}', f'rain_{t+1}', f'hum_{t+1}'))
    X = []
    for _, row in df_wide.iterrows():
        seq = []
        for (ndc, tc, rc, hc) in seq_cols:
            nd = float(row.get(ndc, 0.0))
            te = float(row.get(tc, 0.0))
            ra = float(row.get(rc, 0.0))
            hu = float(row.get(hc, 0.0))
            seq.append([nd, te, ra, hu])
        X.append(seq)
    X = np.array(X, dtype=float)
    # variety -> label encode
    if 'variety' in df_wide.columns:
        le = LabelEncoder()
        var_ids = le.fit_transform(df_wide['variety'].astype(str).values)
    elif 'var_id' in df_wide.columns:
        var_ids = df_wide['var_id'].astype(int).values
        le = None
    else:
        var_ids = np.zeros(df_wide.shape[0], dtype=int)
        le = None
    y = df_wide['yield'].astype(float).values if 'yield' in df_wide.columns else np.zeros(df_wide.shape[0], dtype=float)
    return X, var_ids, y

# -------------------------
# Encoding / artifacts
# -------------------------
def encode_varieties(variety_list: List[str]) -> LabelEncoder:
    """
    Fit and return a LabelEncoder on provided list of variety names.
    """
    le = LabelEncoder()
    le.fit(variety_list)
    return le

def save_artifacts(scaler: StandardScaler = None, labelencoder: LabelEncoder = None, folder: str = "models"):
    """
    Save scaler and labelencoder to folder as scaler.pkl and labelencoder.pkl
    """
    os.makedirs(folder, exist_ok=True)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    if labelencoder is not None:
        joblib.dump(labelencoder, os.path.join(folder, "labelencoder.pkl"))

def load_artifacts(folder: str = "models") -> Tuple[StandardScaler, LabelEncoder]:
    """
    Load scaler and labelencoder from models folder. Returns (scaler, labelencoder)
    If not found, returns (None, None).
    """
    scaler_path = os.path.join(folder, "scaler.pkl")
    le_path = os.path.join(folder, "labelencoder.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    le = joblib.load(le_path) if os.path.exists(le_path) else None
    return scaler, le

# -------------------------
# Permutation importance (simple)
# -------------------------
def permutation_importance(model, X_seq: np.ndarray, X_var: np.ndarray, y_true: np.ndarray = None,
                           n_repeats: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    Approximate feature importance by permuting each feature across time steps.
    - model must accept dict inputs: {'sequence_input': X_seq, 'var_input': X_var} or similar.
    - X_seq: (n_samples, T, F)
    - X_var: (n_samples,) integer array for variety inputs
    Returns dict mapping feature names to average absolute change in MAE (or prediction).
    NOTE: For single-sample demo, pass n_samples=1 arrays and y_true=None.
    """
    rng = np.random.RandomState(random_state)
    n_samples, T, F = X_seq.shape
    # baseline predictions
    try:
        base_pred = model.predict({'sequence_input': X_seq, 'var_input': X_var}).ravel()
    except Exception:
        # fallback common naming if your model inputs have different names
        base_pred = model.predict([X_seq, X_var]).ravel()
    if y_true is not None:
        from sklearn.metrics import mean_absolute_error
        base_error = mean_absolute_error(y_true, base_pred)
    else:
        base_error = None
    importance = {}
    feat_labels = ['NDVI', 'Temp', 'Rain', 'Hum'][:F]
    for f in range(F):
        diffs = []
        for _ in range(n_repeats):
            X_perm = X_seq.copy()
            # permute the feature values across time within each sample
            for i in range(n_samples):
                X_perm[i,:,f] = rng.permutation(X_perm[i,:,f])
            try:
                p = model.predict({'sequence_input': X_perm, 'var_input': X_var}).ravel()
            except Exception:
                p = model.predict([X_perm, X_var]).ravel()
            if y_true is not None:
                from sklearn.metrics import mean_absolute_error
                err = mean_absolute_error(y_true, p)
                diffs.append(err - base_error)
            else:
                # use mean absolute change in prediction
                diffs.append(np.mean(np.abs(p - base_pred)))
        importance[feat_labels[f]] = float(np.mean(diffs))
    return importance

# -------------------------
# Small utility: build single-sample input for prediction
# -------------------------
def prepare_single_input(ndvi_seq: np.ndarray, weekly_climate_df: pd.DataFrame,
                         scaler: StandardScaler, labelencoder: LabelEncoder,
                         time_steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_seq_scaled: shape (1, time_steps, 4)
      var_id_arr: shape (1,)
    Expects labelencoder to map variety names to ids. If None, var_id = 0.
    """
    # if weekly_climate_df has no 'variety' column, user must pass variety separately
    if 'variety' in weekly_climate_df.columns:
        variety = weekly_climate_df['variety'].iloc[0]
    else:
        variety = None
    seq = merge_ndvi_climate_to_sequence(ndvi_seq, weekly_climate_df, time_steps=time_steps)
    # scale
    flat = seq.reshape(-1, seq.shape[-1])
    scaled_flat = scaler.transform(flat)
    scaled_seq = scaled_flat.reshape(1, time_steps, seq.shape[-1])
    # variety id
    if labelencoder is not None and variety is not None and variety in labelencoder.classes_:
        var_id = int(np.where(labelencoder.classes_ == variety)[0][0])
    else:
        var_id = 0
    return scaled_seq, np.array([var_id])

# -------------------------
# If run as script, simple demo
# -------------------------
if __name__ == "__main__":
    # quick sanity check
    print("utils.py helper module. Run functions from your main scripts.")
