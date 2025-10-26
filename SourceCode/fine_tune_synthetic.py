import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------
# Load synthetic data
# ------------------------------
data = pd.read_csv(r"D:\rice-yield-project\data\synthetic_rice_yield_data.csv")
print("Columns in CSV:", data.columns.tolist())

# ------------------------------
# Define feature columns
# ------------------------------
ndvi_cols = [f"ndvi_{i}" for i in range(1, 13)]
temp_cols = [f"temp_{i}" for i in range(1, 13)]
rain_cols = [f"rain_{i}" for i in range(1, 13)]
rh_cols = [f"rh_{i}" for i in range(1, 13)]
variety_col = ["variety_id"]
target_col = ["yield"]

# ------------------------------
# Split features
# ------------------------------
X_env = data[ndvi_cols + temp_cols + rain_cols + rh_cols].values
X_variety = data[variety_col].values
y = data[target_col].values

# ------------------------------
# Scale environmental features
# ------------------------------
scaler = StandardScaler()
X_env = scaler.fit_transform(X_env)

# ------------------------------
# Encode variety_id as one-hot
# ------------------------------
encoder = OneHotEncoder(sparse=False)
X_variety = encoder.fit_transform(X_variety)

# ------------------------------
# Load existing model
# ------------------------------
model_path = r"D:\rice-yield-project\models\lstm_rice_yield_model.h5"
try:
    model = load_model(model_path, compile=True)
    print("Model loaded and ready for fine-tuning!")
except Exception as e:
    print("Error loading model:", e)
    exit()

# ------------------------------
# Compile model (just in case)
# ------------------------------
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ------------------------------
# Callbacks
# ------------------------------
checkpoint = ModelCheckpoint(
    r"D:\rice-yield-project\models\lstm_rice_yield_model_finetuned.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# ------------------------------
# Fine-tune model
# ------------------------------
history = model.fit(
    [X_env, X_variety],  # <- important: pass as list for 2-input model
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)

print("Fine-tuning completed!")
