# train.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Replace this loader with your merged dataset
# The merged dataset must have columns:
# ndvi_1..ndvi_T, temp_1..temp_T, rain_1..rain_T, hum_1..hum_T, variety, yield
# For demo, we'll load synthetic CSV or provide your real training CSV path here
DATA_CSV = "training_ready.csv"  # replace with your merged CSV
TIME_STEPS = 12
N_FEATURES = 4  # ndvi,temp,rain,hum
EMBED_DIM = 4

# ---------- Load dataset ----------
df = pd.read_csv(DATA_CSV)
# encode variety
le = LabelEncoder()
df['var_id'] = le.fit_transform(df['variety'])
n_varieties = len(le.classes_)
print("Varieties:", list(le.classes_))

# build X_seq, var, y
def build_sequences(df):
    X = []
    for idx, row in df.iterrows():
        seq = []
        for t in range(TIME_STEPS):
            seq.append([
                row[f'ndvi_{t+1}'],
                row[f'temp_{t+1}'],
                row[f'rain_{t+1}'],
                row[f'hum_{t+1}']
            ])
        X.append(seq)
    return np.array(X)

X = build_sequences(df)
y = df['yield'].values
var = df['var_id'].values

# Scale features (fit on flattened training)
nsamples = X.shape[0]
scaler = StandardScaler()
X_flat = X.reshape(-1, N_FEATURES)
scaler.fit(X_flat)
X_scaled = scaler.transform(X_flat).reshape(nsamples, TIME_STEPS, N_FEATURES)

# split
X_train, X_test, var_train, var_test, y_train, y_test = train_test_split(
    X_scaled, var, y, test_size=0.2, random_state=42
)

# model
seq_input = Input(shape=(TIME_STEPS, N_FEATURES), name='seq_in')
x = LSTM(64, return_sequences=True)(seq_input)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.15)(x)

var_input = Input(shape=(1,), dtype='int32', name='var_in')
var_embed = Embedding(input_dim=n_varieties, output_dim=EMBED_DIM)(var_input)
var_flat = Flatten()(var_embed)

conc = Concatenate()([x, var_flat])
d = Dense(32, activation='relu')(conc)
d = Dropout(0.1)(d)
d = Dense(16, activation='relu')(d)
out = Dense(1, activation='linear')(d)

model = Model([seq_input, var_input], out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model.fit([X_train, var_train], y_train, validation_split=0.1,
          epochs=80, batch_size=32, callbacks=[es], verbose=2)

# save
model.save("models/lstm_rice_yield_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/labelencoder.pkl")
print("Saved model & scaler & labelencoder to models/")
