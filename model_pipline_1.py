"""
Sales Forecasting — ANN Training Pipeline (v2)
================================================
All CSV columns must be lowercase. Expected columns:
  id, date, country, store, product, num_sold, day, month,
  count_of_week_acc_to_year, year, days_num, weekday_or_weekends,
  count_of_week_acc_to_month, holiday(canada), holiday(italy),
  holiday(norway), holiday(finland), holiday(kenya), holiday(singapore)
"""

import os, math, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CAT_COLS = ['country', 'store', 'product', 'day', 'month']

HOLIDAY_COLS = [
    'holiday(canada)', 'holiday(italy)', 'holiday(finland)',
    'holiday(norway)',  'holiday(kenya)', 'holiday(singapore)',
]

TARGET_COL   = 'num_sold'
LOOK_BACK    = 15
ARTEFACT_DIR = 'artefacts'

# FEATURE_COLS is built dynamically after encoding — see build_feature_cols()
FEATURE_COLS = None   # set after first encode


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Clean raw CSV
# ──────────────────────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Force all column names lowercase
    - Convert holiday TRUE/FALSE strings → 1 / 0
    - Convert weekday_or_weekends → 0 / 1
    - Drop useless columns (id)
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Drop id if present
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Holiday columns → int 0/1
    for col in HOLIDAY_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.upper()
                .map(lambda x: 1 if x == 'TRUE' else 0)
                .astype(int)
            )

    # weekday_or_weekends → 0/1
    if 'weekday_or_weekends' in df.columns:
        df['weekday_or_weekends'] = (
            df['weekday_or_weekends'].astype(str).str.strip().str.lower()
            .map({'weekday': 0, 'weekend': 1})
            .fillna(0).astype(int)
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Fit encoders
# ──────────────────────────────────────────────────────────────────────────────

def fit_encoders(df: pd.DataFrame) -> dict:
    """One OHE per categorical column, fitted on full data."""
    encoders = {}
    for col in CAT_COLS:
        enc = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ).set_output(transform='pandas')
        enc.fit(df[[col]])
        encoders[f'ohe_{col}'] = enc
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply pre-fitted OHE encoders to any split."""
    df = df.copy()
    for col in CAT_COLS:
        enc         = encoders[f'ohe_{col}']
        transformed = enc.transform(df[[col]])
        df = pd.concat([df.drop(columns=[col]), transformed], axis=1)
    df.fillna(0, inplace=True)
    return df


def build_feature_cols(df_enc: pd.DataFrame) -> list:
    """
    Derive FEATURE_COLS from the actual encoded DataFrame.
    Excludes: date, num_sold, and any leftover id-like columns.
    """
    exclude = {TARGET_COL, 'date'}
    cols = [c for c in df_enc.columns if c not in exclude]
    return cols


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Load & encode
# ──────────────────────────────────────────────────────────────────────────────

def load_and_encode(filepath: str):
    global FEATURE_COLS

    df_raw = pd.read_csv(filepath)
    df_raw = clean_df(df_raw)

    df_raw['date'] = pd.to_datetime(df_raw['date'], format='%d-%m-%Y')
    df_raw.set_index('date', inplace=True)

    # Keep ALL rows for encoding; only drop missing target for training splits
    encoders = fit_encoders(df_raw.reset_index())
    df_enc   = apply_encoders(df_raw.reset_index(), encoders)

    df_enc['date'] = pd.to_datetime(df_enc['date'])
    df_enc.set_index('date', inplace=True)

    # Derive feature columns dynamically from actual encoded output
    FEATURE_COLS = build_feature_cols(df_enc)

    print(f'   Encoded columns ({len(df_enc.columns)}): {list(df_enc.columns)}')
    print(f'   Feature cols used ({len(FEATURE_COLS)}): {FEATURE_COLS}')

    return df_enc, df_raw, encoders


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Split & scale
# ──────────────────────────────────────────────────────────────────────────────

def split_and_scale(df_enc: pd.DataFrame):
    # Drop rows with missing target before splitting
    df_enc = df_enc.dropna(subset=[TARGET_COL])

    train = df_enc.loc[:'2014']
    val   = df_enc.loc['2015']
    test  = df_enc.loc['2016']

    feat_scaler = MinMaxScaler()
    feat_scaler.fit(train[FEATURE_COLS])

    tgt_scaler = MinMaxScaler()
    tgt_scaler.fit(train[[TARGET_COL]])

    def scale(split):
        f = pd.DataFrame(
            feat_scaler.transform(split[FEATURE_COLS]),
            columns=FEATURE_COLS, index=split.index
        )
        t = pd.DataFrame(
            tgt_scaler.transform(split[[TARGET_COL]]),
            columns=[TARGET_COL], index=split.index
        )
        return pd.concat([f, t], axis=1)

    return scale(train), scale(val), scale(test), feat_scaler, tgt_scaler


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Sequences
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(df: pd.DataFrame, look_back: int):
    """
    Build sliding windows using only FEATURE_COLS.
    X shape: (N - look_back,  look_back * len(FEATURE_COLS))
    y shape: (N - look_back,)
    """
    feat = df[FEATURE_COLS].values.astype(np.float32)  # (N, F)
    tgt  = df[TARGET_COL].values.astype(np.float32)    # (N,)
    n, F = feat.shape
    wins = n - look_back

    print(f'   create_sequences: n={n}, F={F}, look_back={look_back}, wins={wins}, X shape will be ({wins}, {look_back*F})')

    # Explicit stack — no reshape ambiguity
    X = np.stack([feat[i : i + look_back].flatten() for i in range(wins)])
    y = tgt[look_back:]
    return X.astype(np.float32), y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.1),

        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae'],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val,
                epochs=100, batch_size=512):
    model = build_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y_true, tgt_scaler, label="") -> dict:
    y_pred_s    = model.predict(X, verbose=0).flatten()
    y_true_orig = tgt_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = tgt_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

    rmse = math.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true_orig, y_pred_orig)

    # sMAPE: symmetric MAPE — safe when actual values are 0 or near-0
    # regular MAPE divides by actual → explodes when actual ≈ 0
    smape = np.mean(
        2 * np.abs(y_true_orig - y_pred_orig) /
        (np.abs(y_true_orig) + np.abs(y_pred_orig) + 1e-9)
    ) * 100

    # RMSE as % of mean actual — easy to interpret
    mean_actual  = np.mean(y_true_orig)
    rmse_pct     = (rmse / mean_actual) * 100 if mean_actual > 0 else np.nan

    print(f"[{label:5s}]  RMSE={rmse:8.2f} ({rmse_pct:.1f}% of mean)  MAE={mae:7.2f}  R²={r2:.4f}  sMAPE={smape:.2f}%")
    print(f"        mean_actual={mean_actual:.2f}  min={y_true_orig.min():.0f}  max={y_true_orig.max():.0f}")
    return dict(rmse=rmse, rmse_pct=rmse_pct, mae=mae, r2=r2, smape=smape,
                y_true=y_true_orig, y_pred=y_pred_orig)




# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────

def save_artefacts(model, feat_scaler, tgt_scaler, encoders, out_dir=ARTEFACT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    model.save(f'{out_dir}/ann_model.keras')
    joblib.dump(feat_scaler,   f'{out_dir}/feat_scaler.pkl')
    joblib.dump(tgt_scaler,    f'{out_dir}/tgt_scaler.pkl')
    joblib.dump(encoders,      f'{out_dir}/encoders.pkl')
    joblib.dump(FEATURE_COLS,  f'{out_dir}/feature_cols.pkl')
    print(f'Artefacts saved → {out_dir}/')


def load_artefacts(out_dir=ARTEFACT_DIR):
    global FEATURE_COLS
    from keras.models import load_model
    model        = load_model(f'{out_dir}/ann_model.keras', compile=False)
    feat_scaler  = joblib.load(f'{out_dir}/feat_scaler.pkl')
    tgt_scaler   = joblib.load(f'{out_dir}/tgt_scaler.pkl')
    encoders     = joblib.load(f'{out_dir}/encoders.pkl')
    FEATURE_COLS = joblib.load(f'{out_dir}/feature_cols.pkl')
    return model, feat_scaler, tgt_scaler, encoders


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'final_data.csv'

    print('▶ Loading & encoding …')
    df_enc, df_raw, encoders = load_and_encode(data_path)
    print(f'   Total rows after encoding: {len(df_enc):,}')

    print('▶ Splitting & scaling …')
    train_s, val_s, test_s, feat_scaler, tgt_scaler = split_and_scale(df_enc)
    print(f'   train={len(train_s):,}  val={len(val_s):,}  test={len(test_s):,}')

    print('▶ Building sequences …')
    X_train, y_train = create_sequences(train_s, LOOK_BACK)
    X_val,   y_val   = create_sequences(val_s,   LOOK_BACK)
    X_test,  y_test  = create_sequences(test_s,  LOOK_BACK)
    print(f'   X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}')

    print('▶ Training …')
    model, history = train_model(X_train, y_train, X_val, y_val)

    print('\n▶ Metrics (original scale):')
    evaluate(model, X_train, y_train, tgt_scaler, 'Train')
    evaluate(model, X_val,   y_val,   tgt_scaler, 'Val  ')
    evaluate(model, X_test,  y_test,  tgt_scaler, 'Test ')

    print('\n▶ Saving artefacts …')
    save_artefacts(model, feat_scaler, tgt_scaler, encoders)
    print('Done ✓')