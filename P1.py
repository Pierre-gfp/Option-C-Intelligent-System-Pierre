# --- P1: LSTM stock prediction (yfinance version – no yahoo_fin needed) ---
import os, warnings, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # masque logs INFO/WARN TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # évite petites variations numériques
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from collections import deque

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# seeds pour reproductibilité
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

FEATURES = ["adjclose", "volume", "open", "high", "low"]

def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(ticker, n_steps=60, lookup_step=1, test_size=0.2,
              split_by_date=True, scale=True, feature_columns=FEATURES,
              start="2018-01-01", end=None):
    """Télécharge OHLCV via yfinance, scale, fenêtrage séquentiel et split train/test."""
    # --- téléchargement des données (colonnes simples) ---
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, group_by="column")
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}. Check ticker/dates.")
    df.dropna(inplace=True)

    # --- harmoniser les noms de colonnes (gère MultiIndex) ---
    if isinstance(df.columns, pd.MultiIndex):
        # ex: ('Adj Close','AAPL') -> 'adjclose'
        df.columns = [str(col[0]).lower().replace(" ", "") for col in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "") for c in df.columns]

    # garantir 'adjclose'
    if "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]

    result = {"df": df.copy()}

    # vérifier les features demandées
    for col in feature_columns:
        assert col in df.columns, f"'{col}' not in dataframe columns: {df.columns.tolist()}"

    # ajouter la colonne date
    if "date" not in df.columns:
        df["date"] = df.index

    # scaling 0..1
    column_scaler = {}
    if scale:
        for col in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[col] = scaler.fit_transform(np.expand_dims(df[col].values, axis=1))
            column_scaler[col] = scaler
    result["column_scaler"] = column_scaler

    # label = future adjclose (shift du lookup_step)
    df["future"] = df["adjclose"].shift(-lookup_step)

    # séquence de fin (pour inférence future si besoin)
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # enlever NaN après le shift
    df.dropna(inplace=True)

    # construire les séquences glissantes
    sequences, sequence_data = deque(maxlen=n_steps), []
    for row, target in zip(df[feature_columns + ["date"]].values, df["future"].values):
        sequences.append(row)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # garder la dernière séquence complète
    last_seq = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    result["last_sequence"] = np.array(last_seq, dtype=np.float32)

    # X / y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq); y.append(target)
    X, y = np.array(X), np.array(y)

    # split
    if split_by_date:
        train_n = int((1 - test_size) * len(X))
        result["X_train"], result["y_train"] = X[:train_n], y[:train_n]
        result["X_test"],  result["y_test"]  = X[train_n:], y[train_n:]
        shuffle_in_unison(result["X_train"], result["y_train"])
        shuffle_in_unison(result["X_test"],  result["y_test"])
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=True)
        result.update({"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te})

    # retirer la colonne date des tenseurs et caster
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"]  = result["X_test"][:,  :, :len(feature_columns)].astype(np.float32)
    return result

def create_model(sequence_length, n_features, units=128, n_layers=2,
                 cell=LSTM, dropout=0.3, bidirectional=False,
                 loss="mean_absolute_error", optimizer="rmsprop"):
    m = Sequential()
    # 1ère couche avec input_shape
    layer = cell(units, return_sequences=(n_layers > 1), input_shape=(sequence_length, n_features))
    m.add(Bidirectional(layer) if bidirectional else layer)
    m.add(Dropout(dropout))
    # couches cachées
    for i in range(1, n_layers):
        is_last = (i == n_layers - 1)
        layer = cell(units, return_sequences=not is_last)
        m.add(Bidirectional(layer) if bidirectional else layer)
        m.add(Dropout(dropout))
    m.add(Dense(1, activation="linear"))
    m.compile(loss=loss, optimizer=optimizer, metrics=["mean_absolute_error"])
    return m

if __name__ == "__main__":
    TICKER = "AAPL"
    START  = "2018-01-01"
    END    = None
    N_STEPS = 60
    LOOKUP  = 1

    data = load_data(TICKER, n_steps=N_STEPS, lookup_step=LOOKUP,
                     start=START, end=END)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    model = create_model(sequence_length=X_train.shape[1],
                         n_features=X_train.shape[2],
                         units=128, n_layers=2, dropout=0.3,
                         bidirectional=False)

    history = model.fit(X_train, y_train, epochs=25, batch_size=64,
                        validation_split=0.1, verbose=1)

    # prédictions
    y_pred = model.predict(X_test).reshape(-1)

    # inverse scaling en dollars (adjclose)
    scaler = data["column_scaler"]["adjclose"]
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)

    # métriques + graphe
    mae  = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.legend()
    plt.title(f"P1 – {TICKER} | MAE={mae:.2f} RMSE={rmse:.2f}")
    plt.savefig("outputs/p1_predictions.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/p1_predictions.png")
    plt.show()  
