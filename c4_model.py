# c4_model.py
# Build Keras models from a concise spec: type, units, layers, dropout, bidirectional.

from typing import List, Dict, Optional
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional

LayerSpec = Dict[str, object]
# Example spec:
# [
#   {"type": "LSTM", "units": 64, "dropout": 0.2, "bidirectional": False},
#   {"type": "LSTM", "units": 32, "dropout": 0.2, "bidirectional": False},
# ]

def make_model(
    input_shape,                        # (timesteps, n_features)
    layers: List[LayerSpec],
    loss: str = "mean_squared_error",
    optimizer: str = "adam",
    metrics: Optional[list] = None,
) -> tf.keras.Model:
    """
    Build a Sequential model from a list of layer specs.
    Each spec supports keys:
      - type: "LSTM" | "GRU" | "RNN"
      - units: int
      - dropout: float in [0, 1] (optional)
      - bidirectional: bool (optional)
      - return_sequences: bool (optional)  # if omitted we auto-set it
    """
    if metrics is None:
        metrics = ["mean_absolute_error"]

    RNN_MAP = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}

    model = Sequential()
    n = len(layers)
    for i, spec in enumerate(layers):
        ltype = str(spec.get("type", "LSTM")).upper()
        units = int(spec.get("units", 64))
        dropout = float(spec.get("dropout", 0.0))
        bidir = bool(spec.get("bidirectional", False))

        Cell = RNN_MAP.get(ltype, LSTM)
        # auto return_sequences: True for all but last recurrent layer
        rs = spec.get("return_sequences", i < n - 1)

        rnn_layer = Cell(units, return_sequences=rs)
        if i == 0:
            # first layer needs input_shape
            rnn_layer = Cell(units, return_sequences=rs, input_shape=input_shape)

        if bidir:
            model.add(Bidirectional(rnn_layer))
        else:
            model.add(rnn_layer)

        if dropout > 0:
            model.add(Dropout(dropout))

    # final regression head (predict next price)
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
