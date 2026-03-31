from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig


@dataclass
class LSTMTrainingResult:
    predictions: pd.DataFrame
    history: pd.DataFrame
    metrics: dict
    latest_forecast: dict
    feature_columns: list[str]
    target_column: str
    prediction_column: str
    sequence_length: int
    forecast_horizon: int


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for the LSTM stage. Install project dependencies first."
        ) from exc
    return tf


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "count": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "directional_accuracy": np.nan,
            "correlation": np.nan,
        }

    absolute_error = np.abs(y_true - y_pred)
    squared_error = (y_true - y_pred) ** 2
    nonzero_mask = np.abs(y_true) > 1e-8
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])))
    else:
        mape = float("nan")

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = float("nan")

    return {
        "count": int(len(y_true)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape": mape,
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "correlation": correlation,
    }


def _build_sequence_frame(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_split_column: str,
    target_date_column: str,
    sequence_length: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_matrix = dataset[feature_columns].to_numpy(dtype=np.float32)
    targets = dataset[target_column].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    y_values: list[float] = []
    metadata_rows: list[dict] = []

    last_end_index = len(dataset) - forecast_horizon
    for end_index in range(sequence_length - 1, last_end_index):
        split_name = dataset.iloc[end_index][target_split_column]
        target_value = targets[end_index]
        target_date = dataset.iloc[end_index][target_date_column]
        if pd.isna(target_value) or pd.isna(target_date) or split_name not in {"train", "validation", "test"}:
            continue

        start_index = end_index - sequence_length + 1
        sequences.append(feature_matrix[start_index : end_index + 1])
        y_values.append(float(target_value))
        metadata_rows.append(
            {
                "date": dataset.iloc[end_index]["date"],
                "target_date": target_date,
                "sample_split": split_name,
            }
        )

    if not sequences:
        raise ValueError("No LSTM samples were created. Reduce sequence length/horizon or provide more data.")

    return np.stack(sequences), np.asarray(y_values, dtype=np.float32), pd.DataFrame(metadata_rows)


def _build_lstm_model(config: PipelineConfig, feature_count: int):
    tf = _import_tensorflow()
    tf.keras.utils.set_random_seed(config.lstm_seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.lstm_sequence_length, feature_count)),
            tf.keras.layers.LSTM(
                config.lstm_units,
                dropout=config.lstm_dropout,
                recurrent_dropout=0.0,
            ),
            tf.keras.layers.Dense(config.lstm_dense_units, activation="relu"),
            tf.keras.layers.Dense(1, name="forward_return_prediction"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lstm_learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_lstm_model(
    dataset: pd.DataFrame,
    core_asset: str,
    feature_columns: list[str],
    config: PipelineConfig,
) -> LSTMTrainingResult:
    tf = _import_tensorflow()

    target_column = f"{core_asset}_forward_return_{config.lstm_forecast_horizon}p"
    target_split_column = f"{core_asset}_target_split_{config.lstm_forecast_horizon}p"
    target_date_column = f"{core_asset}_target_date_{config.lstm_forecast_horizon}p"
    prediction_column = f"{core_asset}_lstm_pred_forward_return_{config.lstm_forecast_horizon}p"
    actual_column = f"{core_asset}_lstm_actual_forward_return_{config.lstm_forecast_horizon}p"

    X, y, metadata = _build_sequence_frame(
        dataset=dataset,
        feature_columns=feature_columns,
        target_column=target_column,
        target_split_column=target_split_column,
        target_date_column=target_date_column,
        sequence_length=config.lstm_sequence_length,
        forecast_horizon=config.lstm_forecast_horizon,
    )

    split_masks = {
        split: metadata["sample_split"].eq(split).to_numpy()
        for split in ("train", "validation", "test")
    }
    if not split_masks["train"].any():
        raise ValueError("No train samples available for LSTM training.")
    if not split_masks["validation"].any():
        raise ValueError("No validation samples available for LSTM training.")
    if not split_masks["test"].any():
        raise ValueError("No test samples available for LSTM training.")

    model = _build_lstm_model(config, feature_count=len(feature_columns))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.lstm_patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X[split_masks["train"]],
        y[split_masks["train"]],
        validation_data=(X[split_masks["validation"]], y[split_masks["validation"]]),
        epochs=config.lstm_epochs,
        batch_size=config.lstm_batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    prediction_rows: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "model_type": "LSTM",
        "framework": "TensorFlow Keras",
        "sequence_length": config.lstm_sequence_length,
        "forecast_horizon_periods": config.lstm_forecast_horizon,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "epochs_requested": config.lstm_epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_validation_loss": float(np.min(history.history["val_loss"])) if history.history.get("val_loss") else np.nan,
    }

    for split_name in ("train", "validation", "test"):
        mask = split_masks[split_name]
        split_predictions = model.predict(X[mask], verbose=0).reshape(-1)
        split_truth = y[mask]
        split_meta = metadata.loc[mask].copy().reset_index(drop=True)
        split_meta["sample_split"] = split_name
        split_meta[prediction_column] = split_predictions
        split_meta[actual_column] = split_truth
        split_meta[f"{core_asset}_lstm_residual_forward_return_{config.lstm_forecast_horizon}p"] = (
            split_meta[actual_column] - split_meta[prediction_column]
        )
        prediction_rows.append(split_meta)
        metrics[split_name] = _regression_metrics(split_truth, split_predictions)

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))

    latest_window = dataset[feature_columns].tail(config.lstm_sequence_length).to_numpy(dtype=np.float32)
    if len(latest_window) < config.lstm_sequence_length:
        raise ValueError("Not enough rows to generate the latest LSTM forecast window.")
    latest_prediction = float(model.predict(latest_window[np.newaxis, :, :], verbose=0).reshape(-1)[0])
    latest_forecast = {
        "forecast_date": str(dataset["date"].iloc[-1].date()),
        "forecast_horizon_periods": config.lstm_forecast_horizon,
        "predicted_forward_return": latest_prediction,
        "predicted_direction": "Bullish" if latest_prediction >= 0 else "Bearish",
    }

    return LSTMTrainingResult(
        predictions=predictions,
        history=history_frame,
        metrics=metrics,
        latest_forecast=latest_forecast,
        feature_columns=feature_columns,
        target_column=target_column,
        prediction_column=prediction_column,
        sequence_length=config.lstm_sequence_length,
        forecast_horizon=config.lstm_forecast_horizon,
    )
