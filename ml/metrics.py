"""La métrica oficial del reto.

Accuracy = 100 x max(0, 1 - WAPE), calculada por estación y promediada SIN ponderar,
para que una estación de gran volumen no tape el mal desempeño de una pequeña.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def accuracy_por_estacion(df: pd.DataFrame, col_pred: str = "y_pred") -> pd.Series:
    err = (df["y"] - df[col_pred]).abs()
    wape = err.groupby(df["station_id"]).sum() / df["y"].groupby(df["station_id"]).sum()
    return 100 * (1 - wape).clip(lower=0)


def accuracy_oficial(df: pd.DataFrame, col_pred: str = "y_pred") -> float:
    return float(accuracy_por_estacion(df, col_pred).mean())


def resumen(df: pd.DataFrame, col_pred: str = "y_pred") -> dict:
    por_est = accuracy_por_estacion(df, col_pred)
    por_hor = df.groupby("horizon").apply(
        lambda g: accuracy_oficial(g, col_pred), include_groups=False)
    return {
        "accuracy": round(float(por_est.mean()), 2),
        "peor_estacion": round(float(por_est.min()), 2),
        "mejor_estacion": round(float(por_est.max()), 2),
        "mae": round(float((df["y"] - df[col_pred]).abs().mean()), 2),
        "sesgo": round(float((df[col_pred] - df["y"]).mean()), 2),
        "por_horizonte": {int(h): round(float(v), 2) for h, v in por_hor.items()},
        "por_estacion": {k: round(float(v), 2) for k, v in por_est.items()},
        "n": int(len(df)),
    }


def clip_no_negativo(pred: np.ndarray) -> np.ndarray:
    """La API exige predicciones finitas y no negativas."""
    return np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0,
                         posinf=0.0, neginf=0.0).clip(min=0)
