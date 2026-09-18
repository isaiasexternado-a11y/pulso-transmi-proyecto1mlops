"""Construcción de variables para un conjunto de orígenes de ciclo.

Regla que gobierna todo este módulo: en un origen `t` solo existe información
hasta `t`. Del contexto futuro únicamente se usan los pronósticos
(`rain_forecast`, `temperature_forecast`), nunca los valores observados, porque
al momento de predecir todavía no se conocen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import PERIODOS_DIA, PERIODOS_SEMANA

MIN_ORIGEN = PERIODOS_SEMANA  # se necesita una semana de historia para los rezagos

COLS_NUM = [
    "horizon", "lag0", "lag1", "lag2", "lag3", "lag4",
    "lag_dia", "lag_sem", "roll1h", "roll24h", "roll7d",
    "slot", "dow", "es_finde",
    "rain_forecast", "temp_forecast",
    "rain_origen", "temp_origen", "evento_origen",
]


def construir(ancha: pd.DataFrame, ctx: pd.DataFrame,
              origenes: np.ndarray, horizontes=(1, 2, 3, 4)) -> pd.DataFrame:
    origenes = np.asarray(origenes)
    if origenes.min() < MIN_ORIGEN:
        raise ValueError(f"los orígenes deben empezar en el índice {MIN_ORIGEN} o después")

    vals = ancha.to_numpy(dtype=float)
    roll1h = ancha.rolling(4).mean().to_numpy()
    roll24h = ancha.rolling(PERIODOS_DIA).mean().to_numpy()
    roll7d = ancha.rolling(PERIODOS_SEMANA).mean().to_numpy()
    ts = ancha.index

    rain_f = ctx["rain_forecast"].to_numpy(dtype=float)
    temp_f = ctx["temperature_forecast"].to_numpy(dtype=float)
    rain_o = ctx["rain_mm"].to_numpy(dtype=float)
    temp_o = ctx["temperature_c"].to_numpy(dtype=float)
    event_o = ctx["event_intensity"].to_numpy(dtype=float)

    partes = []
    for h in horizontes:
        t = origenes + h
        cal_ts = ts[t]
        for si, est in enumerate(ancha.columns):
            partes.append(pd.DataFrame({
                "target_at": cal_ts,
                "station_id": est,
                "horizon": h,
                "y": vals[t, si],
                # historia disponible en el origen
                "lag0": vals[origenes, si],
                "lag1": vals[origenes - 1, si],
                "lag2": vals[origenes - 2, si],
                "lag3": vals[origenes - 3, si],
                "lag4": vals[origenes - 4, si],
                "roll1h": roll1h[origenes, si],
                "roll24h": roll24h[origenes, si],
                "roll7d": roll7d[origenes, si],
                # mismo instante objetivo en días/semanas anteriores
                "lag_dia": vals[t - PERIODOS_DIA, si],
                "lag_sem": vals[t - PERIODOS_SEMANA, si],
                # calendario del instante objetivo
                "slot": cal_ts.hour * 4 + cal_ts.minute // 15,
                "dow": cal_ts.dayofweek,
                "es_finde": (cal_ts.dayofweek >= 5).astype(int),
                # contexto: pronóstico al objetivo, observado solo hasta el origen
                "rain_forecast": rain_f[t],
                "temp_forecast": temp_f[t],
                "rain_origen": rain_o[origenes],
                "temp_origen": temp_o[origenes],
                "evento_origen": event_o[origenes],
                # calendario del origen: lo usa el ajuste por nivel reciente
                "slot_origen": ts[origenes].hour * 4 + ts[origenes].minute // 15,
                "finde_origen": (ts[origenes].dayofweek >= 5).astype(int),
            }))

    out = pd.concat(partes, ignore_index=True)
    return out.sort_values(["target_at", "station_id", "horizon"]).reset_index(drop=True)


def origenes_por_hora(ancha: pd.DataFrame, desde: pd.Timestamp,
                      hasta: pd.Timestamp, max_horizonte: int = 4) -> np.ndarray:
    """Un origen por hora, como los ciclos de la competencia.

    Se descartan los orígenes cuyo último horizonte caería fuera del panel.
    """
    idx = np.arange(len(ancha))
    ts = ancha.index
    return idx[(ts >= desde) & (ts <= hasta) & (ts.minute == 0)
               & (idx >= MIN_ORIGEN) & (idx + max_horizonte < len(ancha))]
