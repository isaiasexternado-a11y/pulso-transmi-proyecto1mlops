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

# La API dejó de publicar contexto el 2026-09-08 23:45 y `data.cargar` arrastra
# el último valor conocido para que el modelo pueda seguir prediciendo. Es un
# parche honesto, pero el valor envejece: a estas alturas son decenas de
# periodos con el mismo "pronóstico" de lluvia. Un modelo que parte por esas
# cinco variables está partiendo por una constante disfrazada de información,
# y bajo drift eso es peor que ignorarlas.
COLS_CONTEXTO = ["rain_forecast", "temp_forecast",
                 "rain_origen", "temp_origen", "evento_origen"]
COLS_SIN_CONTEXTO = [c for c in COLS_NUM if c not in COLS_CONTEXTO]


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


def construir_para_objetivos(ancha: pd.DataFrame, ctx: pd.DataFrame,
                             origen: pd.Timestamp,
                             objetivos: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """Variables para una lista exacta de (estación, instante objetivo).

    A diferencia de `construir`, los objetivos aquí caen en el futuro: todavía no
    existe `y`. Del contexto en el instante objetivo tampoco hay pronóstico
    publicado, así que se usa el último disponible en el origen. Es una
    aproximación deliberada, no un descuido: el objetivo está a minutos del corte
    y esas variables pesan ~0.05 puntos de accuracy.
    """
    ts = ancha.index
    o = ts.get_loc(origen)
    vals = ancha.to_numpy(dtype=float)
    roll1h = ancha.rolling(4).mean().to_numpy()
    roll24h = ancha.rolling(PERIODOS_DIA).mean().to_numpy()
    roll7d = ancha.rolling(PERIODOS_SEMANA).mean().to_numpy()
    col = {e: i for i, e in enumerate(ancha.columns)}
    paso = ts[1] - ts[0]

    def ctx_en(idx: int, campo: str) -> float:
        i = idx if 0 <= idx < len(ctx) else o
        return float(ctx[campo].to_numpy()[i])

    filas = []
    for est, objetivo in objetivos:
        objetivo = pd.Timestamp(objetivo).tz_convert(ts.tz)
        h = int((objetivo - origen) / paso)
        if h < 1:
            raise ValueError(f"objetivo {objetivo} no es posterior al origen {origen}")
        t = o + h
        si = col[est]
        filas.append({
            "station_id": est, "target_at": objetivo, "horizon": h,
            "lag0": vals[o, si], "lag1": vals[o - 1, si], "lag2": vals[o - 2, si],
            "lag3": vals[o - 3, si], "lag4": vals[o - 4, si],
            "roll1h": roll1h[o, si], "roll24h": roll24h[o, si], "roll7d": roll7d[o, si],
            "lag_dia": vals[t - PERIODOS_DIA, si],
            "lag_sem": vals[t - PERIODOS_SEMANA, si],
            "slot": objetivo.hour * 4 + objetivo.minute // 15,
            "dow": objetivo.dayofweek,
            "es_finde": int(objetivo.dayofweek >= 5),
            "rain_forecast": ctx_en(t, "rain_forecast"),
            "temp_forecast": ctx_en(t, "temperature_forecast"),
            "rain_origen": ctx_en(o, "rain_mm"),
            "temp_origen": ctx_en(o, "temperature_c"),
            "evento_origen": ctx_en(o, "event_intensity"),
            # Calendario del ORIGEN, no del objetivo. `construir` ya las emitía
            # y aquí faltaban, así que cualquier modelo que las use —el perfil
            # con ajuste por nivel reciente, por ejemplo— entrenaba bien y
            # reventaba al predecir. La compuerta de promoción lo habría
            # atrapado, pero costando el candidato: mejor que el camino de
            # inferencia ofrezca las mismas columnas que el de entrenamiento.
            "slot_origen": origen.hour * 4 + origen.minute // 15,
            "finde_origen": int(origen.dayofweek >= 5),
        })
    return pd.DataFrame(filas)


def origenes_por_hora(ancha: pd.DataFrame, desde: pd.Timestamp,
                      hasta: pd.Timestamp, max_horizonte: int = 4) -> np.ndarray:
    """Un origen por hora, como los ciclos de la competencia.

    Se descartan los orígenes cuyo último horizonte caería fuera del panel.
    """
    idx = np.arange(len(ancha))
    ts = ancha.index
    return idx[(ts >= desde) & (ts <= hasta) & (ts.minute == 0)
               & (idx >= MIN_ORIGEN) & (idx + max_horizonte < len(ancha))]
