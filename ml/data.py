"""Carga del panel de demanda y contexto."""
from __future__ import annotations

import os
import pandas as pd

PERIODOS_DIA = 96      # 24 h / 15 min
PERIODOS_SEMANA = 672  # 7 d


def data_dir() -> str:
    return os.environ.get("PULSO_DATA_DIR", "pulso-transmi-sdk/data")


def cargar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (demanda ancha, contexto, estaciones).

    La demanda ancha tiene un índice temporal completo cada 15 min y una columna
    por estación, que es la forma en que el backtest la consulta.
    """
    d = data_dir()
    obs = pd.read_csv(f"{d}/observations.csv", dtype={"station_id": str},
                      parse_dates=["observed_at"])
    ctx = pd.read_csv(f"{d}/context.csv", parse_dates=["observed_at"])
    est = pd.read_csv(f"{d}/stations.csv", dtype={"station_id": str})

    ancha = obs.pivot(index="observed_at", columns="station_id",
                      values="demand").sort_index()
    ctx = ctx.set_index("observed_at").sort_index().reindex(ancha.index)

    esperado = pd.date_range(ancha.index[0], ancha.index[-1], freq="15min")
    if not ancha.index.equals(esperado):
        raise ValueError("el índice temporal tiene huecos; el backtest asume serie completa")

    return ancha, ctx, est


def calendario(ts: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "slot": ts.hour * 4 + ts.minute // 15,
        "dow": ts.dayofweek,
        "es_finde": (ts.dayofweek >= 5).astype(int),
    }, index=range(len(ts)))
