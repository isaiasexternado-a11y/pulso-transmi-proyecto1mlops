"""Carga del panel de demanda y contexto.

Hay dos orígenes y no son intercambiables por gusto:

- **Supabase** es la verdad operativa. Contiene el histórico semilla más todo
  lo que el collector ha traído del stream. Es lo que la inferencia debe usar:
  el corte que pide el ciclo vigente sólo existe aquí.
- **CSV del SDK** es el histórico semilla congelado. Sirve para reproducir un
  backtest viejo exactamente como se corrió, sin que datos nuevos lo muevan.

Por defecto se usa Supabase y se cae al CSV si no hay credenciales, para que
el trabajo exploratorio siga corriendo en una máquina sin configurar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERIODOS_DIA = 96      # 24 h / 15 min
PERIODOS_SEMANA = 672  # 7 d

TZ = "America/Bogota"
# PostgREST recorta toda respuesta a 1000 filas (max-rows del servidor). Pedir
# más no trae más: sólo hace creer que la página estaba incompleta y corta la
# lectura antes de tiempo.
PAGINA = 1000


def data_dir() -> str:
    return os.environ.get("PULSO_DATA_DIR", "pulso-transmi-sdk/data")


# ------------------------------------------------------------------- orígenes

def _leer_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data_dir()
    obs = pd.read_csv(f"{d}/observations.csv", dtype={"station_id": str},
                      parse_dates=["observed_at"])
    ctx = pd.read_csv(f"{d}/context.csv", parse_dates=["observed_at"])
    est = pd.read_csv(f"{d}/stations.csv", dtype={"station_id": str})
    return obs, ctx, est


def _traer_todo(sb, tabla: str, columnas: str, orden: str) -> list[dict]:
    """PostgREST no devuelve tablas grandes de un solo golpe."""
    filas, desplazamiento = [], 0
    while True:
        lote = sb.seleccionar(tabla, select=columnas, order=orden,
                              limit=PAGINA, offset=desplazamiento)
        filas.extend(lote)
        if not lote or len(lote) < PAGINA:
            return filas
        desplazamiento += len(lote)


def _leer_supabase() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from collector.entorno import Supabase

    sb = Supabase()
    obs = pd.DataFrame(_traer_todo(
        sb, "observations", "station_id,observed_at,demand",
        "observed_at.asc,station_id.asc"))
    ctx = pd.DataFrame(_traer_todo(
        sb, "context",
        "observed_at,rain_mm,rain_forecast,temperature_c,temperature_forecast,event_intensity",
        "observed_at.asc"))
    est = pd.DataFrame(sb.seleccionar("stations", select="*", limit=PAGINA))

    for marco in (obs, ctx):
        marco["observed_at"] = pd.to_datetime(
            marco["observed_at"], utc=True).dt.tz_convert(TZ)
    obs["station_id"] = obs["station_id"].astype(str)
    if "station_id" in est.columns:
        est["station_id"] = est["station_id"].astype(str)
    return obs, ctx, est


# ---------------------------------------------------------------------- carga

def cargar(origen: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (demanda ancha, contexto, estaciones).

    La demanda ancha tiene un índice temporal completo cada 15 min y una columna
    por estación, que es la forma en que el backtest la consulta.

    `origen` es "supabase", "csv", o None para decidirlo por el entorno.
    """
    origen = origen or os.environ.get("PULSO_ORIGEN_DATOS") or "auto"

    if origen == "csv":
        obs, ctx, est = _leer_csv()
    elif origen == "supabase":
        obs, ctx, est = _leer_supabase()
    elif origen == "auto":
        try:
            obs, ctx, est = _leer_supabase()
        except SystemExit:
            print("[data] sin credenciales de Supabase; uso el CSV semilla",
                  file=sys.stderr)
            obs, ctx, est = _leer_csv()
    else:
        raise ValueError(f"origen desconocido: {origen}")

    ancha = obs.pivot(index="observed_at", columns="station_id",
                      values="demand").sort_index()
    ctx = ctx.set_index("observed_at").sort_index().reindex(ancha.index)

    # La API publica observaciones nuevas cada ciclo pero dejó el contexto
    # congelado al final del histórico semilla. Sin relleno, las cinco
    # features de contexto llegan como NaN y el modelo no puede predecir.
    #
    # Se arrastra el último valor conocido. Es un parche consciente y usa
    # sólo información del pasado, así que no hay fuga temporal, pero el
    # valor envejece: un "pronóstico" de lluvia de hace horas vale poco.
    # Mientras el desfase crezca, hay que medir cuánto aporta el contexto
    # y considerar un champion que no dependa de él.
    sin_contexto = int(ctx.isna().all(axis=1).sum())
    if sin_contexto:
        ultimo = ctx.dropna(how="all").index.max()
        print(f"[data] contexto congelado en {ultimo}: {sin_contexto} periodos "
              f"sin clima. Se arrastra el último valor conocido.", file=sys.stderr)
        ctx = ctx.ffill()

    esperado = pd.date_range(ancha.index[0], ancha.index[-1], freq="15min")
    if not ancha.index.equals(esperado):
        faltan = esperado.difference(ancha.index)
        raise ValueError(
            f"el índice temporal tiene {len(faltan)} huecos; el backtest asume "
            f"serie completa. Primero: {list(faltan[:3])}")

    return ancha, ctx, est


def calendario(ts: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "slot": ts.hour * 4 + ts.minute // 15,
        "dow": ts.dayofweek,
        "es_finde": (ts.dayofweek >= 5).astype(int),
    }, index=range(len(ts)))
