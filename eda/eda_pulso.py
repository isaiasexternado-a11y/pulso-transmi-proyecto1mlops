"""EDA del corte inicial de Pulso TransMi.

Objetivo: entender estructura, tipologia y calidad de las tres tablas
(stations, observations, context) antes de construir features o modelos.

Uso:
    python eda/eda_pulso.py

Escribe un resumen legible por consola y vuelca todas las cifras en
eda/eda_summary.json para reutilizarlas en el reporte.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    os.getenv("PULSO_DATA_DIR", Path(__file__).resolve().parent.parent / "pulso-transmi-sdk" / "data")
)
OUT_JSON = Path(__file__).resolve().parent / "eda_summary.json"

PERIODS_PER_DAY = 96          # 15 min
WEEKLY_LAG = PERIODS_PER_DAY * 7
DOW_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

summary: dict = {}


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"no serializable: {type(obj)}")


# ---------------------------------------------------------------- carga
# station_id se lee como string a proposito: los IDs traen ceros iniciales
# ("03000") y leerlos como int los destruye, rompiendo cualquier join.
stations = pd.read_csv(DATA_DIR / "stations.csv", dtype={"station_id": "string"})
observations = pd.read_csv(
    DATA_DIR / "observations.csv",
    dtype={"station_id": "string"},
    parse_dates=["observed_at"],
)
context = pd.read_csv(DATA_DIR / "context.csv", parse_dates=["observed_at"])
metadata = json.loads((DATA_DIR / "metadata.json").read_text())

# hora local de Bogota para los patrones de calendario
for frame in (observations, context):
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).dt.tz_convert(
        "America/Bogota"
    )

observations = observations.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
context = context.sort_values("observed_at").reset_index(drop=True)


# ------------------------------------------------- 1. estructura y tipos
rule("1. ESTRUCTURA Y TIPOLOGIA")

def schema(frame: pd.DataFrame, name: str) -> list[dict]:
    rows = []
    for col in frame.columns:
        s = frame[col]
        rows.append(
            {
                "columna": col,
                "dtype": str(s.dtype),
                "nulos": int(s.isna().sum()),
                "unicos": int(s.nunique(dropna=True)),
                "ejemplo": str(s.dropna().iloc[0]) if s.notna().any() else "",
            }
        )
    print(f"\n-- {name}: {len(frame):,} filas x {frame.shape[1]} columnas")
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


summary["schema"] = {
    "stations": schema(stations, "stations"),
    "observations": schema(observations, "observations"),
    "context": schema(context, "context"),
}
summary["shapes"] = {
    "stations": list(stations.shape),
    "observations": list(observations.shape),
    "context": list(context.shape),
}
summary["metadata"] = metadata

print("\n-- rangos numericos")
num_ranges = {}
for name, frame in (("observations", observations), ("context", context)):
    desc = frame.select_dtypes("number").describe().T
    if not desc.empty:
        print(f"\n{name}:")
        print(desc.round(3).to_string())
        num_ranges[name] = desc.round(4).to_dict(orient="index")
summary["describe"] = num_ranges


# --------------------------------------------------------- 2. calidad
rule("2. CALIDAD DE DATOS")

key_dupes = int(observations.duplicated(subset=["observed_at", "station_id"]).sum())
ctx_dupes = int(context.duplicated(subset=["observed_at"]).sum())
station_dupes = int(stations.duplicated(subset=["station_id"]).sum())

ids_obs = set(observations["station_id"].unique())
ids_cat = set(stations["station_id"].unique())

# rejilla temporal esperada: cada 15 min, sin huecos, igual para las 12 estaciones
grid = pd.date_range(
    observations["observed_at"].min(), observations["observed_at"].max(), freq="15min"
)
per_station = observations.groupby("station_id")["observed_at"].agg(["count", "min", "max"])
gaps_by_station = {
    sid: int(len(grid) - c) for sid, c in per_station["count"].items() if c != len(grid)
}

ctx_grid_gaps = int(len(grid) - len(context))
neg_demand = int((observations["demand"] < 0).sum())
zero_demand = int((observations["demand"] == 0).sum())
non_integer = int((observations["demand"] % 1 != 0).sum())

quality = {
    "nulos_totales": {
        "stations": int(stations.isna().sum().sum()),
        "observations": int(observations.isna().sum().sum()),
        "context": int(context.isna().sum().sum()),
    },
    "duplicados_clave_observations": key_dupes,
    "duplicados_clave_context": ctx_dupes,
    "duplicados_station_id": station_dupes,
    "estaciones_catalogo": len(ids_cat),
    "estaciones_en_observations": len(ids_obs),
    "huerfanas_sin_catalogo": sorted(ids_obs - ids_cat),
    "catalogo_sin_observaciones": sorted(ids_cat - ids_obs),
    "periodos_esperados_por_estacion": len(grid),
    "estaciones_con_huecos": gaps_by_station,
    "huecos_en_context": ctx_grid_gaps,
    "demanda_negativa": neg_demand,
    "demanda_cero": zero_demand,
    "demanda_no_entera": non_integer,
    "rango_temporal": [
        observations["observed_at"].min().isoformat(),
        observations["observed_at"].max().isoformat(),
    ],
    "dias_cubiertos": int(observations["observed_at"].dt.normalize().nunique()),
}
for k, v in quality.items():
    print(f"  {k:38}: {v}")
summary["quality"] = quality

# el join observations x context debe ser 1:1 sobre observed_at
merged = observations.merge(context, on="observed_at", how="left", validate="many_to_one")
print(f"\n  join observations x context          : {len(merged):,} filas, "
      f"{int(merged['rain_mm'].isna().sum())} sin contexto")
summary["quality"]["join_sin_contexto"] = int(merged["rain_mm"].isna().sum())


# ------------------------------------------------- 3. perfil por estacion
rule("3. PERFIL DE DEMANDA POR ESTACION")

stats = (
    observations.groupby("station_id")["demand"]
    .agg(
        media="mean", mediana="median", std="std", minimo="min", maximo="max",
        p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95),
    )
    .round(2)
)
stats["cv"] = (stats["std"] / stats["media"]).round(3)
stats["ratio_pico_valle"] = (stats["p95"] / stats["p05"].replace(0, np.nan)).round(2)
stats = stats.join(stations.set_index("station_id")[["station_name", "corridor", "latitude", "longitude"]])
stats = stats.sort_values("media", ascending=False)
print(stats.to_string())

total = observations["demand"].sum()
share = (observations.groupby("station_id")["demand"].sum() / total * 100).round(2)
stats["share_pct"] = share
print(f"\n  demanda total del corte: {total:,}")
print(f"  concentracion: la estacion mayor aporta {share.max():.1f}% y la menor {share.min():.1f}%")
summary["station_stats"] = stats.reset_index().to_dict(orient="records")
summary["corridor_stats"] = (
    stats.groupby("corridor")
    .agg(estaciones=("media", "size"), demanda_media=("media", "mean"))
    .round(2)
    .reset_index()
    .to_dict(orient="records")
)


# ------------------------------------------------ 4. patrones temporales
rule("4. PATRONES TEMPORALES")

obs = observations.copy()
obs["fecha"] = obs["observed_at"].dt.normalize()
obs["hora"] = obs["observed_at"].dt.hour
obs["slot"] = obs["observed_at"].dt.hour * 4 + obs["observed_at"].dt.minute // 15
obs["dow"] = obs["observed_at"].dt.dayofweek
obs["es_finde"] = obs["dow"] >= 5

perfil_horario = obs.groupby("hora")["demand"].mean().round(2)
print("\n-- demanda media por hora del dia (todas las estaciones)")
print(perfil_horario.to_string())

pico = perfil_horario.idxmax()
valle = perfil_horario.idxmin()
print(f"\n  hora pico: {pico}:00 ({perfil_horario.max():.1f})   "
      f"hora valle: {valle}:00 ({perfil_horario.min():.1f})   "
      f"ratio {perfil_horario.max() / perfil_horario.min():.1f}x")

perfil_dow = obs.groupby("dow")["demand"].mean().round(2)
print("\n-- demanda media por dia de semana")
for d, v in perfil_dow.items():
    print(f"  {DOW_ES[d]:10}: {v:8.2f}")

lab = obs.loc[~obs["es_finde"], "demand"].mean()
fin = obs.loc[obs["es_finde"], "demand"].mean()
print(f"\n  laboral {lab:.2f} vs fin de semana {fin:.2f}  -> caida {100 * (1 - fin / lab):.1f}%")

serie_diaria = obs.groupby("fecha")["demand"].sum()
primera_sem = serie_diaria.head(7).mean()
ultima_sem = serie_diaria.tail(7).mean()
print(f"\n  total diario primera semana {primera_sem:,.0f} vs ultima {ultima_sem:,.0f} "
      f"-> variacion {100 * (ultima_sem / primera_sem - 1):+.1f}%")

# autocorrelacion en los lags que importan para features
serie_pivot = obs.pivot_table(index="observed_at", columns="station_id", values="demand")
acf = {}
for lag, etiqueta in ((1, "15 min"), (4, "1 hora"), (PERIODS_PER_DAY, "1 dia"),
                      (WEEKLY_LAG, "1 semana")):
    acf[etiqueta] = round(
        float(np.mean([serie_pivot[c].autocorr(lag) for c in serie_pivot.columns])), 4
    )
print("\n-- autocorrelacion media (promedio de las 12 estaciones)")
for k, v in acf.items():
    print(f"  lag {k:10}: {v:+.4f}")

summary["temporal"] = {
    "perfil_horario": perfil_horario.to_dict(),
    "perfil_slot": obs.groupby("slot")["demand"].mean().round(2).to_dict(),
    "perfil_dow": {DOW_ES[d]: float(v) for d, v in perfil_dow.items()},
    "hora_pico": int(pico),
    "hora_valle": int(valle),
    "ratio_pico_valle": round(float(perfil_horario.max() / perfil_horario.min()), 2),
    "media_laboral": round(float(lab), 2),
    "media_finde": round(float(fin), 2),
    "serie_diaria": {d.strftime("%Y-%m-%d"): float(v) for d, v in serie_diaria.items()},
    "primera_semana": round(float(primera_sem), 1),
    "ultima_semana": round(float(ultima_sem), 1),
    "autocorrelacion": acf,
}
summary["perfil_horario_por_estacion"] = {
    sid: g.groupby("hora")["demand"].mean().round(2).to_dict()
    for sid, g in obs.groupby("station_id")
}


# ------------------------------------------------------ 5. contexto
rule("5. VARIABLES DE CONTEXTO")

ctx = context.copy()
print(ctx[["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast",
           "event_intensity"]].describe().round(3).to_string())

# OJO: rain_mm casi nunca es exactamente 0 (es una serie continua) y
# event_intensity tiene cola de valores diminutos. Contar "> 0" da 99.9% y
# 89.5% respectivamente, lo cual no significa nada. Se usan umbrales.
print("\n-- prevalencia por umbral (contar '> 0' no sirve en estas series)")
umbrales = {}
for col, cortes in (("rain_mm", (0.01, 0.1, 0.5, 1.0)), ("event_intensity", (0.01, 0.1, 0.5))):
    umbrales[col] = {str(t): round(float(100 * (ctx[col] >= t).mean()), 2) for t in cortes}
    umbrales[col]["exactamente_0"] = round(float(100 * (ctx[col] == 0).mean()), 2)
    print(f"  {col}: " + "  ".join(f">={k}: {v:.1f}%" for k, v in umbrales[col].items() if k != "exactamente_0")
          + f"   (exactamente 0: {umbrales[col]['exactamente_0']:.1f}%)")

# el pronostico es lo unico disponible en tiempo real: cuanto se equivoca?
err_lluvia = (ctx["rain_forecast"] - ctx["rain_mm"])
err_temp = (ctx["temperature_forecast"] - ctx["temperature_c"])
print(f"\n  sesgo pronostico lluvia (mm)   : {err_lluvia.mean():+.3f}  MAE {err_lluvia.abs().mean():.3f}")
print(f"  sesgo pronostico temp (C)      : {err_temp.mean():+.3f}  MAE {err_temp.abs().mean():.3f}")
print(f"  corr(rain_forecast, rain_mm)   : {ctx['rain_forecast'].corr(ctx['rain_mm']):+.3f}")
print(f"  corr(temp_forecast, temp_c)    : {ctx['temperature_forecast'].corr(ctx['temperature_c']):+.3f}")

demanda_periodo = obs.groupby("observed_at")["demand"].sum().rename("demanda_total")
ctx_join = ctx.set_index("observed_at").join(demanda_periodo)
corrs = {
    col: round(float(ctx_join[col].corr(ctx_join["demanda_total"])), 4)
    for col in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]
}
print("\n-- correlacion cruda con la demanda total por periodo (CONFUNDIDA por la hora)")
for k, v in corrs.items():
    print(f"  {k:24}: {v:+.4f}")

# La correlacion cruda mezcla el efecto del clima con el ciclo diario: hace mas
# calor a las 15h, que tambien es hora de alta demanda. Para aislar el clima se
# divide la demanda por su perfil esperado (estacion x slot x tipo de dia) y se
# analiza ese residuo, que vale ~1.0 cuando el periodo es "normal".
perfil_esperado = obs.groupby(["station_id", "slot", "es_finde"])["demand"].transform("mean")
obs["residuo"] = obs["demand"] / perfil_esperado
residuo_periodo = obs.groupby("observed_at")["residuo"].mean().rename("residuo")
ctx_join = ctx_join.join(residuo_periodo)

corrs_residuo = {
    col: round(float(ctx_join[col].corr(ctx_join["residuo"])), 4)
    for col in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]
}
print("\n-- correlacion con el residuo (ciclo diario y semanal ya removido)")
for k, v in corrs_residuo.items():
    print(f"  {k:24}: {v:+.4f}")

efectos = {}
print("\n-- efecto sobre el residuo (1.00 = demanda igual a la esperada)")
for etiqueta, mask in (
    ("lluvia fuerte (>=1mm)", ctx_join["rain_mm"] >= 1.0),
    ("lluvia moderada (>=0.5mm)", ctx_join["rain_mm"] >= 0.5),
    ("seco (<0.01mm)", ctx_join["rain_mm"] < 0.01),
    ("evento fuerte (>=0.5)", ctx_join["event_intensity"] >= 0.5),
    ("evento leve (>=0.1)", ctx_join["event_intensity"] >= 0.1),
    ("sin evento (<0.01)", ctx_join["event_intensity"] < 0.01),
):
    sub = ctx_join.loc[mask, "residuo"]
    efectos[etiqueta] = {"n": int(len(sub)), "residuo_medio": round(float(sub.mean()), 4)}
    print(f"  {etiqueta:28} n={len(sub):5}  residuo {sub.mean():.3f}  "
          f"({100 * (sub.mean() - 1):+.1f}% vs esperado)")

summary["context"] = {
    "describe": ctx[["rain_mm", "rain_forecast", "temperature_c",
                     "temperature_forecast", "event_intensity"]].describe().round(4).to_dict(),
    "umbrales": umbrales,
    "sesgo_lluvia": round(float(err_lluvia.mean()), 4),
    "mae_lluvia": round(float(err_lluvia.abs().mean()), 4),
    "sesgo_temp": round(float(err_temp.mean()), 4),
    "mae_temp": round(float(err_temp.abs().mean()), 4),
    "corr_forecast_lluvia": round(float(ctx["rain_forecast"].corr(ctx["rain_mm"])), 4),
    "corr_forecast_temp": round(float(ctx["temperature_forecast"].corr(ctx["temperature_c"])), 4),
    "corr_demanda_cruda": corrs,
    "corr_demanda_residuo": corrs_residuo,
    "efectos_residuo": efectos,
}


# --------------------------------------------- 6. baselines de referencia
rule("6. BASELINES CON LA METRICA OFICIAL")

def accuracy_por_estacion(frame: pd.DataFrame, col: str) -> pd.Series:
    err = (frame["demand"] - frame[col]).abs()
    wape = err.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()
    return (100 * (1 - wape).clip(lower=0)).round(2)


base = observations.copy()
g = base.groupby("station_id")["demand"]
base["naive_dia"] = g.shift(PERIODS_PER_DAY)       # mismo momento de ayer
base["naive_semana"] = g.shift(WEEKLY_LAG)         # mismo momento de la semana pasada
base["media_movil_1d"] = g.transform(lambda s: s.shift(1).rolling(PERIODS_PER_DAY).mean())

cutoff = base["observed_at"].max() - pd.Timedelta("7D")
val = base.loc[base["observed_at"] > cutoff].dropna(
    subset=["naive_dia", "naive_semana", "media_movil_1d"]
)

# perfil (estacion, slot, tipo de dia) calculado SOLO con los 38 dias de train
train = base.loc[base["observed_at"] <= cutoff].copy()
for f in (train, val):
    f["slot"] = f["observed_at"].dt.hour * 4 + f["observed_at"].dt.minute // 15
    f["es_finde"] = f["observed_at"].dt.dayofweek >= 5
perfil = train.groupby(["station_id", "slot", "es_finde"])["demand"].mean().rename("perfil_slot")
val = val.join(perfil, on=["station_id", "slot", "es_finde"])

print(f"\n  corte de validacion: ultimos 7 dias (> {cutoff:%Y-%m-%d %H:%M})")
print(f"  filas de validacion: {len(val):,}\n")

baselines = {}
for col, nombre in (
    ("naive_dia", "Naive d-1 (mismo momento de ayer)"),
    ("naive_semana", "Naive s-1 (misma hora, semana pasada)"),
    ("media_movil_1d", "Media movil 24h"),
    ("perfil_slot", "Perfil (estacion x slot x tipo de dia)"),
):
    sc = accuracy_por_estacion(val.dropna(subset=[col]), col)
    baselines[nombre] = {
        "accuracy_promedio": round(float(sc.mean()), 2),
        "peor_estacion": round(float(sc.min()), 2),
        "mejor_estacion": round(float(sc.max()), 2),
        "por_estacion": sc.to_dict(),
    }
    print(f"  {nombre:42} accuracy {sc.mean():6.2f}  "
          f"[peor {sc.min():.2f} / mejor {sc.max():.2f}]")

summary["baselines"] = baselines

# ------------------------------------------------- 7. drift en el historico
rule("7. ESTABILIDAD DEL HISTORICO (linea base de drift)")

obs["semana"] = ((obs["observed_at"] - obs["observed_at"].min()).dt.days // 7) + 1
sem = obs.groupby("semana")["demand"].mean().round(2)
print("\n-- demanda media por semana (la semana 7 solo tiene 3 dias)")
print(sem.to_string())

pv = obs.groupby(["station_id", "semana"])["demand"].mean().unstack()
indice = (pv.div(pv[1], axis=0) * 100).round(1)   # base 100 = semana 1
delta_6v1 = (indice[6] - 100).round(1).sort_values()
print("\n-- variacion semana 6 vs semana 1 por estacion (%)")
print(delta_6v1.to_string())
print(f"\n  rango de variacion: {delta_6v1.min():+.1f}% a {delta_6v1.max():+.1f}%")

pico_semanal = {
    int(s_): int(obs[(obs["semana"] == s_) & obs["hora"].between(14, 21)]
                 .groupby("hora")["demand"].mean().idxmax())
    for s_ in sorted(obs["semana"].unique())
}
print(f"  hora del pico de la tarde por semana: {pico_semanal}")

summary["drift"] = {
    "media_semanal": {int(k): float(v) for k, v in sem.items()},
    "indice_base100": indice.to_dict(orient="index"),
    "delta_semana6_vs_1": delta_6v1.to_dict(),
    "pico_tarde_por_semana": pico_semanal,
    "rango_variacion": [float(delta_6v1.min()), float(delta_6v1.max())],
}


OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable))
print(f"\n\nResumen completo escrito en {OUT_JSON}")
