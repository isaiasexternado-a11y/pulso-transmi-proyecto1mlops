"""¿Cuánto recupera mezclar el champion con el último valor observado?

En el Corte 1 (ciclos abiertos desde 2026-09-24 05:00Z) el champion marca
81,46 % mientras la cabeza del leaderboard va en 84,6 %. El error no está
repartido: desde el viernes 11 virtual, 05000, 07107, 07111 y 09122 subieron
13-29 % y alargaron el pico, y el champion se queda 40-50 % abajo en la cola
(8-10 h y 18-22 h). Lleva roll1h y lag0, pero pesa demasiado el perfil.

El experimento del 23 descartó el ajuste por nivel con folds de tres semanas
en los que el drift pesaba poco. Aquí se mide sobre lo que de verdad importa:
las entregas reales, ya resueltas, con la predicción exacta que se envió.

Receta: `(1 - w_h) · champion + w_h · último observado en data_cutoff`, con
un peso por horizonte. No reentrena nada, así que no hay pickle nuevo.

Validación temporal, nunca aleatoria:

    calibración   ciclos del corte antes de --particion  -> elige w_h
    prueba        ciclos del corte desde --particion     -> mide
    control       ciclos anteriores al corte (régimen sin drift)

Los pesos se eligen sólo con calibración. Prueba y control no los tocan.

Uso:
    python3 -m ml.experimento_persistencia
    python3 -m ml.experimento_persistencia --registrar   # deja el candidato en `models`
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase    # noqa: E402
from data import cargar                    # noqa: E402
from metrics import accuracy_oficial       # noqa: E402

INICIO_CORTE = pd.Timestamp("2026-09-24T05:00:00Z")
PARTICION = pd.Timestamp("2026-09-25T00:00:00Z")
GRILLA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def todas(sb: Supabase, tabla: str, **params) -> list[dict]:
    """PostgREST corta en 1000 filas; se pagina hasta agotar."""
    filas, desde = [], 0
    while True:
        lote = sb.seleccionar(tabla, limit=1000, offset=desde, **params)
        filas += lote
        if len(lote) < 1000:
            return filas
        desde += 1000


def entregas(sb: Supabase) -> pd.DataFrame:
    """Lo que se envió de verdad, con su corte y el valor real que tuvo."""
    pred = pd.DataFrame(todas(
        sb, "predictions", submitted="is.true", order="cycle_id,station_id,target_at",
        select="cycle_id,station_id,target_at,horizon,y_pred"))
    ciclos = pd.DataFrame(todas(sb, "forecast_cycles",
                                select="cycle_id,opens_at,data_cutoff"))
    df = pred.merge(ciclos, on="cycle_id")
    for c in ("target_at", "opens_at", "data_cutoff"):
        df[c] = pd.to_datetime(df[c], utc=True)

    ancha, _, _ = cargar(origen="supabase")
    largo = ancha.stack().rename("valor")
    largo.index = largo.index.set_names(["t", "station_id"])
    largo = largo.reset_index()
    largo["t"] = largo["t"].dt.tz_convert("UTC")

    df = df.merge(largo.rename(columns={"t": "target_at", "valor": "y"}),
                  on=["target_at", "station_id"], how="inner")
    df = df.merge(largo.rename(columns={"t": "data_cutoff", "valor": "ultimo"}),
                  on=["data_cutoff", "station_id"], how="left")
    df["ultimo"] = df["ultimo"].fillna(df["y_pred"])
    return df


def mezclar(df: pd.DataFrame, pesos: dict[int, float]) -> pd.Series:
    w = df["horizon"].map(pesos)
    return (1 - w) * df["y_pred"] + w * df["ultimo"]


def elegir_pesos(cal: pd.DataFrame) -> dict[int, float]:
    """Un peso por horizonte, el que maximiza la accuracy oficial en calibración."""
    pesos = {}
    for h, g in cal.groupby("horizon"):
        pesos[int(h)] = max(GRILLA, key=lambda w: accuracy_oficial(
            g.assign(m=(1 - w) * g["y_pred"] + w * g["ultimo"]), "m"))
    return pesos


def medir(df: pd.DataFrame, pesos: dict[int, float]) -> dict:
    d = df.assign(m=mezclar(df, pesos))
    return {"ciclos": int(d["cycle_id"].nunique()), "n": int(len(d)),
            "champion": round(accuracy_oficial(d, "y_pred"), 2),
            "mezcla": round(accuracy_oficial(d, "m"), 2)}


def registrar(sb: Supabase, pesos: dict[int, float], evidencia: dict) -> str:
    """Candidato = el mismo artefacto del champion con la mezcla en la ficha."""
    padre = sb.seleccionar("models", select="*", status="eq.active",
                           order="activated_at.desc", limit=1)[0]
    version = padre["hyperparams"]["version"] + "-pers-" + \
        "-".join(f"{round(pesos[h] * 10):02d}" for h in sorted(pesos))
    ya = sb.seleccionar("models", select="model_id",
                        **{"hyperparams->>version": f"eq.{version}"})
    if ya:
        print(f"ya estaba registrado: {ya[0]['model_id']}")
        return ya[0]["model_id"]

    fila = {k: padre[k] for k in (
        "kind", "algorithm", "feature_set", "feature_list", "train_start",
        "train_end", "n_train_rows", "artifact_uri", "artifact_sha256")}
    fila.update({
        "name": padre["name"] + " x persistencia",
        "hyperparams": {**padre["hyperparams"], "version": version,
                        "mezcla_persistencia": {str(h): w for h, w in pesos.items()},
                        "evidencia_mezcla": evidencia},
        "git_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "parent_model_id": padre["model_id"],
        "status": "candidate",
    })
    nuevo = sb.insertar("models", [fila], devolver=True)[0]
    print(f"candidato registrado: {nuevo['model_id']}  ({version})")
    return nuevo["model_id"]


def main(particion: pd.Timestamp, guardar_candidato: bool) -> None:
    sb = Supabase()
    df = entregas(sb)

    corte = df[df["opens_at"] >= INICIO_CORTE]
    cal = corte[corte["opens_at"] < particion]
    prueba = corte[corte["opens_at"] >= particion]
    control = df[df["opens_at"] < INICIO_CORTE]

    pesos = elegir_pesos(cal)
    print("pesos elegidos en calibración (horizonte -> w):", pesos)

    tramos = {"calibracion": medir(cal, pesos), "prueba": medir(prueba, pesos),
              "control_sin_drift": medir(control, pesos),
              "corte_completo": medir(corte, pesos)}
    print(f"\n{'tramo':20} {'ciclos':>7} {'champion':>9} {'mezcla':>8} {'delta':>7}")
    for k, v in tramos.items():
        print(f"{k:20} {v['ciclos']:7} {v['champion']:9.2f} {v['mezcla']:8.2f} "
              f"{v['mezcla'] - v['champion']:+7.2f}")

    resultado = {"calculado": datetime.now(timezone.utc).isoformat(),
                 "inicio_corte": str(INICIO_CORTE), "particion": str(particion),
                 "pesos": pesos, "tramos": tramos}
    salida = Path("ml/resultados/experimento_persistencia.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"\ndetalle -> {salida}")

    if guardar_candidato:
        if tramos["prueba"]["mezcla"] <= tramos["prueba"]["champion"]:
            raise SystemExit("la mezcla no gana en prueba: no se registra")
        registrar(sb, pesos, {"prueba": tramos["prueba"],
                              "control_sin_drift": tramos["control_sin_drift"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default=str(PARTICION))
    ap.add_argument("--registrar", action="store_true")
    a = ap.parse_args()
    main(pd.Timestamp(a.particion), a.registrar)
