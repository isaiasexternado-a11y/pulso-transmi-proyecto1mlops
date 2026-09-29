"""¿Una corrección de nivel más corta sigue mejor a la continuación del escenario?

El 16 virtual a las 08:00 el generador subió la dificultad: 02300 y 05000 pasaron
a correr entre 2 y 4,7 veces la semana anterior, 05100 cayó a ~0,15 y 03000 a
~0,4. La corrección del champion (`correccion_nivel`: L=12 h, α=0,75, r en
[0,4; 2,5]) llega tarde a un salto así: la mitad de su ventana todavía es el
régimen viejo, y aunque la viera, el tope no la deja seguirlo.

Los tres primeros del leaderboard corrigen con ventanas de 1-2 h (ver notas en
el commit). Este experimento barre la ventana corta sobre el mismo champion y
agrega el modo *quiebre* de uno de ellos:

    r_L  = Σ observado / Σ champion   sobre los objetivos de las últimas L h
    r_2L = lo mismo con 2L h
    normal:  final = champion · (1 + α (clip(r_L) − 1))
    quiebre: si r_L y r_2L se desvían de 1 más de UMBRAL_QUIEBRE en el mismo
             sentido, la estación cambió de régimen y se corrige completo:
             final = champion · clip_amplio(r_L)

El denominador son backcasts del champion SIN su corrección de nivel (sí con la
mezcla de persistencia), igual que en producción: `factores_nivel` los
recalcula con `_mezclado`.

Validación temporal, tres regímenes:

    control       9 → 13 virtual       sin drift: no se puede empeorar
    calibración   13 → 16 08:00        primer drift: elige la receta
    prueba        16 08:00 → fin       la continuación: se mide, no se elige

Regla de elección: la mejor en calibración entre las que no le cuestan más de
COSTO_CONTROL al control. La prueba se reporta completa para ver si la elegida
sobrevive al régimen nuevo.

Uso:
    python3 -m ml.experimento_nivel_corto
    python3 -m ml.experimento_nivel_corto --particion 2026-09-16T13:00:00Z
    python3 -m ml.experimento_nivel_corto --registrar   # deja el candidato en `models`
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase                  # noqa: E402
from data import cargar                                  # noqa: E402
from experimento_nivel import backcasts                  # noqa: E402
from metrics import accuracy_oficial, accuracy_por_estacion  # noqa: E402
from pipeline.entregar import champion, version_en_contrato  # noqa: E402

INICIO_STREAM = pd.Timestamp("2026-09-09T05:00:00Z")
INICIO_DRIFT = pd.Timestamp("2026-09-13T05:00:00Z")
PARTICION = pd.Timestamp("2026-09-16T13:00:00Z")      # 08:00 Bogotá del 16 virtual
VENTANAS_H = [1, 2, 3, 4, 6, 12]
ALFAS = [0.5, 0.75, 1.0]
LIMITES = {"estrecho": (0.4, 2.5), "amplio": (0.15, 5.0)}
UMBRAL_QUIEBRE = 0.2
COSTO_CONTROL = 0.10
ACTUAL = ("normal", 12, 0.75, "estrecho")               # lo que entrega hoy el champion


def razones(bc: pd.DataFrame, ventana_h: int) -> pd.DataFrame:
    """r sin acotar por (origen, estación) con objetivos en (origen − L, origen]."""
    ancho = pd.Timedelta(hours=int(ventana_h))
    por_t = bc.groupby(["station_id", "target_at"])[["y", "base"]].first()
    out = []
    for est, g in por_t.groupby(level="station_id"):
        suma = g.droplevel("station_id").sort_index().rolling(ancho, closed="right").sum()
        out.append(pd.DataFrame({"station_id": est, "origen": suma.index,
                                 "r": (suma["y"] / suma["base"].where(suma["base"] > 0)).values}))
    return pd.concat(out).set_index(["origen", "station_id"])["r"]


def corregir(bc: pd.DataFrame, r: dict, receta: tuple) -> np.ndarray:
    modo, L, alfa, lim = receta
    clave = pd.MultiIndex.from_arrays([bc["origen"], bc["station_id"]])
    corto = r[L].reindex(clave).to_numpy()
    lo, hi = LIMITES[lim]
    f = 1 + alfa * (np.clip(np.nan_to_num(corto, nan=1.0), lo, hi) - 1)
    if modo == "quiebre":
        largo = r[2 * L].reindex(clave).to_numpy()
        q = ((np.abs(corto - 1) > UMBRAL_QUIEBRE) & (np.abs(largo - 1) > UMBRAL_QUIEBRE)
             & (np.sign(corto - 1) == np.sign(largo - 1)))
        f = np.where(q, np.clip(np.nan_to_num(corto, nan=1.0), *LIMITES["amplio"]), f)
    return bc["base"].to_numpy() * f


def como_ficha(receta: tuple) -> dict:
    """La receta en el formato de `hyperparams.correccion_nivel`."""
    modo, L, alfa, lim = receta
    nivel = {"ventana_h": L, "alfa": alfa, "r_min": LIMITES[lim][0], "r_max": LIMITES[lim][1]}
    if modo == "quiebre":
        nivel["quiebre"] = {"umbral": UMBRAL_QUIEBRE,
                            "r_min": LIMITES["amplio"][0], "r_max": LIMITES["amplio"][1]}
    return nivel


def registrar(sb: Supabase, padre: dict, nivel: dict, evidencia: dict) -> str:
    """Candidato = el mismo artefacto y la misma mezcla; cambia sólo la
    corrección de nivel de la ficha."""
    raiz = re.sub(r"-niv[a-z]*-\d+-\d+$", "", padre["hyperparams"]["version"])
    marca = "nivq" if "quiebre" in nivel else "niv"
    version = f"{raiz}-{marca}-{nivel['ventana_h']:02d}-{round(nivel['alfa'] * 100):03d}"
    if not version_en_contrato(version):
        raise SystemExit(f"versión fuera de contrato: {version!r}")
    ya = sb.seleccionar("models", select="model_id", **{"hyperparams->>version": f"eq.{version}"})
    if ya:
        print(f"ya estaba registrado: {ya[0]['model_id']}")
        return ya[0]["model_id"]

    fila = {k: padre[k] for k in (
        "kind", "algorithm", "feature_set", "feature_list", "train_start",
        "train_end", "n_train_rows", "artifact_uri", "artifact_sha256")}
    fila.update({
        "name": re.sub(r"( x nivel.*)?$", "", padre["name"], count=1)
                + (" x nivel con quiebre" if "quiebre" in nivel else " x nivel"),
        "hyperparams": {**padre["hyperparams"], "version": version,
                        "correccion_nivel": nivel, "evidencia_nivel": evidencia},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "parent_model_id": padre["model_id"],
        "status": "candidate",
    })
    nuevo = sb.insertar("models", [fila], devolver=True)[0]
    print(f"candidato registrado: {nuevo['model_id']}  ({version})")
    return nuevo["model_id"]


def main(particion: pd.Timestamp, cache: Path | None, guardar_candidato: bool) -> None:
    sb = Supabase()
    ancha, ctx, _ = cargar(origen="supabase")
    modelo, ficha = champion(sb)
    print(f"champion: {ficha['name']}  ({ficha['hyperparams']['version']})")
    # Base = lo que el champion entrega ANTES de su corrección de nivel.
    base = {**ficha, "hyperparams": {k: v for k, v in ficha["hyperparams"].items()
                                     if k != "correccion_nivel"}}

    if cache and cache.exists() and pd.read_pickle(cache)["target_at"].max() >= ancha.index[-1]:
        bc = pd.read_pickle(cache)
        print(f"backcasts desde caché: {cache}")
    else:
        desde = INICIO_STREAM - pd.Timedelta(hours=2 * max(VENTANAS_H))
        bc = backcasts(modelo, base, ancha, ctx, desde.tz_convert(ancha.index.tz))
        if cache:
            bc.to_pickle(cache)

    o = bc["origen"].dt.tz_convert("UTC")
    tramos = {
        "control_sin_drift": (o >= INICIO_STREAM) & (o < INICIO_DRIFT),
        "calibracion": (o >= INICIO_DRIFT) & (o < particion),
        "prueba": o >= particion,
    }
    r = {L: razones(bc, L) for L in sorted({x for L in VENTANAS_H for x in (L, 2 * L)})}

    recetas = [("sin", 0, 0.0, "estrecho")] + [
        (m, L, a, lim) for m, L, a, lim in itertools.product(
            ["normal", "quiebre"], VENTANAS_H, ALFAS, LIMITES)]
    filas = []
    for rec in recetas:
        pred = bc["base"].to_numpy() if rec[0] == "sin" else corregir(bc, r, rec)
        d = bc.assign(m=pred)
        filas.append({"receta": rec, **{k: accuracy_oficial(d[msk], "m") for k, msk in tramos.items()}})
    tabla = pd.DataFrame(filas).set_index("receta")
    ctl_actual = tabla.loc[[ACTUAL], "control_sin_drift"].iloc[0]
    tabla["costo_control"] = ctl_actual - tabla["control_sin_drift"]

    admisibles = tabla[tabla["costo_control"] <= COSTO_CONTROL]
    elegida = admisibles["calibracion"].idxmax()
    ciclos = {k: int(bc.loc[msk, "origen"].nunique()) for k, msk in tramos.items()}
    print(f"\nciclos: {ciclos}")
    print("\ntop 15 por calibración (admisibles):")
    print(admisibles.sort_values("calibracion", ascending=False).head(15).round(2).to_string())
    print("\ntop 10 por prueba (sólo para leer; no se elige con ella):")
    print(tabla.sort_values("prueba", ascending=False).head(10).round(2).to_string())
    print("\nreferencias:")
    print(tabla.loc[[("sin", 0, 0.0, "estrecho"), ACTUAL, elegida]].round(2).to_string())

    p = bc[tramos["prueba"]]
    est = pd.DataFrame({
        "actual": accuracy_por_estacion(p.assign(m=corregir(p, r, ACTUAL)), "m"),
        "elegida": accuracy_por_estacion(p.assign(m=corregir(p, r, elegida)), "m"),
    }).round(1)
    est["delta"] = est["elegida"] - est["actual"]
    print(f"\nprueba por estación, elegida = {elegida}:\n{est.to_string()}")

    salida = RAIZ / "ml/resultados/experimento_nivel_corto.json"
    salida.write_text(json.dumps({
        "calculado": datetime.now(timezone.utc).isoformat(),
        "champion": ficha["hyperparams"]["version"],
        "particion": str(particion), "ciclos": ciclos,
        "elegida": list(elegida), "actual": list(ACTUAL),
        "tabla": {"|".join(map(str, k)): {c: round(float(v), 3) for c, v in fila.items()}
                  for k, fila in tabla.iterrows()},
        "prueba_por_estacion": est.to_dict(orient="index"),
    }, indent=2, ensure_ascii=False))
    print(f"\ndetalle -> {salida}")

    if guardar_candidato:
        fila_e, fila_a = tabla.loc[[elegida]].iloc[0], tabla.loc[[ACTUAL]].iloc[0]
        if fila_e["prueba"] <= fila_a["prueba"] or fila_e["calibracion"] <= fila_a["calibracion"]:
            raise SystemExit("la receta elegida no le gana a la actual: no se registra")
        registrar(sb, ficha, como_ficha(elegida), {
            "receta": list(elegida), "ciclos": ciclos,
            **{k: {"actual": round(float(fila_a[k]), 2), "elegida": round(float(fila_e[k]), 2)}
               for k in tramos}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default=str(PARTICION))
    ap.add_argument("--cache", type=Path, default=None,
                    help="pickle de backcasts para no recalcularlos entre corridas")
    ap.add_argument("--registrar", action="store_true")
    a = ap.parse_args()
    main(pd.Timestamp(a.particion), a.cache, a.registrar)
