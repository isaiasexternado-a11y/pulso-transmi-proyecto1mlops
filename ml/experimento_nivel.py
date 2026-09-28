"""¿Cuánto recupera corregir el nivel de cada estación con lo que acaba de pasar?

Desde el 13 virtual el drift es de nivel, no de forma: 05100 cayó a menos de
la mitad (~62k -> ~26k pasajeros al día) y 07111 subió 40 %. El champion con
mezcla de persistencia sólo corrige el +15; en +45/+60 el nivel es 80 % GBM, y
en 05100 la accuracy de 24 h quedó en 21,8 %. Sin esa estación, 82,6 %.

Receta: el champion tal como se entrega (con su mezcla), multiplicado por un
factor por estación

    r_s = Σ observado / Σ champion   sobre los objetivos ya resueltos en las
                                     últimas L horas antes de data_cutoff
    final = champion · (1 + α (r_s − 1)),   r_s acotado a [R_MIN, R_MAX]

El denominador son *backcasts*: lo que el champion habría predicho en cada
origen horario previo, recalculado con los datos de ese momento. No se usan las
predicciones guardadas, porque una vez activa la corrección esas ya vendrían
corregidas y el factor se realimentaría.

Validación temporal, nunca aleatoria:

    control       ciclos antes de INICIO_DRIFT (régimen sin drift)
    calibración   ciclos del drift antes de --particion  -> elige L y α
    prueba        ciclos del drift desde --particion     -> mide

Regla de elección: la mejor accuracy en calibración entre las combinaciones
que no le quitan más de COSTO_CONTROL puntos al control. Sin esa guarda la
calibración elegía L=6 h con α=0,75, que ganaba en drift pero costaba 0,33
sin él: con seis horas nocturnas el factor es ruido. La guarda se agregó
después de ver esa primera corrida, y así queda dicho. Es la misma lógica
de las compuertas de `entrenar.py`: no empeorar lo que ya funcionaba.

Uso:
    python3 -m ml.experimento_nivel
    python3 -m ml.experimento_nivel --particion 2026-09-15T00:00:00Z
    python3 -m ml.experimento_nivel --registrar   # deja el candidato en `models`
"""
from __future__ import annotations

import argparse
import json
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
from metrics import accuracy_oficial, accuracy_por_estacion  # noqa: E402
from pipeline.entregar import champion, predecir, version_en_contrato  # noqa: E402

INICIO_STREAM = pd.Timestamp("2026-09-09T05:00:00Z")   # primer ciclo evaluable
INICIO_DRIFT = pd.Timestamp("2026-09-13T05:00:00Z")    # 00:00 Bogotá del 13 virtual
PARTICION = pd.Timestamp("2026-09-14T17:00:00Z")
VENTANAS_H = [6, 12, 24, 48, 72]
ALFAS = [0.0, 0.25, 0.5, 0.75, 1.0]
R_MIN, R_MAX = 0.4, 2.5
COSTO_CONTROL = 0.10
HORIZONTES = 4


def backcasts(modelo, ficha, ancha, ctx, desde: pd.Timestamp) -> pd.DataFrame:
    """Predicción del champion en cada origen horario, con su valor real."""
    ts = ancha.index
    fin = ts[-1 - HORIZONTES]
    origenes = [o for o in ts[(ts >= desde) & (ts <= fin)] if o.minute == 0]
    paso = ts[1] - ts[0]
    filas = []
    for i, o in enumerate(origenes):
        objetivos = [(e, o + h * paso) for e in ancha.columns
                     for h in range(1, HORIZONTES + 1)]
        valores = predecir(modelo, ficha, ancha, ctx, o, objetivos)
        for (e, t), v in zip(objetivos, valores):
            filas.append((o, e, t, round((t - o) / paso), float(ancha.at[t, e]), max(v, 0.0)))
        if i % 24 == 0:
            print(f"  backcast {o}  ({i + 1}/{len(origenes)})", file=sys.stderr)
    return pd.DataFrame(filas, columns=["origen", "station_id", "target_at",
                                        "horizon", "y", "base"])


def factores(bc: pd.DataFrame, ventana_h: int) -> pd.DataFrame:
    """r por (origen, estación) usando sólo objetivos con target_at <= origen."""
    ancho = pd.Timedelta(hours=int(ventana_h))
    por_t = bc.groupby(["station_id", "target_at"])[["y", "base"]].first()
    out = []
    for est, g in por_t.groupby(level="station_id"):
        g = g.droplevel("station_id").sort_index()
        suma = g.rolling(ancho, closed="right").sum()   # (t − L, t]
        r = (suma["y"] / suma["base"]).clip(R_MIN, R_MAX)
        out.append(pd.DataFrame({"station_id": est, "origen": r.index, "r": r.values}))
    return pd.concat(out)


def aplicar(bc: pd.DataFrame, fac: pd.DataFrame, alfa: float) -> pd.Series:
    d = bc.merge(fac, on=["origen", "station_id"], how="left")
    r = d["r"].fillna(1.0).to_numpy()
    return pd.Series(d["base"].to_numpy() * (1 + alfa * (r - 1)), index=bc.index)


def registrar(sb: Supabase, padre: dict, nivel: dict, evidencia: dict) -> str:
    """Candidato = el mismo artefacto del champion con la corrección en la ficha."""
    version = (f"{padre['hyperparams']['version']}-niv-{nivel['ventana_h']:02d}"
               f"-{round(nivel['alfa'] * 100):03d}")
    if not version_en_contrato(version):
        raise SystemExit(f"versión fuera de contrato: {version!r}")
    ya = sb.seleccionar("models", select="model_id",
                        **{"hyperparams->>version": f"eq.{version}"})
    if ya:
        print(f"ya estaba registrado: {ya[0]['model_id']}")
        return ya[0]["model_id"]

    fila = {k: padre[k] for k in (
        "kind", "algorithm", "feature_set", "feature_list", "train_start",
        "train_end", "n_train_rows", "artifact_uri", "artifact_sha256")}
    fila.update({
        "name": padre["name"] + " x nivel",
        "hyperparams": {**padre["hyperparams"], "version": version,
                        "correccion_nivel": nivel, "evidencia_nivel": evidencia},
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
    ancha, ctx, _ = cargar(origen="supabase")
    modelo, ficha = champion(sb)
    print(f"champion: {ficha['name']}  ({ficha['hyperparams']['version']})")

    # Hace falta historia de backcasts antes del primer ciclo evaluado.
    desde = INICIO_STREAM - pd.Timedelta(hours=int(max(VENTANAS_H)))
    bc = backcasts(modelo, ficha, ancha, ctx, desde.tz_convert(ancha.index.tz))
    o_utc = bc["origen"].dt.tz_convert("UTC")
    evaluable = o_utc >= INICIO_STREAM
    tramos = {
        "control_sin_drift": evaluable & (o_utc < INICIO_DRIFT),
        "calibracion": (o_utc >= INICIO_DRIFT) & (o_utc < particion),
        "prueba": o_utc >= particion,
    }
    tramos["drift_completo"] = tramos["calibracion"] | tramos["prueba"]

    cache = {L: factores(bc, L) for L in VENTANAS_H}
    cal, ctl = bc[tramos["calibracion"]], bc[tramos["control_sin_drift"]]
    ctl_base = accuracy_oficial(ctl, "base")
    rejilla, costo = {}, {}
    for L in VENTANAS_H:
        for a in ALFAS:
            corr = aplicar(bc, cache[L], a)
            rejilla[(L, a)] = accuracy_oficial(cal.assign(m=corr[cal.index]), "m")
            costo[(L, a)] = ctl_base - accuracy_oficial(ctl.assign(m=corr[ctl.index]), "m")
    admisibles = [k for k in rejilla if costo[k] <= COSTO_CONTROL]
    L, a = max(admisibles, key=rejilla.get)
    print(f"\nelegido en calibración: L={L} h, α={a}  "
          f"(calibración {rejilla[(L, a)]:.2f}, sin corrección {rejilla[(L, 0.0)]:.2f}, "
          f"costo en control {costo[(L, a)]:+.2f})")

    bc["m"] = aplicar(bc, cache[L], a)
    resultado = {}
    print(f"\n{'tramo':20} {'ciclos':>7} {'champion':>9} {'nivel':>8} {'delta':>7}")
    for k, msk in tramos.items():
        d = bc[msk]
        ch, m = accuracy_oficial(d, "base"), accuracy_oficial(d, "m")
        resultado[k] = {"ciclos": int(d["origen"].nunique()), "n": int(len(d)),
                        "champion": round(ch, 2), "nivel": round(m, 2)}
        print(f"{k:20} {d['origen'].nunique():7} {ch:9.2f} {m:8.2f} {m - ch:+7.2f}")

    p = bc[tramos["prueba"]]
    est = pd.DataFrame({"champion": accuracy_por_estacion(p, "base"),
                        "nivel": accuracy_por_estacion(p, "m")}).round(1)
    est["delta"] = est["nivel"] - est["champion"]
    print("\nprueba por estación:\n" + est.to_string())

    print("\nrobustez en prueba (L x α, sin reelegir):")
    tabla = pd.DataFrame({al: {Lx: accuracy_oficial(p.assign(m=aplicar(bc, cache[Lx], al)[p.index]), "m")
                               for Lx in VENTANAS_H} for al in ALFAS}).round(2)
    print(tabla.to_string())

    salida = Path("ml/resultados/experimento_nivel.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "calculado": datetime.now(timezone.utc).isoformat(),
        "champion": ficha["hyperparams"]["version"],
        "inicio_drift": str(INICIO_DRIFT), "particion": str(particion),
        "elegido": {"ventana_h": L, "alfa": a, "r_min": R_MIN, "r_max": R_MAX},
        "tramos": resultado,
        "prueba_por_estacion": est.to_dict(orient="index"),
        "robustez_prueba": {str(k): v for k, v in tabla.to_dict().items()},
    }, indent=2, ensure_ascii=False, default=float))
    print(f"\ndetalle -> {salida}")

    if guardar_candidato:
        if resultado["prueba"]["nivel"] <= resultado["prueba"]["champion"]:
            raise SystemExit("la corrección no gana en prueba: no se registra")
        registrar(sb, ficha, {"ventana_h": L, "alfa": a, "r_min": R_MIN, "r_max": R_MAX},
                  {k: resultado[k] for k in ("prueba", "control_sin_drift")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default=str(PARTICION))
    ap.add_argument("--registrar", action="store_true")
    a = ap.parse_args()
    main(pd.Timestamp(a.particion), a.registrar)
