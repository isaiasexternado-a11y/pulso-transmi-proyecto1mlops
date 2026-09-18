"""Backtest con origen deslizante, imitando la competencia.

Cada origen es el comienzo de un ciclo: se predicen los 4 horizontes (+15, +30,
+45 y +60 min) para las 12 estaciones usando solo información hasta ese instante.
Es deliberadamente distinto de "predecir los próximos 7 días": el reto entrega
pronósticos de una hora, y un modelo elegido con el otro montaje sería el
equivocado.

Uso:
    python3 ml/backtest.py              # todos los candidatos
    python3 ml/backtest.py --folds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import PERIODOS_DIA, cargar          # noqa: E402
from features import construir, origenes_por_hora  # noqa: E402
from metrics import resumen                     # noqa: E402
from models import candidatos                   # noqa: E402

DIAS_VALIDACION = 7


def folds(ancha: pd.DataFrame, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Ventanas de validación consecutivas, con entrenamiento expansivo."""
    fin = ancha.index[-1]
    out = []
    for k in range(n, 0, -1):
        val_fin = fin - pd.Timedelta(f"{DIAS_VALIDACION * (k - 1)}D")
        val_ini = val_fin - pd.Timedelta(f"{DIAS_VALIDACION}D")
        out.append((val_ini, val_fin, val_ini))  # (ini val, fin val, fin train)
    return out


def correr(n_folds: int = 3) -> dict:
    ancha, ctx, _ = cargar()
    print(f"panel: {ancha.shape[0]:,} periodos x {ancha.shape[1]} estaciones "
          f"({ancha.index[0]:%Y-%m-%d} -> {ancha.index[-1]:%Y-%m-%d})\n")

    modelos = candidatos()
    acumulado: dict[str, list] = {m.nombre: [] for m in modelos}
    detalle: dict[str, dict] = {}

    for i, (val_ini, val_fin, train_fin) in enumerate(folds(ancha, n_folds), 1):
        o_train = origenes_por_hora(ancha, ancha.index[0], train_fin)
        o_val = origenes_por_hora(ancha, val_ini, val_fin)
        # el último origen de train no puede alcanzar la ventana de validación
        o_train = o_train[o_train + 4 < ancha.index.get_loc(val_ini)]

        train = construir(ancha, ctx, o_train)
        val = construir(ancha, ctx, o_val)
        train = train.dropna(subset=["y"]).dropna()
        val = val.dropna(subset=["y"])

        print(f"fold {i}: train {len(train):,} filas (hasta {ancha.index[o_train[-1]]:%m-%d %H:%M})"
              f"  |  val {len(val):,} filas ({val_ini:%m-%d} -> {val_fin:%m-%d})")

        for m in modelos:
            t0 = time.time()
            m.fit(train)
            pred = m.predict(val)
            r = resumen(val.assign(y_pred=pred))
            r["segundos"] = round(time.time() - t0, 1)
            acumulado[m.nombre].append(r)
            detalle.setdefault(m.nombre, {})[f"fold{i}"] = r
            print(f"   {m.nombre:34} accuracy {r['accuracy']:6.2f}   "
                  f"mae {r['mae']:7.2f}  ({r['segundos']:.1f}s)")
        print()

    tabla = []
    for m in modelos:
        accs = [r["accuracy"] for r in acumulado[m.nombre]]
        tabla.append({
            "modelo": m.nombre,
            "es_baseline": m.es_baseline,
            "accuracy_media": round(sum(accs) / len(accs), 2),
            "accuracy_min": round(min(accs), 2),
            "por_fold": accs,
            "por_horizonte": acumulado[m.nombre][-1]["por_horizonte"],
        })
    tabla.sort(key=lambda r: r["accuracy_media"], reverse=True)

    print("=" * 78)
    print(f"{'modelo':36} {'media':>7} {'peor fold':>10}   por horizonte (+15..+60)")
    print("-" * 78)
    for r in tabla:
        marca = "  " if r["es_baseline"] else "* "
        h = " ".join(f"{v:5.1f}" for v in r["por_horizonte"].values())
        print(f"{marca}{r['modelo']:34} {r['accuracy_media']:7.2f} {r['accuracy_min']:10.2f}   {h}")
    print("=" * 78)
    print("* = candidato de ML   (sin marca = baseline)")

    mejor = tabla[0]
    mejor_base = max((r for r in tabla if r["es_baseline"]), key=lambda r: r["accuracy_media"])
    print(f"\nmejor modelo   : {mejor['modelo']}  ({mejor['accuracy_media']:.2f})")
    print(f"mejor baseline : {mejor_base['modelo']}  ({mejor_base['accuracy_media']:.2f})")
    print(f"ganancia       : {mejor['accuracy_media'] - mejor_base['accuracy_media']:+.2f} puntos")

    return {"tabla": tabla, "detalle": detalle,
            "config": {"folds": n_folds, "dias_validacion": DIAS_VALIDACION,
                       "periodos_dia": PERIODOS_DIA}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="ml/resultados/backtest.json")
    a = ap.parse_args()

    res = correr(a.folds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nresultados -> {a.out}")
