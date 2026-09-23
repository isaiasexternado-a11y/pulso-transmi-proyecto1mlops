"""¿Cuánto cuesta predecir con el contexto congelado?

La pregunta nace de una asimetría entre cómo se mide el champion y cómo
trabaja. El backtest lo valida con el clima real de esos días. Producción, en
cambio, le da clima arrastrado: la API dejó de publicar contexto el
2026-09-08 23:45 y `data.cargar` repite el último valor conocido para que el
modelo pueda seguir prediciendo. El backtest NUNCA lo midió así.

Entre backtest (87,82 %) y producción (85,68 %) hay 2,14 puntos. La demanda
baja de la madrugada explica ~0,71. Este experimento va por los otros ~1,43.

El montaje es una comparación controlada: mismos folds, mismos datos, mismo
modelo entrenado igual. Lo ÚNICO que cambia es el contexto de la ventana de
validación:

    A  contexto real        — como lo mide el backtest
    B  contexto congelado   — como lo sufre producción

La diferencia A − B es, limpiamente, lo que cuesta el contexto rancio.

Dos cuidados de diseño:

1. **Se reporta por antigüedad del congelamiento.** Congelar siete días y
   promediar exageraría: hoy producción arrastra entre 17 y 34 horas, no una
   semana. Partir el resultado por tramos dice qué cuesta ahora y qué costará
   mañana si la API no vuelve a publicar.

2. **`sin contexto` va como control.** Esa receta no lee esas columnas, así
   que su A y su B deben salir idénticas. Si difieren, el experimento está
   mal montado y el resto del resultado no vale nada.

Uso:
    python3 -m ml.experimento_contexto
    python3 -m ml.experimento_contexto --folds 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from data import cargar                                   # noqa: E402
from features import (COLS_CONTEXTO, COLS_SIN_CONTEXTO,   # noqa: E402
                      construir, origenes_por_hora)
from metrics import accuracy_por_estacion                 # noqa: E402
from models import GBMconPerfil, Perfil                   # noqa: E402

DIAS_VALIDACION = 7
TRAMOS = [(0, 24), (24, 48), (48, 168)]


def recetas():
    sc = COLS_SIN_CONTEXTO
    return [
        ("gbm + perfil (champion)", lambda: GBMconPerfil("gbm + perfil (mae)"), True),
        ("gbm + perfil sin contexto", lambda: GBMconPerfil("sin ctx", cols=sc), False),
        ("perfil (control)", lambda: Perfil(), False),
    ]


def congelar(ctx: pd.DataFrame, desde: pd.Timestamp) -> pd.DataFrame:
    """Replica lo que sufre producción: del corte en adelante, el último valor.

    No es borrar el contexto —eso sería otro experimento— sino dejarlo
    mintiendo, que es la situación real: el modelo sigue recibiendo un número
    con pinta de pronóstico, sólo que lleva horas sin actualizarse.
    """
    fuera = ctx.copy()
    ultimo = fuera.loc[fuera.index <= desde].iloc[-1]
    for c in COLS_CONTEXTO_REALES:
        fuera.loc[fuera.index > desde, c] = ultimo[c]
    return fuera


# `features.COLS_CONTEXTO` nombra las columnas del FRAME de features; en el
# marco de contexto crudo se llaman distinto.
COLS_CONTEXTO_REALES = ["rain_mm", "rain_forecast", "temperature_c",
                        "temperature_forecast", "event_intensity"]


def accuracy(val: pd.DataFrame, pred) -> float:
    return float(accuracy_por_estacion(val.assign(y_pred=pred)).mean())


def main(n_folds: int) -> None:
    ancha, ctx, _ = cargar(origen="supabase")
    fin = ancha.index[-1]
    print(f"panel: {ancha.index[0]:%Y-%m-%d} -> {fin:%Y-%m-%d %H:%M}\n")

    filas = []
    for k in range(n_folds, 0, -1):
        val_fin = fin - pd.to_timedelta(DIAS_VALIDACION * (k - 1), unit="D")
        val_ini = val_fin - pd.to_timedelta(DIAS_VALIDACION, unit="D")

        o_tr = origenes_por_hora(ancha, ancha.index[0], val_ini)
        o_tr = o_tr[o_tr + 4 < ancha.index.get_loc(val_ini)]
        o_val = origenes_por_hora(ancha, val_ini, val_fin)
        if not len(o_tr) or not len(o_val):
            continue

        # Entrenamiento SIEMPRE con contexto real: lo que se estudia es la
        # inferencia con contexto rancio, no un entrenamiento degradado.
        train = construir(ancha, ctx, o_tr).dropna()
        val_A = construir(ancha, ctx, o_val).dropna(subset=["y"])
        val_B = construir(ancha, congelar(ctx, val_ini), o_val).dropna(subset=["y"])

        horas = (val_A["target_at"] - val_ini).dt.total_seconds() / 3600
        fold = n_folds - k + 1
        print(f"fold {fold}: val {val_ini:%m-%d} -> {val_fin:%m-%d}  ({len(val_A):,} filas)")

        for nombre, crear, usa_ctx in recetas():
            m = crear().fit(train)
            pA, pB = m.predict(val_A), m.predict(val_B)
            for lo, hi in TRAMOS:
                sel = (horas >= lo) & (horas < hi)
                if sel.sum() < 200:
                    continue
                a = accuracy(val_A[sel.values], pA[sel.values])
                b = accuracy(val_B[sel.values], pB[sel.values])
                filas.append({"fold": fold, "receta": nombre, "usa_contexto": usa_ctx,
                              "tramo": f"{lo}-{hi}h", "A_real": a, "B_congelado": b,
                              "costo": a - b, "n": int(sel.sum())})
            a = accuracy(val_A, pA); b = accuracy(val_B, pB)
            filas.append({"fold": fold, "receta": nombre, "usa_contexto": usa_ctx,
                          "tramo": "todo", "A_real": a, "B_congelado": b,
                          "costo": a - b, "n": int(len(val_A))})
            print(f"   {nombre:28} A {a:6.2f}   B {b:6.2f}   costo {a-b:+.2f}")

    d = pd.DataFrame(filas)
    print("\n" + "=" * 74)
    print(f"{'receta':28} {'tramo':>8} {'A real':>8} {'B congelado':>12} {'costo':>8}")
    print("-" * 74)
    for (rec, tr), g in d.groupby(["receta", "tramo"], sort=False):
        print(f"{rec:28} {tr:>8} {g['A_real'].mean():8.2f} "
              f"{g['B_congelado'].mean():12.2f} {g['costo'].mean():+8.2f}")
    print("=" * 74)

    ctrl = d[(~d.usa_contexto) & (d.tramo == "todo")]["costo"].abs().max()
    print(f"\ncontrol: la mayor diferencia A−B en recetas sin contexto es {ctrl:.4f}")
    print("  (debe ser 0: no leen esas columnas. Si no lo es, el montaje está mal.)")

    champ = d[(d.usa_contexto) & (d.tramo == "0-24h")]["costo"].mean()
    print(f"\ncosto del contexto congelado en las primeras 24 h: {champ:+.2f} puntos")
    print("brecha sin explicar entre backtest y producción  : 1.43 puntos")

    salida = Path("ml/resultados/experimento_contexto.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(filas, indent=2, ensure_ascii=False))
    print(f"\ndetalle -> {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    main(ap.parse_args().folds)
