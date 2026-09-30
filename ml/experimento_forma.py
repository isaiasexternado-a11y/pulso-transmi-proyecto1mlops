"""¿Cómo aguanta cada estrategia un cambio de FORMA de la demanda?

El 2026-09-30 a las 14:52 Bogotá el profesor activó la revisión 2 del drift
(`docs/drift-operations.md` de su repo): una continuación que "modifica la forma
temporal de la demanda", con seis horas virtuales de transición y un régimen
estable después. Sus referencias privadas: un Random Forest con variables
recientes FIJO saca 71,5 %; la misma familia REENTRENADA periódicamente,
86,8-88 %. Y advierte que la referencia que sólo ajustaba la escala del perfil
—lo mismo que hace nuestra corrección de nivel— deja de servir.

La continuación real es privada, así que aquí se fabrica una sobre datos
reales y se simula producción encima:

    escenario   a partir de T0, cada estación se transforma (pico corrido,
                amplitud distinta, forma invertida...) y la serie pasa de la
                vieja a la nueva linealmente en TRANSICION_H horas. Todo lo
                anterior a T0 queda intacto.
    producción  cada CADA_H horas cada estrategia se reentrena con lo resuelto
                hasta el corte (menos su reserva) y predice los ciclos
                horarios de las CADA_H horas siguientes. Las capas del
                champion (mezcla y corrección de nivel) se aplican con
                `entrenar.aplicar_capas`, calentando la ventana de nivel con
                backcasts del mismo modelo, como en producción.
    selector    simula `medir_vivo`: en cada corte elige, entre las recetas,
                la de mejor accuracy entregada en las últimas 24 h.

Se reporta, como el profesor, la adaptación (primeras 18 h) aparte del
régimen estable (18-54 h).

Validación temporal: ningún objetivo de entrenamiento es posterior al corte
menos la reserva, y las capas sólo leen `y` de objetivos ya resueltos.

Uso:
    python3 -m ml.experimento_forma
    python3 -m ml.experimento_forma --escenarios severo --anclas 2026-09-01T06:00
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from entrenar import aplicar_capas                               # noqa: E402
from features import COLS_SIN_CONTEXTO, construir, origenes_por_hora  # noqa: E402
from metrics import accuracy_oficial, accuracy_por_estacion      # noqa: E402
from models import GBMconPerfil, Perfil                          # noqa: E402

PASO = pd.to_timedelta(15, unit="min")
TRANSICION_H = 6
HORAS = 54                    # 6 de transición + 48 de observación, como el anuncio
ADAPTACION_H = 18             # el profesor excluye las primeras 18 h
CADA_H = 2                    # cadencia real de train.yml
CALENTAR_H = 5
SC = COLS_SIN_CONTEXTO
ESC3 = ("roll24h", "roll1h", "roll7d")

# Anclas: una sin drift (semilla) y otra encima del drift real del stream.
ANCLAS = ["2026-09-01T06:00", "2026-09-15T12:00"]

# Capas de producción al 2026-09-30 (hyperparams del champion activo).
MEZCLA = {"1": 0.6, "2": 0.4, "3": 0.2, "4": 0.2}
NIVEL_Q = {"ventana_h": 2, "alfa": 0.5, "r_min": 0.4, "r_max": 2.5,
           "quiebre": {"umbral": 0.2, "r_min": 0.15, "r_max": 5.0}}
CAPAS = {
    "sin capas": {},
    "mezcla": {"mezcla_persistencia": MEZCLA},
    "mezcla + nivel": {"mezcla_persistencia": MEZCLA, "correccion_nivel": NIVEL_Q},
    "mezcla + nivel sin quiebre": {"mezcla_persistencia": MEZCLA,
                                   "correccion_nivel": {k: v for k, v in NIVEL_Q.items()
                                                        if k != "quiebre"}},
}


def recetas() -> dict:
    """nombre -> (constructor, días de ventana o None)."""
    g = lambda **kw: (lambda: GBMconPerfil(cols=SC, **kw))       # noqa: E731
    return {
        "sc 14d (champion)": (g(), 14),
        "sc 3d": (g(), 3),
        "sc 14d semivida 6h": (g(semivida_h=6), 14),
        "sc 14d semivida 12h": (g(semivida_h=12), 14),
        "sc 14d semivida 24h": (g(semivida_h=24), 14),
        "norm3 14d": (g(escalas=ESC3), 14),
        "norm3 14d semivida 12h": (g(escalas=ESC3, semivida_h=12), 14),
        "norm3 14d semivida 24h": (g(escalas=ESC3, semivida_h=24), 14),
    }


# ------------------------------------------------------------------ escenario

def _amplitud(x: pd.Series, k: float) -> pd.Series:
    m = x.rolling(96, center=True, min_periods=48).mean()
    return (m + k * (x - m)).clip(lower=0)


def transformar(ancha: pd.DataFrame, t0: pd.Timestamp, escenario: str,
                semilla: int) -> tuple[pd.DataFrame, dict]:
    """La serie nueva de cada estación y la mezcla lineal de la transición."""
    rng = np.random.default_rng(semilla)
    nuevo, receta = ancha.copy(), {}
    for est in ancha.columns:
        x = ancha[est]
        if escenario == "corrimiento":
            k = int(rng.choice([-6, -4, 4, 6]))            # ±60-90 min
            y, desc = x.shift(k), f"corre {k * 15:+d} min"
        elif escenario == "amplitud":
            k = float(rng.choice([0.5, 1.6]))
            y, desc = _amplitud(x, k), f"amplitud x{k}"
        else:                                               # severo: de todo
            tipo = rng.choice(["corre", "amplitud", "invierte", "nivel"], p=[.4, .3, .15, .15])
            if tipo == "corre":
                k = int(rng.choice([-8, -6, 6, 8]))         # ±1,5-2 h
                y, desc = x.shift(k), f"corre {k * 15:+d} min"
            elif tipo == "amplitud":
                k = float(rng.choice([0.4, 1.8]))
                y, desc = _amplitud(x, k), f"amplitud x{k}"
            elif tipo == "invierte":                        # la mañana pasa a la tarde
                y, desc = x.shift(44), "invierte (+11 h)"
            else:
                k = float(rng.choice([0.6, 1.5]))
                y = _amplitud(x, 1.3) * k
                desc = f"nivel x{k} y amplitud x1.3"
        nuevo[est] = y.bfill().ffill()
        receta[est] = desc
    w = np.clip((ancha.index - t0) / pd.Timedelta(hours=TRANSICION_H), 0, 1)
    w = np.asarray(w, dtype=float)[:, None]
    out = (1 - w) * ancha.to_numpy(dtype=float) + w * nuevo.to_numpy(dtype=float)
    return pd.DataFrame(np.rint(out), index=ancha.index, columns=ancha.columns), receta


# ------------------------------------------------------------------ simulación

_TODO: pd.DataFrame | None = None


def _bloque(args):
    """Una receta, un corte, una reserva: predicciones crudas del bloque."""
    nombre, corte, reserva_h = args
    crear, dias = recetas()[nombre]
    todo = _TODO
    hasta = corte + pd.Timedelta(hours=CADA_H)
    ini = corte - pd.Timedelta(hours=CALENTAR_H)
    b = todo[(todo["origen"] >= ini) & (todo["origen"] < hasta)].dropna(subset=["y"])
    tr = todo[todo["target_at"] <= corte - pd.Timedelta(hours=reserva_h)].dropna()
    if dias:
        tr = tr[tr["target_at"] > corte - pd.Timedelta(days=dias)]
    p = crear().fit(tr).predict(b)
    return nombre, corte, reserva_h, b.index.to_numpy(), p


def simular(ancha, ctx, t0, reservas, procesos):
    global _TODO
    todo = construir(ancha, ctx, origenes_por_hora(ancha, ancha.index[0], ancha.index[-1]))
    todo["origen"] = todo["target_at"] - todo["horizon"] * PASO
    _TODO = todo
    fin = t0 + pd.Timedelta(hours=HORAS)
    cortes = list(pd.date_range(t0, fin - pd.Timedelta(hours=CADA_H), freq=f"{CADA_H}h"))
    tareas = [(n, c, r) for n in recetas() for c in cortes for r in reservas]
    # Congelado: la receta del champion entrenada en T0 y nunca más.
    crudas = {}
    with ProcessPoolExecutor(procesos, mp_context=mp.get_context("fork")) as ex:
        for nombre, corte, res, idx, p in ex.map(_bloque, tareas, chunksize=4):
            crudas[(nombre, corte, res)] = (idx, p)
    return todo, cortes, crudas


def entregar(todo, cortes, crudas, nombre, res, capas, congelado=False):
    """Lo entregado por una estrategia en los ciclos evaluados (serie por fila)."""
    partes = []
    for c in cortes:
        clave = (nombre, cortes[0] if congelado else c, res)
        idx0, p0 = crudas[clave]
        hasta = c + pd.Timedelta(hours=CADA_H)
        ini = c - pd.Timedelta(hours=CALENTAR_H)
        b = todo.loc[idx0]
        if congelado and c != cortes[0]:
            # el modelo de T0 predice también los bloques posteriores
            b = todo[(todo["origen"] >= ini) & (todo["origen"] < hasta)].dropna(subset=["y"])
            p0 = _MODELOS_CONGELADOS[res].predict(b)
        e = aplicar_capas(b, p0, capas)
        m = (b["origen"] >= c).to_numpy()
        partes.append(pd.Series(e[m], index=b.index[m]))
    return pd.concat(partes)


_MODELOS_CONGELADOS: dict = {}


def main(anclas, escenarios, reservas, procesos, semilla):
    from data import cargar
    cache = Path(os.environ.get("CACHE_ANCHA", ""))
    if cache.is_file():
        ancha, ctx = pd.read_pickle(cache), pd.read_pickle(cache.with_name("ctx.pkl"))
    else:
        ancha, ctx, _ = cargar(origen="supabase")
    tz = ancha.index.tz
    resultados = {}
    for ancla in anclas:
        t0 = pd.Timestamp(ancla).tz_localize(tz)
        for esc in escenarios:
            t = time.time()
            mod, rec = transformar(ancha, t0, esc, semilla)
            todo, cortes, crudas = simular(mod, ctx, t0, reservas, procesos)
            evaluados = todo[(todo["origen"] >= t0)
                             & (todo["origen"] < t0 + pd.Timedelta(hours=HORAS))].dropna(subset=["y"])
            horas = (evaluados["origen"] - t0) / pd.Timedelta(hours=1)
            tramos = {"adaptación 0-18 h": horas < ADAPTACION_H,
                      "estable 18-54 h": horas >= ADAPTACION_H, "todo": horas >= 0}

            # congelado: champion entrenado en T0 (sin reserva), nunca reentrenado
            crear, dias = recetas()["sc 14d (champion)"]
            tr = todo[todo["target_at"] <= t0].dropna()
            _MODELOS_CONGELADOS.clear()
            for r in reservas:
                _MODELOS_CONGELADOS[r] = crear().fit(tr[tr["target_at"] > t0 - pd.Timedelta(days=dias)])

            filas, entregas = {}, {}
            for capas_n, capas in CAPAS.items():
                for r in reservas:
                    estr = [(n, False) for n in recetas()] + [("sc 14d (champion)", True)]
                    for n, cong in estr:
                        etiqueta = (("congelado" if cong else n), r, capas_n)
                        s = entregar(todo, cortes, crudas, n, r, capas, congelado=cong)
                        entregas[etiqueta] = s
                        d = evaluados.assign(m=s.reindex(evaluados.index))
                        filas[etiqueta] = {k: accuracy_oficial(d[msk], "m") for k, msk in tramos.items()}
                    # selector: en cada corte, la receta con mejor accuracy entregada en 24 h
                    sel = []
                    for c in cortes:
                        o = evaluados["origen"]
                        pasado = evaluados[(o >= c - pd.Timedelta(hours=24)) & (o <= c - pd.Timedelta(hours=1))]
                        if len(pasado):
                            mejor = max(recetas(), key=lambda n: accuracy_oficial(
                                pasado.assign(m=entregas[(n, r, capas_n)].reindex(pasado.index)), "m"))
                        else:
                            mejor = "sc 14d (champion)"
                        blk = evaluados[(o >= c) & (o < c + pd.Timedelta(hours=CADA_H))]
                        sel.append(entregas[(mejor, r, capas_n)].reindex(blk.index))
                    s = pd.concat(sel)
                    d = evaluados.assign(m=s.reindex(evaluados.index))
                    filas[("selector 24 h", r, capas_n)] = {k: accuracy_oficial(d[msk], "m")
                                                            for k, msk in tramos.items()}
            tabla = pd.DataFrame(filas).T.round(2)
            tabla.index.names = ["estrategia", "reserva_h", "capas"]
            print(f"\n=== ancla {ancla}  escenario {esc}  ({time.time() - t:.0f}s)")
            print("transformación:", "; ".join(f"{k} {v}" for k, v in rec.items()))
            print(tabla.sort_values("estable 18-54 h", ascending=False).head(25).to_string())
            print("...\n" + tabla.loc[["congelado"]].to_string())
            resultados[f"{ancla}|{esc}"] = {"transformacion": rec,
                                             "tabla": {"|".join(map(str, k)): v for k, v in
                                                       tabla.to_dict(orient="index").items()}}

    # Resumen: promedio sobre anclas y escenarios, para no elegir por un caso.
    todas = pd.concat({k: pd.DataFrame(v["tabla"]).T for k, v in resultados.items()})
    media = todas.groupby(level=1).mean().round(2)
    print("\n=== PROMEDIO sobre anclas y escenarios")
    print(media.sort_values("estable 18-54 h", ascending=False).head(30).to_string())
    salida = RAIZ / "ml/resultados/experimento_forma.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "calculado": datetime.now(timezone.utc).isoformat(),
        "anclas": anclas, "escenarios": escenarios, "reservas": reservas,
        "semilla": semilla, "cada_h": CADA_H, "transicion_h": TRANSICION_H,
        "resultados": resultados, "promedio": media.to_dict(orient="index"),
    }, indent=2, ensure_ascii=False, default=float))
    print(f"\ndetalle -> {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anclas", nargs="+", default=ANCLAS)
    ap.add_argument("--escenarios", nargs="+", default=["corrimiento", "amplitud", "severo"])
    ap.add_argument("--reservas", nargs="+", type=int, default=[6, 2])
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=20260930)
    a = ap.parse_args()
    main(a.anclas, a.escenarios, a.reservas, a.procesos, a.semilla)
