"""Entrena candidatos, los compara temporalmente y registra al ganador.

Este archivo NO promueve. Deja al ganador en `models` con `status='candidate'`
y ahí termina su trabajo: quien decide si entra a producción es
`pipeline/promover.py`, que corre desde `main` y lo carga con el mismo código
que usa la inferencia. Entrenar y promover son dos preguntas distintas —"¿es
mejor?" y "¿funciona donde va a vivir?"— y mezclarlas es cómo se promueven
modelos que miden bien y fallan el primer ciclo.

Qué se compara y por qué
------------------------
Bajo drift la decisión que más pesa no es el algoritmo sino la VENTANA. Un
modelo entrenado con 45 días donde 44 son anteriores al cambio de patrón
aprende sobre todo el patrón viejo. Por eso cada familia se prueba con varias
ventanas y gana la que gane, no la que "debería".

La segunda decisión es el CONTEXTO. Las cinco variables de clima y evento
llevan congeladas desde el 2026-09-08 y hoy son una constante arrastrada. Un
candidato sin ellas no es una poda cosmética: es quitarle al modelo una
variable que finge informar.

Y el perfil se queda siempre en la comparación. No necesita entrenarse, se
recalcula solo con datos nuevos, y bajo drift fuerte eso lo vuelve un rival
serio, no un trámite.

Restricción de compatibilidad
-----------------------------
Un candidato puede cambiar hiperparámetros, ventana y lista de features, pero
NO puede introducir una clase que `main` no conozca. El artefacto viaja como
pickle y lo deserializa el código de `main`; al hacerlo se restaura el
`__dict__` sin llamar a `__init__`, así que atributos distintos funcionan y una
clase desconocida no. Por eso `models.py` parametriza en vez de subclasear.

Uso:
    python3 -m ml.entrenar --dry-run   # compara y reporta, no registra nada
    python3 -m ml.entrenar             # registra al ganador como candidate
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
# `features.py` importa `data` plano, y el pickle referencia `models` como
# módulo de primer nivel: ml/ tiene que estar en el path antes de importarlos.
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase                       # noqa: E402
from collector.recolectar import procedencia                 # noqa: E402
from data import cargar                                      # noqa: E402
from features import (COLS_NUM, COLS_SIN_CONTEXTO,           # noqa: E402
                      construir, origenes_por_hora)
from metrics import resumen                                  # noqa: E402
from models import Columna, GBMconPerfil, Perfil             # noqa: E402

BUCKET = "modelos"
DIAS_VALIDACION = 7
N_FOLDS = 3

# --------------------------------------------------------------- compuertas
# Un candidato no reemplaza al champion por empatarle. El margen existe para
# que el ruido de tres folds no se disfrace de mejora.
MARGEN_CHAMPION = 0.15      # puntos de accuracy sobre el champion
TOLERANCIA_ESTACION = 2.0   # ninguna estación puede empeorar más que esto


# ------------------------------------------------------------------ recetas

class Receta:
    """Una forma de entrenar: qué modelo, con qué features, con qué ventana."""

    def __init__(self, etiqueta, crear, dias=None, es_champion=False):
        self.etiqueta = etiqueta
        self.crear = crear
        self.dias = dias                 # None = todo el histórico
        self.es_champion = es_champion


def recetas() -> list[Receta]:
    sc = COLS_SIN_CONTEXTO
    return [
        # El champion vigente, con su receta exacta. Es la vara: sin medirlo
        # en los mismos folds no hay forma honesta de decir que otro es mejor.
        Receta("gbm + perfil (mae)",
               lambda: GBMconPerfil("gbm + perfil (mae)"), None, es_champion=True),

        # Sin las cinco variables congeladas, a varias ventanas.
        Receta("gbm + perfil sin contexto",
               lambda: GBMconPerfil("gbm + perfil sin contexto (mae)", cols=sc), None),
        Receta("gbm + perfil sin contexto · 14d",
               lambda: GBMconPerfil("gbm + perfil sin contexto (mae)", cols=sc), 14),
        Receta("gbm + perfil sin contexto · 7d",
               lambda: GBMconPerfil("gbm + perfil sin contexto (mae)", cols=sc), 7),

        # Con contexto pero ventana corta: separa "sobra el contexto" de
        # "sobran los datos viejos". Si gana ésta, el problema era la ventana.
        Receta("gbm + perfil · 14d",
               lambda: GBMconPerfil("gbm + perfil (mae)"), 14),

        # Baselines. El perfil es el piso real del reto (88,11 %) y se
        # recalcula sin entrenar, así que bajo drift es un rival de verdad.
        Receta("perfil", lambda: Perfil(), None),
        Receta("perfil · 14d", lambda: Perfil(), 14),
        Receta("naive s-1", lambda: Columna("naive s-1", "lag_sem"), None),
    ]


# ------------------------------------------------------------ validación temporal

def folds(ancha: pd.DataFrame, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas de validación consecutivas, siempre hacia adelante.

    Nunca un split aleatorio: mezclar futuro y pasado infla los resultados y
    elegiría el modelo equivocado.
    """
    fin = ancha.index[-1]
    out = []
    for k in range(n, 0, -1):
        val_fin = fin - pd.to_timedelta(DIAS_VALIDACION * (k - 1), unit="D")
        val_ini = val_fin - pd.to_timedelta(DIAS_VALIDACION, unit="D")
        out.append((val_ini, val_fin))
    return out


def comparar(ancha, ctx, n_folds: int) -> dict[str, dict]:
    todas = recetas()
    acum: dict[str, list] = {r.etiqueta: [] for r in todas}

    for i, (val_ini, val_fin) in enumerate(folds(ancha, n_folds), 1):
        o_train = origenes_por_hora(ancha, ancha.index[0], val_ini)
        # Ningún origen de entrenamiento puede alcanzar la ventana de
        # validación con sus horizontes: ahí es donde se cuela la fuga.
        o_train = o_train[o_train + 4 < ancha.index.get_loc(val_ini)]
        o_val = origenes_por_hora(ancha, val_ini, val_fin)
        if len(o_train) == 0 or len(o_val) == 0:
            print(f"fold {i}: sin orígenes suficientes, se salta")
            continue

        # Se construye UNA vez el frame completo y las ventanas cortas se
        # recortan de ahí. Reconstruir por receta multiplicaría el trabajo.
        train_full = construir(ancha, ctx, o_train).dropna()
        val = construir(ancha, ctx, o_val).dropna(subset=["y"])
        print(f"\nfold {i}: train {len(train_full):,} filas  |  "
              f"val {len(val):,} filas  ({val_ini:%m-%d} -> {val_fin:%m-%d})")

        for r in todas:
            tr = train_full
            if r.dias is not None:
                corte = val_ini - pd.to_timedelta(r.dias, unit="D")
                tr = train_full[train_full["target_at"] >= corte]
            if len(tr) < 500:
                print(f"   {r.etiqueta:38} sin datos suficientes ({len(tr)})")
                continue

            t0 = time.time()
            m = r.crear().fit(tr)
            res = resumen(val.assign(y_pred=m.predict(val)))
            res["segundos"] = round(time.time() - t0, 1)
            res["n_train"] = int(len(tr))
            acum[r.etiqueta].append(res)
            print(f"   {r.etiqueta:38} accuracy {res['accuracy']:6.2f}  "
                  f"peor est {res['peor_estacion']:6.2f}  "
                  f"({len(tr):,} filas, {res['segundos']:.0f}s)")

    tabla = {}
    for r in todas:
        rs = acum[r.etiqueta]
        if not rs:
            continue
        accs = [x["accuracy"] for x in rs]
        estaciones = {}
        for est in rs[-1]["por_estacion"]:
            vals = [x["por_estacion"].get(est) for x in rs
                    if x["por_estacion"].get(est) is not None]
            estaciones[est] = round(sum(vals) / len(vals), 2)
        tabla[r.etiqueta] = {
            "receta": r,
            "accuracy_media": round(sum(accs) / len(accs), 2),
            "accuracy_min": round(min(accs), 2),
            "por_fold": accs,
            "por_estacion": estaciones,
            "por_horizonte": rs[-1]["por_horizonte"],
            "folds": rs,
        }
    return tabla


# ------------------------------------------------------------------ compuertas

def elegir(tabla: dict) -> tuple[str | None, list[str]]:
    """Devuelve (etiqueta ganadora o None, razones)."""
    champion = next((k for k, v in tabla.items() if v["receta"].es_champion), None)
    if champion is None:
        return None, ["no se pudo medir la receta del champion; sin vara no hay comparación"]

    ref = tabla[champion]
    rivales = sorted((v for k, v in tabla.items() if k != champion),
                     key=lambda v: v["accuracy_media"], reverse=True)
    if not rivales:
        return None, ["no hubo rivales medibles"]

    mejor_base = max((v for v in tabla.values() if v["receta"].crear().es_baseline),
                     key=lambda v: v["accuracy_media"], default=None)

    razones = []
    for cand in rivales:
        etiqueta = cand["receta"].etiqueta
        fallos = []

        margen = cand["accuracy_media"] - ref["accuracy_media"]
        if margen < MARGEN_CHAMPION:
            fallos.append(f"gana al champion por {margen:+.2f}, "
                          f"se exigen {MARGEN_CHAMPION:+.2f}")

        if mejor_base and cand["accuracy_media"] <= mejor_base["accuracy_media"] \
                and not cand["receta"].crear().es_baseline:
            fallos.append(f"no le gana al mejor baseline "
                          f"({mejor_base['accuracy_media']:.2f})")

        # Estabilidad, no sólo promedio: un modelo que sube la media hundiendo
        # una estación empeora la métrica oficial, que no pondera por volumen.
        if cand["accuracy_min"] < ref["accuracy_min"]:
            fallos.append(f"su peor fold ({cand['accuracy_min']:.2f}) es peor "
                          f"que el del champion ({ref['accuracy_min']:.2f})")

        hundidas = [e for e, v in cand["por_estacion"].items()
                    if ref["por_estacion"].get(e, 0) - v > TOLERANCIA_ESTACION]
        if hundidas:
            fallos.append(f"empeora más de {TOLERANCIA_ESTACION} puntos en "
                          f"{len(hundidas)} estación(es): {', '.join(hundidas[:3])}")

        if not fallos:
            razones.append(f"{etiqueta}: pasa todas las compuertas "
                           f"({margen:+.2f} sobre el champion)")
            return etiqueta, razones
        razones.append(f"{etiqueta}: {'; '.join(fallos)}")

    return None, razones


# -------------------------------------------------------------------- registro

def registrar(sb: Supabase, ancha, ctx, ganador: dict, tabla: dict) -> dict:
    """Reentrena al ganador con todo lo disponible, lo sube y lo inscribe."""
    r = ganador["receta"]
    fin = ancha.index[-1]
    o_full = origenes_por_hora(ancha, ancha.index[0], fin)
    full = construir(ancha, ctx, o_full).dropna()
    if r.dias is not None:
        full = full[full["target_at"] >= fin - pd.to_timedelta(r.dias, unit="D")]

    modelo = r.crear().fit(full)
    print(f"\nreentrenado con {len(full):,} filas "
          f"(ventana {r.dias or 'completa'})")

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cols = list(getattr(modelo, "cols_", COLS_NUM))
    buf = io.BytesIO()
    joblib.dump({"modelo": modelo, "cols": cols, "version": version},
                buf, compress=3)
    crudo = buf.getvalue()
    sha = hashlib.sha256(crudo).hexdigest()

    ruta = f"champion/champion_{version}.joblib"
    sb.subir(BUCKET, ruta, crudo)
    print(f"artefacto: supabase://{BUCKET}/{ruta}  "
          f"({len(crudo)/1e6:.2f} MB, sha {sha[:12]}...)")

    run = sb.insertar("pipeline_runs", [{
        **procedencia(), "status": "success",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }], devolver=True)[0]

    previo = sb.seleccionar("models", select="model_id", status="eq.active", limit=1)
    sin_contexto = not any(c.startswith(("rain", "temp", "evento")) for c in cols)

    fila = {
        "name": modelo.nombre,
        "kind": "baseline" if modelo.es_baseline else "ml",
        "algorithm": type(modelo).__name__,
        "feature_set": "fs-v2-sin-contexto" if sin_contexto else "fs-v1",
        "feature_list": cols,
        "train_start": str(full["target_at"].min()),
        "train_end": str(fin),
        "n_train_rows": int(len(full)),
        "artifact_uri": f"supabase://{BUCKET}/{ruta}",
        "artifact_sha256": sha,
        "git_commit": os.environ.get("GITHUB_SHA", "local"),
        "trained_by_run_id": run["run_id"],
        "parent_model_id": previo[0]["model_id"] if previo else None,
        "status": "candidate",
        "hyperparams": {
            # `version` no es decorativo: entregar.py lo lee para saber si un
            # ciclo ya tiene recibo de ESTA versión del champion.
            "version": version,
            "sklearn": __import__("sklearn").__version__,
            "ventana_dias": r.dias,
            "receta": r.etiqueta,
            "validacion_accuracy": ganador["accuracy_media"],
            "validacion_peor_fold": ganador["accuracy_min"],
            "por_fold": ganador["por_fold"],
            "champion_previo_accuracy": next(
                (v["accuracy_media"] for v in tabla.values()
                 if v["receta"].es_champion), None),
            "folds": N_FOLDS,
            "dias_validacion": DIAS_VALIDACION,
        },
    }
    creado = sb.insertar("models", [fila], devolver=True)[0]
    print(f"registrado: {creado['model_id']}  status=candidate")

    # La evidencia de la comparación queda en la tabla, no sólo en el log.
    ahora = datetime.now(timezone.utc).isoformat()
    metricas = []
    for i, res in enumerate(ganador["folds"], 1):
        for est, acc in res["por_estacion"].items():
            metricas.append({"model_id": creado["model_id"], "split": "backtest",
                             "fold": i, "station_id": est, "metric": "accuracy",
                             "value": float(acc), "computed_at": ahora})
        metricas.append({"model_id": creado["model_id"], "split": "backtest",
                         "fold": i, "station_id": None, "metric": "accuracy",
                         "value": float(res["accuracy"]), "computed_at": ahora})
        metricas.append({"model_id": creado["model_id"], "split": "backtest",
                         "fold": i, "station_id": None, "metric": "mae",
                         "value": float(res["mae"]), "computed_at": ahora})
    for i in range(0, len(metricas), 500):
        sb.upsert("model_metrics", metricas[i:i + 500],
                  conflicto="model_id,split,fold,station_id,metric")
    print(f"métricas : {len(metricas)} filas de backtest")
    return creado


def anunciar(**kv) -> None:
    destino = os.environ.get("GITHUB_OUTPUT")
    if not destino:
        return
    with open(destino, "a", encoding="utf-8") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


# ------------------------------------------------------------------------ main

def main(dry_run: bool, n_folds: int) -> None:
    sb = Supabase()
    ancha, ctx, _ = cargar(origen="supabase")
    print(f"panel: {ancha.shape[0]:,} periodos x {ancha.shape[1]} estaciones "
          f"({ancha.index[0]:%Y-%m-%d} -> {ancha.index[-1]:%Y-%m-%d %H:%M})")

    tabla = comparar(ancha, ctx, n_folds)

    print("\n" + "=" * 76)
    print(f"{'receta':40} {'media':>7} {'peor fold':>10}")
    print("-" * 76)
    for k, v in sorted(tabla.items(), key=lambda x: -x[1]["accuracy_media"]):
        marca = "<- champion" if v["receta"].es_champion else ""
        print(f"{k:40} {v['accuracy_media']:7.2f} {v['accuracy_min']:10.2f}  {marca}")
    print("=" * 76)

    etiqueta, razones = elegir(tabla)
    print("\ncompuertas:")
    for r in razones:
        print(f"  · {r}")

    if etiqueta is None:
        print("\nNINGÚN CANDIDATO PASA. El champion se queda.")
        anunciar(model_id="", version="")
        return

    print(f"\nGANADOR: {etiqueta}")
    if dry_run:
        print("[dry-run] no se registró nada")
        anunciar(model_id="", version="")
        return

    creado = registrar(sb, ancha, ctx, tabla[etiqueta], tabla)
    anunciar(model_id=creado["model_id"],
             version=creado["hyperparams"]["version"])
    print("\nlisto: el candidato queda registrado, NO promovido. "
          "La promoción la decide pipeline/promover.py desde main.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    main(a.dry_run, a.folds)
