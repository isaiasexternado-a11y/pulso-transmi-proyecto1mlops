"""Verifica un candidato con el código de producción y lo promueve.

Entrenar no es promover. Un candidato puede tener métricas excelentes en la
rama de experimentos y aun así no servir: el artefacto viaja como pickle, y
quien lo carga en producción es el código de `main`, no el que lo entrenó.

Por eso este archivo vive en `main` y usa `entregar.cargar_artefacto`, el mismo
camino exacto por el que el champion entra a la inferencia. Verificar con un
cargador distinto probaría la cosa equivocada: pasaría la prueba y fallaría el
primer ciclo real.

La regla que hace esto posible: un candidato puede cambiar HIPERPARÁMETROS,
VENTANA y LISTA DE FEATURES, pero no puede introducir una clase nueva. Al
deserializar, Python restaura el `__dict__` sin llamar a `__init__`, así que
una instancia entrenada con atributos distintos funciona con la definición de
clase que tenga `main`. Una clase que `main` no conoce, en cambio, revienta.

Uso:
    python3 -m pipeline.promover --model-id <uuid>
    python3 -m pipeline.promover --model-id <uuid> --dry-run
    python3 -m pipeline.promover --rollback     # devuelve el champion anterior
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
# `ml/features.py` importa `data` plano, así que `ml/` tiene que estar en el
# path ANTES de importarlo —y también para que joblib encuentre `models` al
# deserializar—. Se hace aquí, a nivel de módulo, y no enterrado dentro de una
# función: depender del orden de los imports es una trampa esperando a alguien.
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase        # noqa: E402
from pipeline.entregar import cargar_artefacto, predecir  # noqa: E402

HORIZONTES = (1, 2, 3, 4)


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def ficha_de(sb: Supabase, model_id: str) -> dict:
    filas = sb.seleccionar("models", select="*", model_id=f"eq.{model_id}", limit=1)
    if not filas:
        raise SystemExit(f"no existe el modelo {model_id}")
    return filas[0]


def inferencia_de_prueba(sb: Supabase, ficha: dict) -> tuple[bool, str]:
    """¿Sabe predecir, cargado como lo cargaría la inferencia real?

    No mide calidad —para eso está el backtest— sino que el artefacto viaje
    entero: que la huella calce, que el pickle abra con las clases de `main` y
    que salgan 48 valores finitos y no negativos.
    """
    import pandas as pd
    from ml.data import cargar

    modelo = cargar_artefacto(sb, ficha)          # el camino de producción
    ancha, ctx, _ = cargar(origen="supabase")

    origen = ancha.index[-1]
    paso = ancha.index[1] - ancha.index[0]
    objetivos = [(est, origen + h * paso)
                 for h in HORIZONTES for est in ancha.columns]

    pred = predecir(modelo, ficha, ancha, ctx, origen, objetivos)

    esperados = len(ancha.columns) * len(HORIZONTES)
    serie = pd.Series(pred, dtype="float64")
    problemas = []
    if len(serie) != esperados:
        problemas.append(f"{len(serie)} valores en vez de {esperados}")
    if not serie.notna().all():
        problemas.append(f"{int(serie.isna().sum())} valores no finitos")
    if (serie < 0).any():
        problemas.append(f"{int((serie < 0).sum())} valores negativos")

    detalle = (f"{len(serie)} valores · min {serie.min():.1f} · "
               f"max {serie.max():.1f} · media {serie.mean():.1f}")
    if problemas:
        return False, detalle + "  ->  " + "; ".join(problemas)
    return True, detalle


def promover(sb: Supabase, ficha: dict, cierra_decision: bool = True) -> None:
    """Un solo `active` a la vez. El saliente queda `retired`, no borrado:
    sin el anterior disponible no hay rollback posible."""
    salientes = sb.seleccionar("models", select="model_id,name",
                               status="eq.active")
    for viejo in salientes:
        if viejo["model_id"] == ficha["model_id"]:
            continue
        sb.actualizar("models", {"model_id": f"eq.{viejo['model_id']}"},
                      {"status": "retired", "retired_at": ahora()})
        print(f"retirado : {viejo['name']}  ({viejo['model_id']})")

    sb.actualizar("models", {"model_id": f"eq.{ficha['model_id']}"},
                  {"status": "active", "activated_at": ahora()})
    print(f"promovido: {ficha['name']}  ({ficha['model_id']})")

    # La decisión que originó este entrenamiento queda cerrada con el modelo
    # que produjo. Sin esto, `retrain_decisions` diría por qué se reentrenó
    # pero no en qué terminó. Un rollback no responde a ninguna decisión: la
    # cerraría con el modelo que la decisión pedía reemplazar.
    if not cierra_decision:
        return
    ultima = sb.seleccionar("retrain_decisions", select="run_id,decision",
                            decision="eq.retrain", new_model_id="is.null",
                            order="decided_at.desc", limit=1)
    if ultima:
        sb.actualizar("retrain_decisions", {"run_id": f"eq.{ultima[0]['run_id']}"},
                      {"new_model_id": ficha["model_id"]})
        print(f"decisión {ultima[0]['run_id']} cerrada con el modelo nuevo")


def rollback(sb: Supabase, dry_run: bool) -> None:
    """Devuelve el champion al último retirado. Es la salida de emergencia
    cuando un modelo promovido resulta peor en producción de lo que prometía
    en validación."""
    previos = sb.seleccionar("models", select="*", status="eq.retired",
                             order="retired_at.desc", limit=1)
    if not previos:
        raise SystemExit("no hay ningún modelo retirado al que volver")
    ficha = previos[0]
    print(f"candidato a rollback: {ficha['name']}  "
          f"({ficha['hyperparams'].get('version')})")

    ok, detalle = inferencia_de_prueba(sb, ficha)
    print(f"inferencia de prueba: {detalle}")
    if not ok:
        raise SystemExit("el modelo anterior tampoco pasa la prueba; no se revierte")
    if dry_run:
        print("\n[dry-run] no se revirtió nada")
        return
    promover(sb, ficha, cierra_decision=False)
    print("ROLLBACK COMPLETO")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()

    sb = Supabase()

    if a.rollback:
        rollback(sb, a.dry_run)
        return
    if not a.model_id:
        raise SystemExit("hace falta --model-id (o --rollback)")

    ficha = ficha_de(sb, a.model_id)
    print(f"candidato: {ficha['name']}  ({ficha['hyperparams'].get('version')})")
    print(f"estado   : {ficha['status']}")
    if ficha["status"] == "active":
        print("ya está activo; nada que hacer")
        return
    if ficha["status"] != "candidate":
        raise SystemExit(f"sólo se promueve desde 'candidate', no desde "
                         f"'{ficha['status']}'")

    ok, detalle = inferencia_de_prueba(sb, ficha)
    print(f"inferencia de prueba: {detalle}")

    if not ok:
        if not a.dry_run:
            sb.actualizar("models", {"model_id": f"eq.{a.model_id}"},
                          {"status": "rejected"})
            print("marcado como rejected")
        raise SystemExit("la inferencia de prueba falló: NO se promueve")

    if a.dry_run:
        print("\n[dry-run] pasa la prueba, no se promovió")
        return

    promover(sb, ficha)
    print("PROMOVIDO")


if __name__ == "__main__":
    main()
