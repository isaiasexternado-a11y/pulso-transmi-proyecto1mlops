"""Vuelca a MLflow los experimentos y el linaje completo de versiones del modelo.

Qué queda en MLflow
-------------------
- Experimento `pulso-experimentos`: un run por estudio de `ml/resultados/*.json`
  (backtest de baselines y candidatos, persistencia, nivel, quiebre, reentreno,
  forma, contexto, selector) con sus parámetros y métricas, y el JSON como
  artefacto. El backtest abre además un run hijo por modelo comparado.
- Experimento `pulso-versiones`: un run por versión registrada en la tabla
  `models` de Supabase (las 58 de la competencia), con algoritmo, ventana de
  entrenamiento, features, hiperparámetros (incluidas las capas), commit,
  ubicación y hash del artefacto, padre, estado, métricas de backtest por fold y
  métricas de producción (acumulada, 24 h, 7 d). La ficha va como artefacto.
- Model Registry `pulso-transmi-demanda`: una versión por modelo en orden de
  creación, con el estado como etiqueta y el alias `champion` en la activa.

Supabase sigue siendo la fuente de verdad operativa (lo exige el pipeline: el
runner no guarda estado); MLflow es la vista de experimentación y linaje. El
script es idempotente: borra y regenera el almacén cada vez.

Uso (lee SUPABASE_URL y SUPABASE_SECRET_KEY de .env o del entorno):

    python -m ml.registro_mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from collector.entorno import Supabase, cargar_env   # noqa: E402

DESTINO = RAIZ / "mlflow"
RESULTADOS = RAIZ / "ml" / "resultados"
REGISTRO = "pulso-transmi-demanda"
VENTANAS = {0: "acumulada", 1: "24h", 7: "7d"}


def limpio(nombre: str) -> str:
    """MLflow sólo acepta letras, números, _ - . espacio : / en las llaves."""
    return re.sub(r"[^A-Za-z0-9_\-. :/]", "_", nombre)[:250]


def aplanar(d, prefijo: str = "", profundidad: int = 0):
    """Hojas de un JSON como pares (ruta, valor). Las listas largas se omiten:
    son detalle por ciclo que ya está en el artefacto."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from aplanar(v, f"{prefijo}{k}.", profundidad + 1)
    elif isinstance(d, list):
        if len(d) <= 8 and all(not isinstance(x, (dict, list)) for x in d):
            yield prefijo[:-1], json.dumps(d)
        elif len(d) <= 40 and profundidad < 4:
            for i, x in enumerate(d):
                etiqueta = None
                if isinstance(x, dict):
                    etiqueta = next((str(x[k]) for k in ("modelo", "receta", "nombre", "escenario",
                                                         "variante", "ciclo") if k in x), None)
                yield from aplanar(x, f"{prefijo}{etiqueta or i}.", profundidad + 1)
    else:
        yield prefijo[:-1], d


def registrar_json(mlflow, ruta: Path) -> None:
    datos = json.loads(ruta.read_text())
    params, metricas = {}, {}
    for k, v in aplanar(datos):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            params[limpio(k)] = str(v)[:500]
        elif isinstance(v, (int, float)):
            metricas[limpio(k)] = float(v)
    with mlflow.start_run(run_name=ruta.stem) as run:
        mlflow.set_tag("estudio", ruta.stem)
        mlflow.set_tag("codigo", f"ml/{ruta.stem}.py")
        if isinstance(datos, dict) and "calculado" in datos:
            mlflow.set_tag("calculado", datos["calculado"])
        # MLflow limita los params por lote; los que sobran quedan en el artefacto.
        for i, (k, v) in enumerate(sorted(params.items())):
            if i >= 300:
                break
            mlflow.log_param(k, v)
        if metricas:
            mlflow.log_metrics(dict(list(sorted(metricas.items()))[:1000]))
        mlflow.log_artifact(str(ruta), artifact_path="resultados")

        if ruta.stem == "backtest" and isinstance(datos, dict):
            for fila in datos.get("tabla", []):
                with mlflow.start_run(run_name=fila["modelo"], nested=True):
                    mlflow.set_tag("es_baseline", str(fila.get("es_baseline")))
                    mlflow.log_metric("accuracy_media", fila["accuracy_media"])
                    mlflow.log_metric("accuracy_min", fila["accuracy_min"])
                    for i, v in enumerate(fila.get("por_fold") or [], start=1):
                        mlflow.log_metric(f"accuracy_fold{i}", v)
                    for h, v in (fila.get("por_horizonte") or {}).items():
                        mlflow.log_metric(limpio(f"accuracy_h{h}"), v)
    print(f"estudio  : {ruta.stem}  ({len(params)} params, {len(metricas)} métricas)  {run.info.run_id}")


def traer(sb: Supabase, tabla: str, **params) -> list[dict]:
    filas, desde = [], 0
    while True:
        lote = sb.seleccionar(tabla, offset=desde, limit=1000, **params)
        filas += lote
        if len(lote) < 1000:
            return filas
        desde += 1000


def registrar_versiones(mlflow, sb: Supabase) -> None:
    from mlflow import MlflowClient

    modelos = traer(sb, "models", select="*", order="created_at.asc")
    metricas = traer(sb, "model_metrics", select="model_id,split,fold,metric,value,window_end",
                     station_id="is.null")
    por_modelo: dict[str, dict[str, float]] = {}
    for m in metricas:
        if m["split"] == "backtest":
            k = f"backtest_{m['metric']}_fold{m['fold']}"
        else:
            k = f"produccion_{m['metric']}_{VENTANAS.get(m['fold'], m['fold'])}"
        por_modelo.setdefault(m["model_id"], {})[k] = float(m["value"])

    cliente = MlflowClient()
    cliente.create_registered_model(
        REGISTRO, description="Pronóstico de demanda a +15/+30/+45/+60 min en 12 estaciones. "
                              "Una versión por modelo registrado en Supabase `models`.")
    ids_mlflow: dict[str, str] = {}
    for ficha in modelos:
        hp = ficha.get("hyperparams") or {}
        version = hp.get("version") or ficha["model_id"]
        with mlflow.start_run(run_name=ficha["name"][:200]) as run:
            ids_mlflow[ficha["model_id"]] = run.info.run_id
            mlflow.set_tags({
                "model_id": ficha["model_id"], "version": version, "status": ficha["status"],
                "kind": ficha["kind"], "git_commit": ficha.get("git_commit") or "",
                "artifact_uri": ficha.get("artifact_uri") or "",
                "artifact_sha256": ficha.get("artifact_sha256") or "",
                "parent_model_id": ficha.get("parent_model_id") or "",
                "parent_run_id": ids_mlflow.get(ficha.get("parent_model_id") or "", ""),
                "activated_at": ficha.get("activated_at") or "",
                "retired_at": ficha.get("retired_at") or "",
                "created_at": ficha["created_at"],
            })
            params = {"algorithm": ficha.get("algorithm"), "feature_set": ficha.get("feature_set"),
                      "train_start": ficha.get("train_start"), "train_end": ficha.get("train_end"),
                      "n_train_rows": ficha.get("n_train_rows"),
                      "n_features": len(ficha.get("feature_list") or [])}
            for k, v in aplanar(hp, "hp."):
                params[limpio(k)] = v
            mlflow.log_params({k: str(v)[:500] for k, v in list(params.items())[:300]})
            if por_modelo.get(ficha["model_id"]):
                pm = por_modelo[ficha["model_id"]]
                folds = [v for k, v in pm.items() if k.startswith("backtest_accuracy_fold")]
                if folds:
                    pm["backtest_accuracy_media"] = sum(folds) / len(folds)
                mlflow.log_metrics(pm)
            mlflow.log_dict(ficha, "ficha/ficha.json")
            fuente = f"{run.info.artifact_uri}/ficha"
        mv = cliente.create_model_version(REGISTRO, source=fuente, run_id=run.info.run_id,
                                          description=f"{ficha['name']} ({version})",
                                          tags={"status": ficha["status"], "model_id": ficha["model_id"]})
        if ficha["status"] == "active":
            cliente.set_registered_model_alias(REGISTRO, "champion", mv.version)
        print(f"versión {mv.version:>2}: {ficha['status']:<9} {version}")


def relativizar() -> None:
    """MLflow guarda las rutas de artefactos absolutas aunque se le den
    relativas. Se reescriben relativas a la raíz del repo para que el almacén
    versionado abra en cualquier clon (`mlflow ui` se lanza desde la raíz)."""
    import sqlite3

    prefijo = f"{RAIZ}/"
    con = sqlite3.connect(DESTINO / "mlflow.db")
    for tabla, col in [("runs", "artifact_uri"), ("experiments", "artifact_location"),
                       ("model_versions", "source"), ("model_versions", "storage_location")]:
        cols = {c[1] for c in con.execute(f"pragma table_info({tabla})")}
        if col in cols:
            con.execute(f"update {tabla} set {col} = replace({col}, ?, '')", (prefijo,))
    con.execute("update experiments set artifact_location = 'mlflow/artefactos/0' "
                "where experiment_id = '0'")
    con.commit()
    con.close()
    shutil.rmtree(RAIZ / "mlruns", ignore_errors=True)


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    cargar_env()
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir()
    # Rutas relativas a la raíz del repo: el almacén se versiona y debe abrirse
    # igual en cualquier clon.
    os.chdir(RAIZ)
    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow/mlflow.db")

    for nombre, desc in [("pulso-experimentos", "Estudios con validación temporal (ml/resultados)"),
                         ("pulso-versiones", "Linaje de versiones registradas en Supabase models")]:
        mlflow.create_experiment(nombre, artifact_location=f"mlflow/artefactos/{nombre}",
                                 tags={"mlflow.note.content": desc})

    mlflow.set_experiment("pulso-experimentos")
    for ruta in sorted(RESULTADOS.glob("*.json")):
        registrar_json(mlflow, ruta)

    mlflow.set_experiment("pulso-versiones")
    registrar_versiones(mlflow, Supabase())
    relativizar()
    print(f"\nlisto: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db  (desde {RAIZ})")


if __name__ == "__main__":
    main()
