"""Empaqueta el champion como artefacto versionado.

La ficha del modelo la exige la guía: identificador, corte de datos, commit,
features, métrica de validación, ubicación del artefacto y estado. Aquí se
genera todo junto para que un artefacto nunca quede sin explicación de dónde
salió.

El nombre del archivo lleva versión: un artefacto nuevo no pisa al anterior.

Uso:
    python3 ml/empaquetar.py
    python3 ml/empaquetar.py --modelo "perfil (estacion x slot x finde)"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import cargar                              # noqa: E402
from features import COLS_NUM, construir, origenes_por_hora  # noqa: E402
from metrics import resumen                          # noqa: E402
from models import candidatos                        # noqa: E402

DIR_ARTEFACTOS = Path("ml/artefactos")
DIAS_VALIDACION = 7


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "desconocido"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def empaquetar(nombre_modelo: str = "gbm + perfil (mae)") -> dict:
    ancha, ctx, _ = cargar()
    fin = ancha.index[-1]
    corte_val = fin - pd.Timedelta(f"{DIAS_VALIDACION}D")

    modelo = next((m for m in candidatos() if m.nombre == nombre_modelo), None)
    if modelo is None:
        raise SystemExit(f"no existe el modelo {nombre_modelo!r}")

    # 1. Validación honesta: entrenar sin la última semana y medir sobre ella.
    o_tr = origenes_por_hora(ancha, ancha.index[0], corte_val)
    o_tr = o_tr[o_tr + 4 < ancha.index.get_loc(corte_val)]
    o_val = origenes_por_hora(ancha, corte_val, fin)
    tr = construir(ancha, ctx, o_tr).dropna()
    val = construir(ancha, ctx, o_val).dropna(subset=["y"])
    modelo.fit(tr)
    metricas = resumen(val.assign(y_pred=modelo.predict(val)))
    print(f"validacion (ultimos {DIAS_VALIDACION} dias): accuracy {metricas['accuracy']:.2f}")

    # 2. El artefacto que se promueve se reentrena con TODO el histórico:
    #    tirar la última semana en producción sería desperdiciar datos.
    o_full = origenes_por_hora(ancha, ancha.index[0], fin)
    full = construir(ancha, ctx, o_full).dropna()
    modelo.fit(full)
    print(f"reentrenado con todo el historico: {len(full):,} filas")

    # 3. Inferencia de prueba: la guía exige que un champion sepa predecir
    #    antes de promoverse, no solo tener buenas métricas.
    ultimo_origen = o_full[-1:]
    prueba = construir(ancha, ctx, ultimo_origen)
    pred = modelo.predict(prueba)
    ok = (len(pred) == 48 and pd.notna(pred).all() and (pred >= 0).all())
    print(f"inferencia de prueba: {len(pred)} valores, "
          f"min {pred.min():.1f} max {pred.max():.1f} -> {'OK' if ok else 'FALLO'}")
    if not ok:
        raise SystemExit("la inferencia de prueba falló: no se promueve")

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_ARTEFACTOS / f"champion_{version}.joblib"

    joblib.dump({
        "modelo": modelo,
        "cols": COLS_NUM + (["perfil"] if hasattr(modelo, "tabla_") else []),
        "version": version,
    }, ruta, compress=3)

    ficha = {
        "version": version,
        "name": nombre_modelo,
        "kind": "baseline" if modelo.es_baseline else "ml",
        "algorithm": type(modelo).__name__,
        "feature_set": "fs-v1",
        "feature_list": COLS_NUM + ["perfil"],
        "train_start": str(ancha.index[0]),
        "train_end": str(fin),
        "n_train_rows": int(len(full)),
        "git_commit": git_commit(),
        "artifact_uri": str(ruta),
        "artifact_sha256": sha256(ruta),
        "artifact_bytes": ruta.stat().st_size,
        "sklearn": __import__("sklearn").__version__,
        "status": "candidate",
        "validacion": metricas,
        "inferencia_prueba": {"n": int(len(pred)), "ok": bool(ok)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (DIR_ARTEFACTOS / f"champion_{version}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False))

    print(f"\nartefacto : {ruta}  ({ruta.stat().st_size / 1e6:.2f} MB)")
    print(f"sha256    : {ficha['artifact_sha256'][:16]}...")
    print(f"ficha     : {DIR_ARTEFACTOS / f'champion_{version}.json'}")
    return ficha


def cargar_artefacto(ruta: str | Path):
    """Carga un artefacto. Requiere que `ml/` esté en el path."""
    return joblib.load(ruta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gbm + perfil (mae)")
    empaquetar(ap.parse_args().modelo)
