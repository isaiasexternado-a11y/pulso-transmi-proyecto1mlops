"""Consulta el ciclo vigente, predice y envía el batch.

El orden importa y no es negociable: primero se le pregunta a la API qué quiere,
después se predice exactamente eso. Nunca al revés. La lista de targets que
devuelve el ciclo es el contrato; fabricar horizontes o timestamps por cuenta
propia hace que la API rechace el lote completo.

Uso:
    python3 ml/enviar.py --dry-run     # arma el payload y lo muestra, sin enviar
    python3 ml/enviar.py               # envía de verdad
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import cargar                              # noqa: E402
from features import construir_para_objetivos        # noqa: E402

DIR_RECIBOS = Path("ml/recibos")


def entorno() -> tuple[str, str]:
    for linea in Path(".env").read_text().splitlines():
        if linea.strip() and not linea.startswith("#") and "=" in linea:
            k, v = linea.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    base = os.environ["PULSO_API_BASE"].rstrip("/")
    key = os.environ["PULSO_API_KEY"]
    return base, key


def pedir(url: str, key: str, metodo="GET", cuerpo=None, extra=None) -> dict:
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    cab = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if datos:
        cab["Content-Type"] = "application/json"
    cab.update(extra or {})
    req = urllib.request.Request(url, data=datos, method=metodo, headers=cab)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"status": r.status, "body": json.loads(r.read() or b"{}")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": json.loads(e.read() or b"{}")}


def ultimo_artefacto() -> tuple[object, dict]:
    rutas = sorted(glob.glob("ml/artefactos/champion_*.joblib"))
    if not rutas:
        raise SystemExit("no hay artefacto; corre primero ml/empaquetar.py")
    paq = joblib.load(rutas[-1])
    ficha = json.loads(Path(rutas[-1].replace(".joblib", ".json")).read_text())
    return paq["modelo"], ficha


def main(dry_run: bool) -> None:
    base, key = entorno()

    ciclo = pedir(f"{base}/v1/forecast-cycles/current", key)["body"]
    if ciclo.get("state") != "open":
        raise SystemExit(f"no hay ciclo abierto (state={ciclo.get('state')})")

    cierra = pd.Timestamp(ciclo["closes_at"])
    restante = cierra - pd.Timestamp.now(tz="UTC")
    print(f"ciclo   : {ciclo['cycle_id']}  ({ciclo['state']})")
    print(f"cutoff  : {ciclo['data_cutoff']}")
    print(f"cierra  : {ciclo['closes_at']}  (faltan {str(restante).split('.')[0]})")
    print(f"pide    : {ciclo['expected_predictions']} predicciones")
    if restante.total_seconds() <= 0:
        raise SystemExit("el ciclo ya cerró")

    objetivos = [(t["station_id"], pd.Timestamp(t["target_at"]))
                 for t in ciclo["targets"]]
    if len(objetivos) != ciclo["expected_predictions"]:
        raise SystemExit("la lista de targets no coincide con expected_predictions")

    modelo, ficha = ultimo_artefacto()
    print(f"modelo  : {ficha['name']}  version {ficha['version']}")

    ancha, ctx, _ = cargar()
    origen = pd.Timestamp(ciclo["data_cutoff"]).tz_convert(ancha.index.tz)
    if origen not in ancha.index:
        raise SystemExit(f"el corte {origen} no está en el histórico local; "
                         "hay que correr el collector primero")

    frame = construir_para_objetivos(ancha, ctx, origen, objetivos)
    pred = modelo.predict(frame)

    predicciones = [
        {"station_id": e, "target_at": t["target_at"], "value": round(float(v), 2)}
        for (e, _), t, v in zip(objetivos, ciclo["targets"], pred)
    ]
    if not all(p["value"] >= 0 for p in predicciones):
        raise SystemExit("hay predicciones negativas; la API rechazaría el lote")

    # Determinista: reintentar con el mismo contenido reusa la llave y no duplica.
    huella = hashlib.sha256(
        json.dumps([ciclo["cycle_id"], ficha["version"], predicciones],
                   sort_keys=True).encode()).hexdigest()
    payload = {
        "schema_version": "1.0",
        "cycle_id": ciclo["cycle_id"],
        "client_run_id": f"local-{ciclo['cycle_id']}-{ficha['version']}",
        "data_cutoff": ciclo["data_cutoff"],
        "model": {
            "version": ficha["version"],
            "trained_at": ficha["created_at"],
            "training_data_end": ficha["train_end"],
            "git_commit": ficha["git_commit"],
        },
        "predictions": predicciones,
    }

    print(f"\npredicciones ({len(predicciones)}):")
    for p in predicciones:
        print(f"   {p['station_id']}  {p['target_at']}  {p['value']:8.2f}")
    print(f"\nIdempotency-Key: {huella[:32]}")

    if dry_run:
        print("\n[dry-run] no se envió nada")
        return

    r = pedir(f"{base}/v1/submissions", key, "POST", payload,
              {"Idempotency-Key": huella[:32]})
    print(f"\nrespuesta HTTP {r['status']}")
    print(json.dumps(r["body"], indent=2, ensure_ascii=False)[:900])

    DIR_RECIBOS.mkdir(parents=True, exist_ok=True)
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = DIR_RECIBOS / f"{ciclo['cycle_id']}_{sello}.json"
    destino.write_text(json.dumps(
        {"status": r["status"], "enviado": payload, "recibo": r["body"],
         "idempotency_key": huella[:32]}, indent=2, ensure_ascii=False))
    print(f"recibo -> {destino}")

    if r["status"] not in (200, 201):
        raise SystemExit("el envío NO fue aceptado")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
