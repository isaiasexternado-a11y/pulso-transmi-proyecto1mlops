"""Despierta, consulta el ciclo vigente, infiere y entrega.

Lo ejecuta GitHub Actions cada 10 minutos. La mayoría de las veces no hace
nada, y eso es correcto: la ventana dura 25 minutos de cada hora y despertar
tres veces dentro de ella no significa entregar tres veces.

Termina en verde (código 0) cuando no hay ciclo abierto o cuando ese ciclo ya
tiene recibo. Sólo falla cuando algo está realmente mal: credencial inválida,
contrato violado o error del servidor.

El orden lo manda la guía operativa v2.0 del curso:

    sincronizar -> consultar ciclo -> ¿ya entregué? -> inferir -> validar
    -> enviar -> guardar recibo

Uso:
    python3 -m pipeline.entregar --dry-run   # arma y valida, no envía
    python3 -m pipeline.entregar             # opera de verdad
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.entorno import Supabase, exigir           # noqa: E402
from collector.recolectar import api_get, procedencia    # noqa: E402

BUCKET = "modelos"


def salir_ok(motivo: str) -> None:
    print(f"nada que hacer: {motivo}")
    raise SystemExit(0)


# ------------------------------------------------------------------- el ciclo

def ciclo_vigente(base: str, key: str) -> dict | None:
    """El 404 `no_open_cycle` es un resultado normal, no un error."""
    try:
        return api_get(base, key, "forecast-cycles/current")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def registrar_ciclo(sb: Supabase, ciclo: dict) -> None:
    """Deja constancia del ciclo tal como lo emitió la API."""
    sb.upsert("forecast_cycles", [{
        "cycle_id": ciclo["cycle_id"],
        "origin_at": ciclo.get("origin_at"),
        "data_cutoff": ciclo["data_cutoff"],
        "opens_at": ciclo.get("opens_at"),
        "closes_at": ciclo.get("closes_at"),
        "forecast_start_at": ciclo.get("forecast_start_at"),
        "forecast_end_at": ciclo.get("forecast_end_at"),
        "station_count": ciclo.get("station_count"),
        "expected_predictions": ciclo.get("expected_predictions"),
    }], conflicto="cycle_id")


# ------------------------------------------------------------------ el modelo

def champion(sb: Supabase) -> tuple[object, dict]:
    """Carga la versión promovida, nunca "el último archivo entrenado"."""
    filas = sb.seleccionar("models", select="*", status="eq.active",
                           order="activated_at.desc", limit=1)
    if not filas:
        raise SystemExit("no hay modelo con status=active en la tabla models")
    ficha = filas[0]

    uri = ficha["artifact_uri"]
    if not uri.startswith(f"supabase://{BUCKET}/"):
        raise SystemExit(f"artifact_uri inesperado: {uri}")
    crudo = sb.descargar(BUCKET, uri.split(f"supabase://{BUCKET}/", 1)[1])

    sha = hashlib.sha256(crudo).hexdigest()
    if sha != ficha["artifact_sha256"]:
        raise SystemExit(
            f"el artefacto no coincide con su huella registrada\n"
            f"  esperado: {ficha['artifact_sha256']}\n  obtenido: {sha}")

    import joblib
    return joblib.load(io.BytesIO(crudo))["modelo"], ficha


def ya_entregado(sb: Supabase, cycle_id: str, version: str) -> dict | None:
    filas = sb.seleccionar("submissions", select="submission_id,http_status",
                           cycle_id=f"eq.{cycle_id}",
                           model_version=f"eq.{version}", limit=1)
    return filas[0] if filas else None


# ---------------------------------------------------------------- validación

def validar(predicciones: list[dict], ciclo: dict) -> None:
    """El checklist previo al POST. Un rechazo por contrato no gasta intento,
    pero tampoco tiene por qué salir de aquí."""
    esperados = {(t["station_id"], t["target_at"]) for t in ciclo["targets"]}
    recibidos = [(p["station_id"], p["target_at"]) for p in predicciones]

    if len(recibidos) != len(set(recibidos)):
        raise SystemExit("hay pares (estación, target_at) duplicados")
    if set(recibidos) != esperados:
        faltan = sorted(esperados - set(recibidos))[:5]
        sobran = sorted(set(recibidos) - esperados)[:5]
        raise SystemExit(f"los targets no calzan. faltan={faltan} sobran={sobran}")
    if len(predicciones) != ciclo["expected_predictions"]:
        raise SystemExit(f"se esperaban {ciclo['expected_predictions']} predicciones")

    for p in predicciones:
        v = p["value"]
        if v != v or v in (float("inf"), float("-inf")):
            raise SystemExit(f"valor no finito en {p['station_id']} {p['target_at']}")
        if v < 0:
            raise SystemExit(f"valor negativo en {p['station_id']} {p['target_at']}")


def llave_estable(cycle_id: str, version: str, predicciones: list[dict]) -> str:
    """Mismo ciclo y mismo contenido reusan la misma llave: un reintento no
    crea una entrega nueva."""
    huella = hashlib.sha256(json.dumps(
        [cycle_id, version, predicciones], sort_keys=True).encode()).hexdigest()
    return huella[:32]


# --------------------------------------------------------------------- envío

def enviar(base: str, key: str, payload: dict, llave: str) -> dict:
    datos = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/v1/submissions", data=datos, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Idempotency-Key": llave})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"status": r.status, "body": json.loads(r.read() or b"{}")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": json.loads(e.read() or b"{}")}


def guardar_recibo(sb: Supabase, ciclo: dict, ficha: dict, llave: str,
                   respuesta: dict, run_id: int | None,
                   predicciones: list[dict]) -> None:
    cuerpo = respuesta["body"]
    sub_id = cuerpo.get("submission_id") or f"local-{llave}"
    sb.upsert("submissions", [{
        "submission_id": sub_id,
        "cycle_id": ciclo["cycle_id"],
        "model_id": ficha["model_id"],
        "model_version": ficha["hyperparams"]["version"],
        "idempotency_key": llave,
        "payload_hash": cuerpo.get("payload_hash"),
        "attempt": cuerpo.get("attempt"),
        "is_official": cuerpo.get("is_official"),
        "http_status": respuesta["status"],
        "response": cuerpo,
        "run_id": run_id,
    }], conflicto="submission_id")

    emitido = datetime.now(timezone.utc).isoformat()
    origen = ciclo["data_cutoff"]
    sb.upsert("predictions", [{
        "run_id": run_id,
        "station_id": p["station_id"],
        "target_at": p["target_at"],
        "model_id": ficha["model_id"],
        "issued_at": emitido,
        "horizon": t["horizon_minutes"],
        "y_pred": p["value"],
        "submitted": respuesta["status"] in (200, 201),
        "submit_response": {"http": respuesta["status"]},
        "cycle_id": ciclo["cycle_id"],
        "submission_id": sub_id,
    } for p, t in zip(predicciones, ciclo["targets"])],
        conflicto="run_id,station_id,target_at")
    print(f"recibo guardado: {sub_id}  (origen {origen})")


# ---------------------------------------------------------------------- main

def main(dry_run: bool) -> None:
    base, key = exigir("PULSO_API_BASE", "PULSO_API_KEY")
    sb = Supabase()

    # 1. Sincronizar. Sin datos frescos el corte del ciclo no existe localmente.
    from collector.recolectar import main as recolectar
    print("--- sincronizando ---")
    recolectar(dry_run=False)

    # 2. Consultar el ciclo. La API es la autoridad.
    print("\n--- ciclo ---")
    ciclo = ciclo_vigente(base, key)
    if ciclo is None:
        salir_ok("no hay ciclo abierto (404 no_open_cycle)")
    if ciclo.get("state") != "open":
        salir_ok(f"el ciclo está en estado {ciclo.get('state')}")

    cierra = datetime.fromisoformat(ciclo["closes_at"].replace("Z", "+00:00"))
    restante = (cierra - datetime.now(timezone.utc)).total_seconds()
    print(f"ciclo   : {ciclo['cycle_id']}")
    print(f"cutoff  : {ciclo['data_cutoff']}")
    print(f"cierra  : {ciclo['closes_at']}  (faltan {restante/60:.1f} min)")
    if restante <= 0:
        salir_ok("la ventana ya cerró")
    registrar_ciclo(sb, ciclo)

    # 3. El modelo promovido, y si ya entregamos este ciclo con él.
    modelo, ficha = champion(sb)
    version = ficha["hyperparams"]["version"]
    print(f"modelo  : {ficha['name']}  version {version}")

    previo = ya_entregado(sb, ciclo["cycle_id"], version)
    if previo:
        salir_ok(f"este ciclo ya tiene recibo ({previo['submission_id']})")

    # 4. Inferir exactamente lo que la API pidió.
    import pandas as pd
    from ml.data import cargar
    from ml.features import construir_para_objetivos

    ancha, ctx, _ = cargar(origen="supabase")
    origen = pd.Timestamp(ciclo["data_cutoff"]).tz_convert(ancha.index.tz)
    if origen not in ancha.index:
        raise SystemExit(
            f"el corte {origen} no está en el histórico; el último dato es "
            f"{ancha.index[-1]}. El collector no alcanzó a traerlo.")

    objetivos = [(t["station_id"], pd.Timestamp(t["target_at"]))
                 for t in ciclo["targets"]]
    valores = modelo.predict(construir_para_objetivos(ancha, ctx, origen, objetivos))
    predicciones = [
        {"station_id": t["station_id"], "target_at": t["target_at"],
         "value": round(float(v), 2)}
        for t, v in zip(ciclo["targets"], valores)]

    validar(predicciones, ciclo)
    llave = llave_estable(ciclo["cycle_id"], version, predicciones)
    print(f"predice : {len(predicciones)} valores  ·  llave {llave}")

    payload = {
        "schema_version": "1.0",
        "cycle_id": ciclo["cycle_id"],
        "client_run_id": f"gha-{ciclo['cycle_id']}-{version}",
        "data_cutoff": ciclo["data_cutoff"],
        "model": {
            "version": version,
            "trained_at": ficha["created_at"],
            "training_data_end": ficha["train_end"],
            "git_commit": ficha["git_commit"],
        },
        "predictions": predicciones,
    }

    if dry_run:
        print("\n[dry-run] validado y sin enviar")
        print(json.dumps(predicciones[:4], indent=2))
        return

    # 5. Enviar y dejar constancia pase lo que pase.
    print("\n--- enviando ---")
    r = enviar(base, key, payload, llave)
    print(f"HTTP {r['status']}")

    run = sb.seleccionar("pipeline_runs", select="run_id",
                         order="run_id.desc", limit=1)
    run_id = run[0]["run_id"] if run else None
    guardar_recibo(sb, ciclo, ficha, llave, r, run_id, predicciones)

    if r["status"] in (200, 201):
        print(f"ENTREGADO · oficial={r['body'].get('is_official')} "
              f"intento={r['body'].get('attempt')}")
        return
    if r["status"] == 409:
        raise SystemExit(f"409: conflicto o límite de intentos -> {r['body']}")
    if r["status"] == 422:
        raise SystemExit(f"422: el batch viola el contrato -> {r['body']}")
    raise SystemExit(f"la entrega no fue aceptada ({r['status']}): {r['body']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
