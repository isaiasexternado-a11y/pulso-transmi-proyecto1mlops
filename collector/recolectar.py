"""Trae lo nuevo de la API y lo deja en Supabase.

Tres reglas gobiernan este archivo:

1. El estado vive en la base, nunca en el disco del runner. El cursor del
   stream se lee de `stream_cursor` al empezar y se guarda al terminar.
2. Idempotencia. Todo entra por upsert sobre la llave primaria. Correr el
   collector dos veces sobre el mismo tramo deja la base igual.
3. El cursor avanza de último. PostgREST no nos da una transacción que abarque
   varias tablas, así que el orden hace el trabajo: primero se confirman las
   filas, después se mueve el cursor. Si algo revienta en medio, el cursor
   sigue atrás y la próxima corrida vuelve a leer ese tramo — que por (2) es
   inofensivo. Al revés sí perderíamos datos en silencio.

Uso:
    python3 collector/recolectar.py --dry-run   # muestra qué traería, no escribe
    python3 collector/recolectar.py             # ingesta de verdad
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.entorno import Supabase, exigir  # noqa: E402

STREAM = "observations"     # nombre lógico del cursor en la tabla
LOTE = 500                  # filas por request, tanto de lectura como de escritura


# --------------------------------------------------------------------------- API

def api_get(base: str, key: str, ruta: str, **params) -> dict:
    destino = f"{base}/v1/{ruta}"
    params = {k: v for k, v in params.items() if v is not None}
    if params:
        destino += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(destino, headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def cursor_de(fila: dict) -> str:
    """Reconstruye el cursor opaco del stream a partir de una fila.

    Hace falta porque la API devuelve `next_cursor` nulo en toda página
    parcial, y en régimen todas lo son: son 48 filas por ciclo, nunca 500.
    Sin esto el cursor se quedaría clavado en la última página llena y cada
    corrida releería el stream entero, que sólo crece.

    El formato se verifica contra un `next_cursor` real cada vez que la API
    da uno (ver `paginar`). Si el profesor lo cambia, la corrida falla de
    frente en lugar de seguir leyendo desde el lugar equivocado.
    """
    partes = [fila["released_at"], fila["observed_at"], fila["station_id"]]
    crudo = json.dumps([p.replace("Z", "+00:00") for p in partes],
                       separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(crudo).decode().rstrip("=")


def paginar(base: str, key: str, ruta: str, cursor: str | None, **extra):
    """Recorre un endpoint paginado y devuelve (filas, último cursor)."""
    es_stream = ruta.startswith("stream/")
    filas, ultimo = [], cursor
    while True:
        pagina = api_get(base, key, ruta, cursor=ultimo, limit=LOTE, **extra)
        lote = pagina.get("data", [])
        if not lote:
            break
        filas.extend(lote)

        dado = pagina.get("next_cursor")
        if dado:
            if es_stream and cursor_de(lote[-1]) != dado.rstrip("="):
                raise RuntimeError(
                    "el formato del cursor del stream cambió; hay que revisar "
                    "cursor_de() antes de seguir ingiriendo")
            ultimo = dado
        elif es_stream:
            ultimo = cursor_de(lote[-1])

        if len(lote) < LOTE:
            break
    return filas, ultimo


# ------------------------------------------------------------------------ git

def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


# El esquema restringe el vocabulario (pipeline_runs_trigger_check): el
# evento de GitHub no entra crudo, se traduce.
TRIGGER = {"schedule": "schedule", "workflow_dispatch": "manual",
           "workflow_run": "retry", "repository_dispatch": "manual"}


def procedencia() -> dict:
    """De dónde viene esta corrida. En Actions lo dice el propio runner."""
    en_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run = os.environ.get("GITHUB_RUN_ID", "")
    evento = os.environ.get("GITHUB_EVENT_NAME", "")
    return {
        "trigger": TRIGGER.get(evento, "manual") if en_actions else "manual",
        "git_commit": os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD") or "desconocido",
        "git_ref": os.environ.get("GITHUB_REF") or git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "run_url": f"https://github.com/{repo}/actions/runs/{run}" if en_actions and repo and run else None,
    }


# ------------------------------------------------------------------- collector

def leer_cursor(sb: Supabase) -> str | None:
    filas = sb.seleccionar("stream_cursor", select="cursor",
                           stream=f"eq.{STREAM}", limit=1)
    return filas[0]["cursor"] if filas else None


def guardar_cursor(sb: Supabase, cursor: str, ultimo_release: str | None,
                   nuevas: int) -> None:
    previas = sb.seleccionar("stream_cursor", select="rows_total",
                             stream=f"eq.{STREAM}", limit=1)
    total = (previas[0]["rows_total"] if previas else 0) + nuevas
    sb.upsert("stream_cursor", [{
        "stream": STREAM,
        "cursor": cursor,
        "last_released_at": ultimo_release,
        "rows_total": total,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }], conflicto="stream")


def en_lotes(filas: list, tam: int = LOTE):
    for i in range(0, len(filas), tam):
        yield filas[i:i + tam]


def main(dry_run: bool) -> None:
    base, key = exigir("PULSO_API_BASE", "PULSO_API_KEY")
    sb = Supabase()

    cursor = leer_cursor(sb)
    print(f"cursor previo : {(cursor or '(ninguno: primera corrida)')[:48]}")

    obs, cursor_nuevo = paginar(base, key, "stream/observations", cursor)
    print(f"observaciones : {len(obs)} filas leídas del stream")

    # El contexto no tiene stream propio: se pide desde el último minuto que
    # ya tenemos en la base hacia adelante.
    tope = sb.seleccionar("context", select="observed_at",
                          order="observed_at.desc", limit=1)
    desde = tope[0]["observed_at"] if tope else None
    ctx, _ = paginar(base, key, "context", None, start=desde)
    ctx = [c for c in ctx if not desde or c["observed_at"] > desde]
    print(f"contexto      : {len(ctx)} filas nuevas")

    if obs:
        marcas = sorted(o["observed_at"] for o in obs)
        print(f"rango observado: {marcas[0]}  ->  {marcas[-1]}")

    if dry_run:
        print("\n[dry-run] no se escribió nada en Supabase")
        return

    if not obs and not ctx:
        print("\nno hay nada nuevo; la base ya está al día")
        return

    # Abrimos la corrida: las filas cuelgan de ella por run_id.
    run = sb.insertar("pipeline_runs", [{
        **procedencia(),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }], devolver=True)[0]
    run_id = run["run_id"]
    print(f"\nrun_id        : {run_id}")

    try:
        # `released_at` es del stream, no de la tabla: se queda afuera.
        filas_obs = [{"station_id": o["station_id"],
                      "observed_at": o["observed_at"],
                      "demand": o["demand"],
                      "run_id": run_id} for o in obs]
        for lote in en_lotes(filas_obs):
            sb.upsert("observations", lote, conflicto="station_id,observed_at")

        for lote in en_lotes([{**c, "run_id": run_id} for c in ctx]):
            sb.upsert("context", lote, conflicto="observed_at")

        # Sólo ahora, con las filas ya confirmadas, movemos el cursor.
        if obs and cursor_nuevo:
            guardar_cursor(sb, cursor_nuevo,
                           max(o["released_at"] for o in obs), len(obs))

        corte = max(o["observed_at"] for o in obs) if obs else None
        sb.actualizar("pipeline_runs", {"run_id": f"eq.{run_id}"}, {
            "status": "success",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "rows_ingested": len(obs) + len(ctx),
            "cutoff_at": corte,
        })
        print(f"ingestado     : {len(obs)} observaciones + {len(ctx)} contexto")
        print(f"corte         : {corte}")

    except Exception as e:
        sb.actualizar("pipeline_runs", {"run_id": f"eq.{run_id}"}, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_stage": "upsert",
            "error_message": str(e)[:1000],
        })
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
