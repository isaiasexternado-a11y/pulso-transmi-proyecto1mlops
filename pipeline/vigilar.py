"""Vigila la tipología de los datos: formato de la API, documentos del
profesor y quiebres de patrón.

El 2026-10-01 el profesor anunció que va a cambiar la "tipología de los
datos". Puede ser el formato (campos, tipos, estaciones, horizontes que
entrega la API) o el comportamiento (otro tipo de drift). Este módulo no
corrige nada: detecta y deja constancia en `data_watch`, de donde el
dashboard saca un banner y el vigía en la nube manda correo. Corre en cada
pasada de evaluate.yml, después de evaluar, y un fallo aquí no tumba nada.

Tres familias de huellas (`kind`):

    formato  por componente (meta, reloj, estaciones, ciclo, fila del stream):
             las llaves y tipos de lo que responde la API, no los valores.
             El ciclo sólo se mira cuando hay uno abierto.
    profe    los encabezados "## Revisión ..." de docs/drift-operations.md y
             el contenido de docs/api-contract.md del repo del profesor: ahí
             documentó cada cambio de drift hasta ahora.
    patron   caída brusca: los dos últimos ciclos resueltos promedian menos
             de PISO_CAIDA y caen más de SALTO_CAIDA puntos frente a los seis
             anteriores.

Una huella conocida (base o aceptada) sólo actualiza `last_seen_at`. Una nueva
es `base` si es la primera de su componente, y `alerta` si no. Para dar por
bueno un formato nuevo después de adaptarse:

    python -m pipeline.vigilar --aceptar
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.entorno import Supabase, exigir           # noqa: E402
from collector.recolectar import api_get                # noqa: E402

REPO_PROFE = "uexternadojz/pulso-transmi"
PISO_CAIDA = 65.0
SALTO_CAIDA = 20.0
SILENCIO_PATRON_H = 6     # una caída que sigue no abre otra alerta en este lapso


def huella(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def forma(x):
    """Llaves y tipos, sin valores: lo que cambia si cambia el formato."""
    if isinstance(x, dict):
        return {k: forma(v) for k, v in x.items()}
    if isinstance(x, list):
        return [forma(x[0])] if x else []
    return type(x).__name__


# ------------------------------------------------------------------ formato

def componentes_formato(sb: Supabase, base: str, key: str) -> dict:
    comp = {}
    meta = api_get(base, key, "meta")
    ds = meta.get("dataset", {})
    comp["meta"] = {"llaves": forma(meta), "api_version": meta.get("api_version"),
                    "frecuencia_min": ds.get("frequency_minutes"),
                    "estaciones": ds.get("station_count"), "zona": ds.get("timezone")}
    comp["reloj"] = {"llaves": forma(api_get(base, key, "clock"))}
    est = api_get(base, key, "stations").get("data", [])
    comp["estaciones"] = {"ids": sorted(e.get("station_id") for e in est),
                          "llaves": forma(est[:1])}
    try:
        ciclo = api_get(base, key, "forecast-cycles/current")
        comp["ciclo"] = {"llaves": sorted(ciclo), "horizontes": ciclo.get("horizons_minutes"),
                         "esperadas": ciclo.get("expected_predictions"),
                         "estaciones": ciclo.get("station_count"),
                         "target": forma((ciclo.get("targets") or [{}])[0])}
    except urllib.error.HTTPError as e:
        if e.code != 404:          # 404 = no hay ciclo abierto: no se mira
            raise
    # Fila del stream del último tick: se arma un cursor un segundo antes del
    # último `released_at` conocido, con el mismo formato que usa el collector.
    cur = sb.seleccionar("stream_cursor", select="last_released_at", limit=1)
    if cur and cur[0].get("last_released_at"):
        t = datetime.fromisoformat(cur[0]["last_released_at"]) - timedelta(seconds=1)
        c = base64.urlsafe_b64encode(json.dumps(
            [t.isoformat(), "1970-01-01T00:00:00+00:00", ""]).encode()).decode().rstrip("=")
        filas = api_get(base, key, "stream/observations", limit=3, cursor=c).get("data", [])
        if filas:
            comp["stream"] = {"llaves": forma(filas[0])}
    return comp


# -------------------------------------------------------------------- profe

def _github(ruta: str) -> dict:
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO_PROFE}/{ruta}",
                                 headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _doc(ruta: str) -> str:
    return base64.b64decode(_github(f"contents/{ruta}")["content"]).decode()


def huellas_profe() -> list[tuple[str, str, dict]]:
    out = []
    drift = _doc("docs/drift-operations.md")
    for h in re.findall(r"^## (Revisi[oó]n[^\n]*)", drift, flags=re.M):
        out.append((f"revision:{h.strip()}", f"El profesor publicó «{h.strip()}» en docs/drift-operations.md",
                    {"encabezado": h.strip()}))
    contrato = _doc("docs/api-contract.md")
    out.append((f"api-contract:{huella(contrato)}", "Cambió docs/api-contract.md del profesor",
                {"bytes": len(contrato)}))
    return out


# ------------------------------------------------------------------- patrón

def caida(sb: Supabase) -> tuple[str, str, dict] | None:
    filas = sb.seleccionar("v_prediction_scores", select="station_id,issued_at,y_pred,y_true",
                           order="issued_at.desc", limit=48 * 12)
    por = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in filas:
        if r["y_true"] is None:
            continue
        e = por[r["issued_at"]][r["station_id"]]
        e[0] += abs(float(r["y_pred"]) - float(r["y_true"]))
        e[1] += float(r["y_true"])
    ciclos = sorted(por, reverse=True)
    if len(ciclos) < 8:
        return None
    acc = {c: sum(100 * max(0.0, 1 - a / b) if b else 0.0 for a, b in por[c].values()) / len(por[c])
           for c in ciclos[:8]}
    ult = (acc[ciclos[0]] + acc[ciclos[1]]) / 2
    previo = sum(acc[c] for c in ciclos[2:8]) / 6
    if ult < PISO_CAIDA and previo - ult > SALTO_CAIDA:
        return (f"caida:{ciclos[1]}",
                f"Caída brusca: los 2 últimos ciclos promedian {ult:.1f} % "
                f"contra {previo:.1f} % de los 6 anteriores",
                {"ultimos_2": round(ult, 2), "previos_6": round(previo, 2),
                 "ciclos": {c: round(acc[c], 1) for c in ciclos[:8]}})
    return None


# ----------------------------------------------------------------- registro

def registrar(sb: Supabase, kind: str, prefijo: str, fp: str, titulo: str, detalle: dict,
              conocidas: list[dict], ahora: str) -> str:
    """Devuelve el estado con que quedó la huella."""
    propia = next((r for r in conocidas if r["fingerprint"] == fp), None)
    if propia:
        sb.actualizar("data_watch", {"kind": f"eq.{kind}", "fingerprint": f"eq.{fp}"},
                      {"last_seen_at": ahora})
        return propia["status"]
    primera = not any(r["fingerprint"].startswith(prefijo) for r in conocidas)
    estado = "base" if primera else "alerta"
    sb.insertar("data_watch", [{"kind": kind, "fingerprint": fp, "status": estado,
                                "title": titulo, "detail": detalle,
                                "first_seen_at": ahora, "last_seen_at": ahora}])
    return estado


def main(aceptar: bool) -> None:
    base, key = exigir("PULSO_API_BASE", "PULSO_API_KEY")
    sb = Supabase()
    try:
        conocidas = sb.seleccionar("data_watch", select="kind,fingerprint,status,last_seen_at")
    except Exception as e:  # la tabla aún no existe: no se rompe nada
        print(f"::warning::data_watch no disponible ({type(e).__name__}); "
              f"falta aplicar sql/20261001_data_watch.sql")
        return
    if aceptar:
        sb.actualizar("data_watch", {"status": "eq.alerta", "kind": "in.(formato,profe)"},
                      {"status": "aceptada"})
        print("alertas de formato y del profe marcadas como aceptadas")
        return

    ahora = datetime.now(timezone.utc).isoformat()
    por_kind = defaultdict(list)
    for r in conocidas:
        por_kind[r["kind"]].append(r)
    nuevas = []

    try:
        for nombre, comp in componentes_formato(sb, base, key).items():
            fp = f"{nombre}:{huella(comp)}"
            est = registrar(sb, "formato", f"{nombre}:", fp,
                            f"Cambió el formato de la API ({nombre})", comp,
                            por_kind["formato"], ahora)
            print(f"formato  {nombre:10} {est}")
            if est == "alerta":
                nuevas.append(fp)
    except Exception as e:
        print(f"::warning::no se pudo leer el formato de la API ({type(e).__name__}: {e})")

    try:
        # Las huellas del profe se comparan como conjunto: la primera corrida
        # deja todas como base; después, cada encabezado nuevo es alerta.
        primera_profe = not por_kind["profe"]
        for fp, titulo, det in huellas_profe():
            prefijo = "" if primera_profe else "\0"
            est = registrar(sb, "profe", prefijo, fp, titulo, det, por_kind["profe"], ahora)
            print(f"profe    {fp[:60]:60} {est}")
            if est == "alerta":
                nuevas.append(fp)
    except Exception as e:
        print(f"::warning::no se pudo leer el repo del profesor ({type(e).__name__}: {e})")

    try:
        ev = caida(sb)
        if ev:
            fp, titulo, det = ev
            limite = datetime.now(timezone.utc) - timedelta(hours=SILENCIO_PATRON_H)
            reciente = [r for r in por_kind["patron"]
                        if datetime.fromisoformat(r["last_seen_at"]) > limite]
            if reciente and not any(r["fingerprint"] == fp for r in reciente):
                # la misma caída sigue: se renueva la alerta abierta, no se abre otra
                r0 = max(reciente, key=lambda r: r["last_seen_at"])
                sb.actualizar("data_watch", {"kind": "eq.patron", "fingerprint": f"eq.{r0['fingerprint']}"},
                              {"last_seen_at": ahora, "detail": det})
                print(f"patron   continúa {r0['fingerprint']}")
            else:
                est = registrar(sb, "patron", "caida:" if por_kind["patron"] else "\0",
                                fp, titulo, det, por_kind["patron"], ahora)
                print(f"patron   {fp} {est}")
                if est == "alerta":
                    nuevas.append(fp)
        else:
            print("patron   sin caída brusca")
    except Exception as e:
        print(f"::warning::no se pudo medir el patrón ({type(e).__name__}: {e})")

    if nuevas:
        print(f"::warning::CAMBIO DETECTADO: {', '.join(nuevas)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aceptar", action="store_true",
                    help="dar por buenos los formatos y documentos que hoy están en alerta")
    main(ap.parse_args().aceptar)
