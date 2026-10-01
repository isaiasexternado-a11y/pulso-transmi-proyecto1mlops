"""Despierta, consulta el ciclo vigente, infiere y entrega.

Lo ejecuta GitHub Actions en bucle, cada pocos minutos durante todo el turno.
La mayoría de las veces no hace nada, y eso es correcto: la ventana dura 25
minutos de cada hora y mirar veinte veces dentro de ella no significa entregar
veinte veces. Quien impide el duplicado es `ya_entregado`, no el reloj.

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
import re
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


# ------------------------------------------------------------------- el reloj

def registrar_reloj(sb: Supabase, base: str, key: str) -> dict | None:
    """Deja en `api_clock` el estado del reloj oficial, para el dashboard.

    Es lo único que le permite al panel distinguir "la API está en pausa" de
    "dejamos de entregar". `cambio_at` sólo se mueve cuando cambia `state`.
    Es informativo: si falla, se avisa y la entrega sigue.
    """
    try:
        reloj = api_get(base, key, "clock")
        previo = sb.seleccionar("api_clock", select="state,cambio_at",
                                reloj="eq.oficial", limit=1)
        ahora = datetime.now(timezone.utc).isoformat()
        cambio = (previo[0]["cambio_at"]
                  if previo and previo[0]["state"] == reloj.get("state") else ahora)
        sb.upsert("api_clock", [{
            "reloj": "oficial",
            "state": reloj.get("state") or "desconocido",
            "code": reloj.get("code"),
            "virtual_now": reloj.get("virtual_now"),
            "tick_number": reloj.get("tick_number"),
            "last_tick_at": reloj.get("last_tick_at"),
            "server_time": reloj.get("server_time"),
            "consultado_at": ahora,
            "cambio_at": cambio,
        }], conflicto="reloj")
        return reloj
    except Exception as e:
        print(f"[reloj] no se pudo registrar ({type(e).__name__}: {e}); se continúa")
        return None


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

def cargar_artefacto(sb: Supabase, ficha: dict) -> object:
    """Baja, verifica y deserializa el artefacto de una ficha de `models`.

    Es el ÚNICO camino por el que un modelo entra en producción. La promoción
    lo reusa a propósito: verificar un candidato con un cargador distinto al
    de la inferencia probaría la cosa equivocada.
    """
    uri = ficha["artifact_uri"]
    if not uri.startswith(f"supabase://{BUCKET}/"):
        raise SystemExit(f"artifact_uri inesperado: {uri}")
    crudo = sb.descargar(BUCKET, uri.split(f"supabase://{BUCKET}/", 1)[1])

    sha = hashlib.sha256(crudo).hexdigest()
    if sha != ficha["artifact_sha256"]:
        raise SystemExit(
            f"el artefacto no coincide con su huella registrada\n"
            f"  esperado: {ficha['artifact_sha256']}\n  obtenido: {sha}")

    # El pickle referencia `models` como módulo de primer nivel, que es como
    # lo importó empaquetar.py al crearlo. Sin esto, joblib no lo encuentra.
    import joblib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
    return joblib.load(io.BytesIO(crudo))["modelo"]


def champion(sb: Supabase) -> tuple[object, dict]:
    """Carga la versión promovida, nunca "el último archivo entrenado"."""
    filas = sb.seleccionar("models", select="*", status="eq.active",
                           order="activated_at.desc", limit=1)
    if not filas:
        raise SystemExit("no hay modelo con status=active en la tabla models")
    ficha = filas[0]
    return cargar_artefacto(sb, ficha), ficha


def predecir(modelo: object, ficha: dict, ancha, ctx, origen, objetivos) -> list[float]:
    """Predicción del champion con los ajustes que traiga su ficha.

    Dos capas, ambas en `hyperparams` y no en el pickle, para que una versión
    nueva no traiga una clase que `main` no conozca y para que el rollback las
    apague sin tocar el artefacto:

    - `mezcla_persistencia` (ver `_mezclado`).
    - `correccion_nivel` = {ventana_h, alfa, r_min, r_max}: cada estación se
      multiplica por `1 + alfa · (r − 1)`, con `r = Σ observado / Σ predicho`
      sobre los objetivos ya resueltos en las últimas `ventana_h` horas. Lo
      "predicho" son backcasts: lo que esta misma ficha, sin la corrección,
      habría dicho en cada origen horario previo con los datos de entonces.
      No se leen las predicciones guardadas, que ya vendrían corregidas y
      realimentarían el factor. Evidencia en `ml/experimento_nivel.py`.

      Con la llave opcional `quiebre` = {umbral, r_min, r_max}, una estación
      cuyo r de `ventana_h` y el de `2 · ventana_h` se desvían de 1 más que
      `umbral` en el mismo sentido cambió de régimen: se corrige completo
      (α = 1) con los límites amplios de `quiebre`. Sin la llave, nada cambia.
      Evidencia en `ml/experimento_nivel_corto.py`.

    - `estacional` = {periodo_min_h, periodo_max_h, ventana_h, n, umbral_pleno,
      umbral_apagado}: si la demanda de las últimas `ventana_h` horas se repite
      con un período P de entre `periodo_min_h` y `periodo_max_h` horas, el
      valor se lleva hacia el promedio de lo observado P, 2P, …, nP horas
      antes del objetivo (ver `estacional`). Se activa sola y se apaga sola.

    La promoción la reusa por la misma razón que `cargar_artefacto`.
    """
    valores = _mezclado(modelo, ficha, ancha, ctx, origen, objetivos)
    hp = ficha.get("hyperparams") or {}
    nivel = hp.get("correccion_nivel")
    if nivel:
        factor = factores_nivel(modelo, ficha, ancha, ctx, origen, nivel)
        valores = [v * factor.get(est, 1.0) for (est, _), v in zip(objetivos, valores)]
    if hp.get("estacional"):
        valores = estacional(ancha, origen, objetivos, valores, hp["estacional"])
    return valores


EMPATE = 0.25    # tolerancia relativa para preferir el período más corto


def detectar_periodo(ancha, origen, cfg: dict) -> tuple[int | None, float]:
    """El período corto (en pasos) que mejor explica las últimas horas.

    Para cada P entre `periodo_min_h` y `periodo_max_h` mide el WAPE global de
    predecir cada observación de (origen − ventana, origen] con la de P antes.
    Sólo lee observaciones hasta `origen`: nada del futuro.

    En el régimen normal ningún rezago corto explica la demanda (WAPE
    0,44-0,56 el 17 virtual); en la revisión 3 del drift la demanda se repite
    cada 4 h y el rezago de 4 h llega a 0,096.
    """
    import numpy as np
    import pandas as pd

    paso = ancha.index[1] - ancha.index[0]
    hasta = ancha.loc[:origen]
    ventana = int(pd.Timedelta(hours=cfg.get("ventana_h", 12)) / paso)
    pmin = int(pd.Timedelta(hours=cfg.get("periodo_min_h", 2)) / paso)
    pmax = int(pd.Timedelta(hours=cfg.get("periodo_max_h", 12)) / paso)
    if len(hasta) < ventana + pmax:
        return None, float("inf")
    v = hasta.to_numpy(dtype=float)
    reciente = v[-ventana:]
    total = np.nansum(reciente)
    if total <= 0:
        return None, float("inf")
    wapes = {p: float(np.nansum(np.abs(reciente - v[-ventana - p:-p])) / total)
             for p in range(pmin, pmax + 1)}
    minimo = min(wapes.values())
    # Los múltiplos del período verdadero empatan casi exacto (4 h y 8 h en la
    # revisión 3). Gana el más corto entre los que quedan a menos de
    # EMPATE del mínimo: con 8 h, el promedio de dos períodos miraría 16 h
    # atrás y caería en la transición. Con ventana de 6 h el mínimo puro elegía 8 h.
    mejor = min(p for p, w in wapes.items() if w <= minimo * (1 + EMPATE))
    return mejor, wapes[mejor]


def estacional(ancha, origen, objetivos, valores, cfg: dict) -> list[float]:
    """Lleva cada valor hacia el promedio de lo observado 1..n períodos antes.

    Peso 1 si el WAPE del período detectado es menor que `umbral_pleno`, 0 si
    supera `umbral_apagado`, lineal entre ambos. Medido el 2026-10-01 en los
    ciclos 14-21 virtuales del 18: promedio de P y 2P con peso 1 da 91,6 %
    (mínimo 90,1) contra ~70 % de lo entregado.
    """
    p, wape = detectar_periodo(ancha, origen, cfg)
    lo, hi = float(cfg.get("umbral_pleno", 0.15)), float(cfg.get("umbral_apagado", 0.25))
    peso = 1.0 if wape <= lo else 0.0 if wape >= hi else (hi - wape) / (hi - lo)
    print(f"estacional: período {p} pasos, wape {wape:.3f}, peso {peso:.2f}")
    if p is None or peso == 0:
        return valores
    paso = ancha.index[1] - ancha.index[0]
    n = int(cfg.get("n", 2))
    out = []
    for (est, t), v in zip(objetivos, valores):
        previos = [ancha.at[t - k * p * paso, est] for k in range(1, n + 1)
                   if (t - k * p * paso) in ancha.index and t - k * p * paso <= origen]
        if not previos:
            out.append(v)
            continue
        out.append((1 - peso) * v + peso * float(sum(previos)) / len(previos))
    return out


def factores_nivel(modelo: object, ficha: dict, ancha, ctx, origen,
                   nivel: dict) -> dict[str, float]:
    """Multiplicador final por estación (ver `predecir`)."""
    L = int(nivel["ventana_h"])
    quiebre = nivel.get("quiebre")
    r = razones_nivel(modelo, ficha, ancha, ctx, origen, [L, 2 * L] if quiebre else [L])
    lo, hi = float(nivel["r_min"]), float(nivel["r_max"])
    alfa = float(nivel["alfa"])
    out = {}
    for est, corto in r[L].items():
        out[est] = 1 + alfa * (min(max(corto, lo), hi) - 1)
        largo = r.get(2 * L, {}).get(est) if quiebre else None
        u = float(quiebre["umbral"]) if quiebre else 0.0
        if largo is not None and abs(corto - 1) > u and abs(largo - 1) > u \
                and (corto > 1) == (largo > 1):
            out[est] = min(max(corto, float(quiebre["r_min"])), float(quiebre["r_max"]))
    return out


def razones_nivel(modelo: object, ficha: dict, ancha, ctx, origen,
                  ventanas_h: list[int]) -> dict[int, dict[str, float]]:
    """r sin acotar por estación, con los objetivos en (origen − L, origen]
    para cada L de `ventanas_h`. Los backcasts se calculan una sola vez."""
    import pandas as pd

    hora = pd.Timedelta(hours=1)
    paso = ancha.index[1] - ancha.index[0]
    pasos_hora = round(hora / paso)
    obs, pred = {L: {} for L in ventanas_h}, {L: {} for L in ventanas_h}
    for k in range(1, max(ventanas_h) + 1):
        o = origen - k * hora
        if o not in ancha.index:
            break
        objetivos = [(est, o + h * paso) for est in ancha.columns
                     for h in range(1, pasos_hora + 1)]
        for (est, t), v in zip(objetivos, _mezclado(modelo, ficha, ancha, ctx, o, objetivos)):
            for L in ventanas_h:
                if k <= L:
                    obs[L][est] = obs[L].get(est, 0.0) + float(ancha.at[t, est])
                    pred[L][est] = pred[L].get(est, 0.0) + max(v, 0.0)
    return {L: {est: obs[L][est] / pred[L][est] for est in obs[L] if pred[L][est] > 0}
            for L in ventanas_h}


def _mezclado(modelo: object, ficha: dict, ancha, ctx, origen, objetivos) -> list[float]:
    """Modelo más la mezcla con persistencia de la ficha.

    `hyperparams.mezcla_persistencia` es {horizonte en pasos: peso}. Cada valor
    queda en `(1 - w) · modelo + w · último observado en el corte`. Sin la
    llave, el modelo sale tal cual.
    """
    from ml.features import construir_para_objetivos

    valores = [float(v) for v in
               modelo.predict(construir_para_objetivos(ancha, ctx, origen, objetivos))]
    pesos = (ficha.get("hyperparams") or {}).get("mezcla_persistencia")
    if not pesos:
        return valores

    paso = ancha.index[1] - ancha.index[0]
    mezclados = []
    for (est, target), v in zip(objetivos, valores):
        w = float(pesos[str(round((target - origen) / paso))])
        mezclados.append((1 - w) * v + w * float(ancha.at[origen, est]))
    return mezclados


def ya_entregado(sb: Supabase, cycle_id: str, version: str) -> dict | None:
    """Sólo una entrega *aceptada* cierra el ciclo.

    Un 409 o un 422 también dejan recibo —queda constancia de que se intentó—
    pero no consumen intento, así que no pueden bloquear el reintento dentro
    de la misma ventana. Filtrar por `http_status` es lo que separa "ya
    entregué" de "ya fallé", y sin ese filtro un rechazo corregible costaba
    la ventana entera.
    """
    filas = sb.seleccionar("submissions", select="submission_id,http_status",
                           cycle_id=f"eq.{cycle_id}",
                           model_version=f"eq.{version}",
                           http_status="in.(200,201)", limit=1)
    return filas[0] if filas else None


# ---------------------------------------------------------------- validación

def version_en_contrato(version: str) -> bool:
    """Mismo patrón que `SAFE_VERSION` en app/contracts.py de la API. Un `+`
    en la versión costó un 422 y casi la ventana del 2026-09-25 19:42."""
    return len(version) <= 64 and bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", version))


def validar(predicciones: list[dict], ciclo: dict, version: str) -> None:
    """El checklist previo al POST. Un rechazo por contrato no gasta intento,
    pero tampoco tiene por qué salir de aquí."""
    if not version_en_contrato(version):
        raise SystemExit(f"versión fuera de contrato: {version!r}")

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

    # No entregues lo que después no vas a poder guardar. `predictions.horizon`
    # se cuenta en pasos de 15 min (1..4) y el esquema lo hace cumplir: si el
    # horizonte viene raro, esto revienta *antes* del POST. El orden importa —
    # ya pasó una vez al revés (HTTP 201 y luego error al guardar), y el ciclo
    # quedó entregado pero inevaluable para siempre.
    for t in ciclo["targets"]:
        minutos = t["horizon_minutes"]
        if minutos % 15 or not 1 <= minutos // 15 <= 4:
            raise SystemExit(
                f"horizonte fuera de contrato: {minutos} min en "
                f"{t['station_id']} {t['target_at']} (se esperan 15, 30, 45 o 60)")


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

    # issued_at no es "ahora": el esquema exige
    #   target_at = issued_at + horizon * 15min
    # así que es el instante desde el que se proyecta, o sea el corte.
    origen = ciclo["data_cutoff"]
    emitido = origen
    sb.upsert("predictions", [{
        "run_id": run_id,
        "station_id": p["station_id"],
        "target_at": p["target_at"],
        "model_id": ficha["model_id"],
        "issued_at": emitido,
        "horizon": t["horizon_minutes"] // 15,   # el esquema los cuenta en pasos, 1..4
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
    reloj = registrar_reloj(sb, base, key)
    if reloj:
        print(f"reloj   : {reloj.get('state')}  virtual {reloj.get('virtual_now', '—')}")
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

    ancha, ctx, _ = cargar(origen="supabase")
    origen = pd.Timestamp(ciclo["data_cutoff"]).tz_convert(ancha.index.tz)
    if origen not in ancha.index:
        raise SystemExit(
            f"el corte {origen} no está en el histórico; el último dato es "
            f"{ancha.index[-1]}. El collector no alcanzó a traerlo.")

    objetivos = [(t["station_id"], pd.Timestamp(t["target_at"]))
                 for t in ciclo["targets"]]
    valores = predecir(modelo, ficha, ancha, ctx, origen, objetivos)
    predicciones = [
        {"station_id": t["station_id"], "target_at": t["target_at"],
         "value": round(float(v), 2)}
        for t, v in zip(ciclo["targets"], valores)]

    validar(predicciones, ciclo, version)
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
