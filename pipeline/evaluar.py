"""Evalúa lo entregado, mide drift y decide si hay que reentrenar.

Corre después de la entrega, cuando la realidad ya alcanzó a los targets. No
entrena ni promueve: sólo mide y deja escrito *por qué*. La decisión queda en
`retrain_decisions` para que reentrenar sea una consecuencia registrada y no
un impulso.

Cuatro decisiones de diseño que conviene no deshacer sin pensarlo:

1. **Las ventanas móviles se anclan al tiempo virtual**, nunca a `now()`. El
   reloj de la competencia va ~12 días detrás del real, así que un filtro
   `target_at >= now() - 24h` no selecciona nada. La vista `v_accuracy_rolling`
   del esquema tiene justo ese defecto y devuelve 0 filas; por eso aquí el
   rolling se recalcula contra `max(target_at)` y no contra el reloj de pared.

2. **`model_metrics` guarda la foto vigente, no la historia.** Su llave natural
   es (model_id, split, fold, station_id, metric), así que cada corrida
   sobrescribe. `fold` codifica la ventana: 0 acumulado, 1 = 24 h, 7 = 7 d.
   No se pierde historia: `drift_signals` sí lleva `run_id`, y cualquier
   métrica se recalcula desde `v_prediction_scores`, que es la fuente.

3. **Las seis señales no son la misma cosa medida seis veces.** `wape_*` ve
   que el modelo falló; `profile_corr` ve que cambió la *forma* del día antes
   de que el WAPE se entere; `level_shift_7d` ve un cambio de nivel que el
   WAPE casi no acusa; `residual_bias` ve que fallamos siempre para el mismo
   lado; `ingest_gap` distingue un hueco de datos de un modelo malo, que se
   parecen y no son lo mismo.

4. **Reentrenar exige las cuatro condiciones**: umbral roto, persistencia,
   datos nuevos suficientes y sin enfriamiento. Un mal periodo no es drift.

`rollback` existe en el vocabulario del esquema pero no se emite aquí: revertir
al champion anterior es una decisión de promoción y vive en el workflow de
entrenamiento, que es quien conoce al modelo saliente.

Uso:
    python3 -m pipeline.evaluar --dry-run   # calcula e imprime, no escribe
    python3 -m pipeline.evaluar             # mide y decide de verdad
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.entorno import Supabase, exigir           # noqa: E402
from collector.recolectar import api_get, procedencia    # noqa: E402

TZ = "America/Bogota"
PAGINA = 1000                      # PostgREST recorta toda respuesta a 1000 filas

# --------------------------------------------------------------------- umbrales
# El piso no es un número redondo: 88,11 % es el baseline de perfil
# (estación x slot x tipo de día). Por debajo de 87 % el champion ya perdió
# contra algo que no necesita entrenarse, y eso es lo que lo vuelve inaceptable.
UMBRAL_ACCURACY = 87.0
UMBRAL_WAPE = round(1 - UMBRAL_ACCURACY / 100, 4)
UMBRAL_LEVEL_SHIFT = 0.15          # cambio relativo de nivel de demanda
UMBRAL_PROFILE_CORR = 0.90         # correlación con el perfil aprendido
UMBRAL_RESIDUAL_BIAS = 0.10        # sesgo sistemático sobre la demanda real
UMBRAL_INGEST_GAP_MIN = 45.0       # un tick son 30 min; 45 ya es atraso

MIN_PUNTOS_SENAL = 96              # 2 ciclos: menos que eso es ruido, no señal
CICLOS_PERSISTENCIA = 3            # corridas seguidas en rojo antes de actuar
COOLDOWN_HORAS = 6                 # tras decidir reentrenar, no volver a decidir
MIN_OBS_NUEVAS = 576               # ~12 h a 48 obs/h: mínimo para validar

# El constructor `Timedelta` quedó deprecado por numpy 2.5 ("generic unit") y
# promete volverse error. `to_timedelta` con unidad explícita no lo está, y
# requirements.txt permite numpy hasta 3.x, así que el runner puede traerlo.
D24 = pd.to_timedelta(24, unit="h")
D7 = pd.to_timedelta(7, unit="D")

VENTANAS = {0: None, 1: D24, 7: D7}


# ------------------------------------------------------------------- utilidades

def paginar(sb: Supabase, tabla: str, columnas: str, orden: str, **filtros) -> list[dict]:
    filas, desplazamiento = [], 0
    while True:
        lote = sb.seleccionar(tabla, select=columnas, order=orden,
                              limit=PAGINA, offset=desplazamiento, **filtros)
        filas.extend(lote)
        if len(lote) < PAGINA:
            return filas
        desplazamiento += len(lote)


def abrir_run(sb: Supabase) -> int:
    fila = sb.insertar("pipeline_runs", [{
        **procedencia(),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }], devolver=True)[0]
    return fila["run_id"]


def cerrar_run(sb: Supabase, run_id: int, status: str, **extra) -> None:
    sb.actualizar("pipeline_runs", {"run_id": f"eq.{run_id}"}, {
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(), **extra})


# --------------------------------------------------------------------- lectura

def traer_scores(sb: Supabase, model_id: str) -> pd.DataFrame:
    """Lo entregado que ya tiene realidad observada, vía la vista del esquema."""
    filas = paginar(sb, "v_prediction_scores",
                    "station_id,target_at,horizon,y_pred,y_true,abs_error",
                    "target_at.asc,station_id.asc", model_id=f"eq.{model_id}")
    if not filas:
        return pd.DataFrame()
    df = pd.DataFrame(filas)
    df["target_at"] = pd.to_datetime(df["target_at"], utc=True)
    df["station_id"] = df["station_id"].astype(str)
    df["resid"] = df["y_pred"] - df["y_true"]
    return df


def traer_observaciones(sb: Supabase) -> pd.DataFrame:
    filas = paginar(sb, "observations", "station_id,observed_at,demand",
                    "observed_at.asc,station_id.asc")
    df = pd.DataFrame(filas)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["station_id"] = df["station_id"].astype(str)
    return df


# --------------------------------------------------------------------- métricas

def metricas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Por estación, y la fila global.

    La global promedia SIN ponderar las accuracies de las estaciones: es la
    definición oficial del reto y existe para que una estación de gran volumen
    no tape el mal desempeño de una pequeña.
    """
    g = df.groupby("station_id")
    err, real = g["abs_error"].sum(), g["y_true"].sum()
    wape = err / real.where(real != 0)
    por = pd.DataFrame({
        "wape": wape,
        "accuracy": (100 * (1 - wape)).clip(lower=0),
        "mae": g["abs_error"].mean(),
        "rmse": g["resid"].apply(lambda s: float((s ** 2).mean())) ** 0.5,
        "bias": g["resid"].mean(),
    })
    glob = {
        "wape": float(por["wape"].mean()),
        "accuracy": float(por["accuracy"].mean()),
        "mae": float(df["abs_error"].mean()),
        "rmse": float((df["resid"] ** 2).mean() ** 0.5),
        "bias": float(df["resid"].mean()),
    }
    return por, glob


def ventana(scores: pd.DataFrame, ancho: pd.Timedelta | None) -> pd.DataFrame:
    """Recorta contra el último target_at, que es el reloj virtual."""
    if ancho is None:
        return scores
    return scores[scores["target_at"] >= scores["target_at"].max() - ancho]


def filas_metricas(scores: pd.DataFrame, model_id: str, ahora: str) -> list[dict]:
    filas = []
    for fold, ancho in VENTANAS.items():
        v = ventana(scores, ancho)
        if v.empty:
            continue
        por, glob = metricas(v)
        base = {"model_id": model_id, "split": "production", "fold": fold,
                "window_start": v["target_at"].min().isoformat(),
                "window_end": v["target_at"].max().isoformat(),
                "computed_at": ahora}
        for est, r in por.iterrows():
            for m in ("wape", "accuracy", "mae", "rmse", "bias"):
                if pd.notna(r[m]):
                    filas.append({**base, "station_id": est,
                                  "metric": m, "value": float(r[m])})
        for m, v_ in glob.items():
            if pd.notna(v_):
                filas.append({**base, "station_id": None,
                              "metric": m, "value": float(v_)})
    return filas


# ---------------------------------------------------------------------- señales

def perfil(obs: pd.DataFrame) -> pd.Series:
    """Demanda media por (estación, slot de 15 min, entre semana / finde)."""
    t = obs["observed_at"].dt.tz_convert(TZ)
    k = pd.DataFrame({
        "station_id": obs["station_id"].values,
        "slot": (t.dt.hour * 4 + t.dt.minute // 15).values,
        "finde": (t.dt.dayofweek >= 5).astype(int).values,
        "demand": obs["demand"].values,
    })
    return k.groupby(["station_id", "slot", "finde"])["demand"].mean()


def senales(run_id: int, scores: pd.DataFrame, obs: pd.DataFrame,
            ficha: dict, base: str, key: str, ahora: str) -> list[dict]:
    filas: list[dict] = []

    def add(signal, station, value, umbral, roto):
        filas.append({"run_id": run_id, "station_id": station, "signal": signal,
                      "value": float(value), "threshold": float(umbral),
                      "breached": bool(roto), "computed_at": ahora})

    # --- desempeño: wape en ventana de 24 h y de 7 d (tiempo virtual)
    # Medir y alarmar no son lo mismo. `model_metrics` guarda la accuracy
    # aunque haya un solo ciclo evaluado; una señal de drift con 48 puntos, en
    # cambio, es ruido con cara de medición y dispararía un reentrenamiento
    # por nada. La métrica se publica siempre; la alarma exige evidencia.
    for signal, ancho in (("wape_24h", D24), ("wape_7d", D7)):
        v = ventana(scores, ancho)
        if len(v) < MIN_PUNTOS_SENAL:
            print(f"  [{signal}] {len(v)} puntos en ventana; "
                  f"se exigen {MIN_PUNTOS_SENAL} para alarmar. No se emite.")
            continue
        por, glob = metricas(v)
        add(signal, None, glob["wape"], UMBRAL_WAPE, glob["wape"] > UMBRAL_WAPE)
        for est, r in por.iterrows():
            if pd.notna(r["wape"]):
                add(signal, est, r["wape"], UMBRAL_WAPE, r["wape"] > UMBRAL_WAPE)

    # --- sesgo: ¿fallamos siempre para el mismo lado?
    v24 = ventana(scores, D24)
    if len(v24) >= MIN_PUNTOS_SENAL and v24["y_true"].sum():
        b = v24["resid"].sum() / v24["y_true"].sum()
        add("residual_bias", None, b, UMBRAL_RESIDUAL_BIAS,
            abs(b) > UMBRAL_RESIDUAL_BIAS)
        for est, g in v24.groupby("station_id"):
            if g["y_true"].sum():
                bb = g["resid"].sum() / g["y_true"].sum()
                add("residual_bias", est, bb, UMBRAL_RESIDUAL_BIAS,
                    abs(bb) > UMBRAL_RESIDUAL_BIAS)

    # --- datos: lo que el modelo NO vio, contra lo que sí vio
    t_ini = pd.Timestamp(ficha["train_start"])
    t_fin = pd.Timestamp(ficha["train_end"])
    fin_obs = obs["observed_at"].max()
    entren = obs[(obs["observed_at"] >= t_ini) & (obs["observed_at"] <= t_fin)]
    corte = max(t_fin, fin_obs - D7)
    reciente = obs[obs["observed_at"] > corte]

    # Con menos de un día de datos nuevos estas dos señales dirían cualquier
    # cosa. Vale más no emitirlas que emitir ruido con cara de medición.
    if len(reciente) >= 96 and not entren.empty:
        pe, pr = perfil(entren), perfil(reciente)
        comun = pe.index.intersection(pr.index)

        # El nivel se compara celda contra celda, nunca media cruda contra
        # media cruda. La ventana reciente casi nunca cubre un día entero, y
        # si le falta la madrugada su media sube sola: eso es composición
        # horaria, no un cambio de demanda. Comparar (estación, slot, finde)
        # con su propio equivalente elimina el artefacto.
        if len(comun) >= 30:
            razon = pr.reindex(comun) / pe.reindex(comun).where(pe.reindex(comun) != 0)
            gl = float((razon - 1).mean())
            add("level_shift_7d", None, gl, UMBRAL_LEVEL_SHIFT,
                abs(gl) > UMBRAL_LEVEL_SHIFT)
            por_est = (razon - 1).groupby(level="station_id").mean()
            for est, v_ in por_est.items():
                if pd.notna(v_):
                    add("level_shift_7d", est, v_, UMBRAL_LEVEL_SHIFT,
                        abs(v_) > UMBRAL_LEVEL_SHIFT)

        if len(comun) >= 30:
            c = float(pe.reindex(comun).corr(pr.reindex(comun)))
            if pd.notna(c):
                add("profile_corr", None, c, UMBRAL_PROFILE_CORR,
                    c < UMBRAL_PROFILE_CORR)
            for est in sorted({i[0] for i in comun}):
                sub = [i for i in comun if i[0] == est]
                if len(sub) >= 20:
                    ce = float(pe.reindex(sub).corr(pr.reindex(sub)))
                    if pd.notna(ce):
                        add("profile_corr", est, ce, UMBRAL_PROFILE_CORR,
                            ce < UMBRAL_PROFILE_CORR)

    # --- ingesta: un hueco de datos se parece a un modelo malo, y no lo es.
    # El reloj de la API es la autoridad; el atraso no se calcula con la hora local.
    reloj = api_get(base, key, "clock")
    virtual = pd.Timestamp(reloj["virtual_now"])
    if virtual.tzinfo is None:
        virtual = virtual.tz_localize("UTC")
    atraso = (virtual - fin_obs).total_seconds() / 60
    add("ingest_gap", None, atraso, UMBRAL_INGEST_GAP_MIN,
        atraso > UMBRAL_INGEST_GAP_MIN)

    return filas


# ----------------------------------------------------------------- leaderboard

def ingestar_leaderboard(sb: Supabase, base: str, key: str) -> None:
    """Guarda la posición propia y los agregados del pelotón.

    El dashboard no puede pedirle esto a la API: haría falta PULSO_API_KEY y
    esa llave no puede estar en variables públicas de Vercel. La trae quien ya
    la tiene —este proceso, que corre en Actions— y la deja como dato derivado.

    No se guardan nombres de terceros. La API los devuelve, pero el dashboard
    es una página pública y republicar ahí el puntaje de los compañeros es otra
    cosa que verlo dentro de la plataforma del curso. Con nuestra fila y los
    agregados alcanza para el panel de "carrera".

    Un fallo aquí no puede tumbar la evaluación: el leaderboard es un adorno
    informativo, las métricas y el drift no.
    """
    try:
        yo = api_get(base, key, "me")["display_name"]
        datos = api_get(base, key, "leaderboard").get("data", [])
    except Exception as e:
        print(f"[leaderboard] no se pudo leer ({type(e).__name__}); se continúa")
        return
    if not datos:
        return

    mio = next((r for r in datos if r.get("display_name") == yo), None)
    if mio is None:
        print("[leaderboard] todavía no aparecemos en la tabla")
        return

    accs = sorted(r["accuracy"] for r in datos)
    mediana = (accs[len(accs) // 2] if len(accs) % 2
               else (accs[len(accs) // 2 - 1] + accs[len(accs) // 2]) / 2)
    lider = min(datos, key=lambda r: r["rank"])

    sb.upsert("leaderboard_snapshots", [{
        "captured_at": mio.get("calculated_at"),
        "rank": mio["rank"],
        "participantes": len(datos),
        "accuracy": mio["accuracy"],
        "coverage": mio["coverage"],
        "raw_wape": mio.get("raw_wape"),
        "accuracy_at_20": mio.get("accuracy_at_20"),
        "lider_accuracy": lider["accuracy"],
        "lider_coverage": lider["coverage"],
        "mediana_accuracy": mediana,
    }], conflicto="captured_at")
    print(f"leaderboard: puesto {mio['rank']}/{len(datos)}  "
          f"accuracy {mio['accuracy']:.2f}  cobertura {mio['coverage']:.2f}")


# --------------------------------------------------------------------- decisión

def decidir(sb: Supabase, run_id: int, filas_senales: list[dict],
            ficha: dict, obs: pd.DataFrame, ahora: str) -> dict:
    rotas = sorted({f["signal"] for f in filas_senales if f["breached"]})
    wape_roto = any(f["signal"] == "wape_24h" and f["station_id"] is None
                    and f["breached"] for f in filas_senales)

    # Persistencia: corridas seguidas con wape_24h global en rojo, contando
    # ésta. Se cuenta hacia atrás y se corta en la primera verde.
    previas = [f for f in sb.seleccionar(
        "drift_signals", select="run_id,breached,computed_at",
        signal="eq.wape_24h", station_id="is.null",
        order="computed_at.desc", limit=10) if f["run_id"] != run_id]
    seguidas = 0
    if wape_roto:
        seguidas = 1
        for f in previas:
            if f["breached"]:
                seguidas += 1
            else:
                break

    nuevas = int((obs["observed_at"] > pd.Timestamp(ficha["train_end"])).sum())

    # El enfriamiento se busca por la última fila QUE TENGA uno, no por la
    # última fila a secas. Sólo las decisiones `retrain` escriben
    # `cooldown_until`; si se mira la más reciente sin filtrar, la primera
    # `blocked` que caiga encima —con el campo en NULL— borra la memoria del
    # enfriamiento y habilita un reentrenamiento a los pocos minutos. Pasó:
    # run 18 enfrió hasta las 20:29, run 22 bloqueó bien, y run 23 volvió a
    # decidir `retrain` a los nueve minutos.
    ult = sb.seleccionar("retrain_decisions",
                         select="decision,cooldown_until,decided_at",
                         cooldown_until="not.is.null",
                         order="cooldown_until.desc", limit=1)
    ahora_dt = pd.Timestamp(datetime.now(timezone.utc))
    enfriando = bool(ult and ult[0]["cooldown_until"]
                     and pd.Timestamp(ult[0]["cooldown_until"]) > ahora_dt)

    cooldown = None
    if not rotas:
        decision = "keep"
        motivo = "ninguna señal por fuera de umbral"
    elif not wape_roto:
        decision = "keep"
        motivo = ("alerta temprana sin caída de desempeño (" + ", ".join(rotas) +
                  "). Se vigila; el WAPE de 24 h sigue dentro de umbral.")
    elif enfriando:
        decision = "blocked"
        motivo = (f"wape_24h en rojo, pero hay enfriamiento vigente hasta "
                  f"{ult[0]['cooldown_until']}. Reentrenar en cadena no arregla nada.")
    elif seguidas < CICLOS_PERSISTENCIA:
        decision = "blocked"
        motivo = (f"wape_24h en rojo {seguidas} corrida(s) seguida(s); se exigen "
                  f"{CICLOS_PERSISTENCIA}. Un mal periodo no es drift.")
    elif nuevas < MIN_OBS_NUEVAS:
        decision = "blocked"
        motivo = (f"sólo {nuevas} observaciones nuevas desde el corte de "
                  f"entrenamiento; se exigen {MIN_OBS_NUEVAS} (~12 h) para que "
                  f"la validación temporal signifique algo.")
    else:
        decision = "retrain"
        cooldown = (datetime.now(timezone.utc) +
                    timedelta(hours=COOLDOWN_HORAS)).isoformat()
        motivo = (f"wape_24h sobre {UMBRAL_WAPE} en {seguidas} corridas seguidas "
                  f"con {nuevas} observaciones nuevas. Señales rotas: "
                  f"{', '.join(rotas)}.")

    return {"run_id": run_id, "decision": decision, "reason": motivo[:2000],
            "breached_signals": rotas, "incumbent_model_id": ficha["model_id"],
            "cooldown_until": cooldown, "decided_at": ahora}


# ------------------------------------------------------------------------- main

def anunciar(decision: str) -> None:
    """Deja la decisión donde el workflow pueda leerla.

    Fuera de Actions no hace nada: `GITHUB_OUTPUT` no existe y la decisión ya
    quedó impresa y guardada en `retrain_decisions`, que es la fuente.
    """
    import os
    destino = os.environ.get("GITHUB_OUTPUT")
    if destino:
        with open(destino, "a", encoding="utf-8") as f:
            f.write(f"decision={decision}\n")


def main(dry_run: bool) -> None:
    base, key = exigir("PULSO_API_BASE", "PULSO_API_KEY")
    sb = Supabase()
    ahora = datetime.now(timezone.utc).isoformat()

    activos = sb.seleccionar("v_active_model", select="*", limit=1)
    if not activos:
        raise SystemExit("no hay modelo con status=active en la tabla models")
    ficha = activos[0]
    print(f"modelo  : {ficha['name']}  ({ficha['hyperparams'].get('version')})")

    scores = traer_scores(sb, ficha["model_id"])
    if scores.empty:
        # Pasa tras cada promoción: el champion nuevo no tiene nada resuelto
        # hasta que el stream alcanza sus primeros targets (~1-2 h). Salir en
        # silencio dejaba el panel en "atrasado" sin distinguir "no hay qué
        # juzgar" de "el evaluador no corre". Se deja constancia con `keep`:
        # no reentrenar un modelo que todavía no se puede juzgar. No toca la
        # racha (se cuenta en `drift_signals`) ni el enfriamiento (sólo mira
        # filas con `cooldown_until`).
        motivo = (f"{ficha['name']} ({ficha['hyperparams'].get('version')}) aún no "
                  f"tiene predicciones con realidad observada; se espera evidencia "
                  f"antes de juzgarlo.")
        print(f"nada que evaluar: {motivo}")
        if dry_run:
            return
        run_id = abrir_run(sb)
        sb.upsert("retrain_decisions", [{
            "run_id": run_id, "decision": "keep", "reason": motivo,
            "breached_signals": [], "incumbent_model_id": ficha["model_id"],
            "cooldown_until": None, "decided_at": ahora}], conflicto="run_id")
        ingestar_leaderboard(sb, base, key)
        cerrar_run(sb, run_id, "success")
        return

    print(f"scores  : {len(scores)} pares predicción/realidad  "
          f"({scores['target_at'].min()} -> {scores['target_at'].max()})")

    for fold, ancho in VENTANAS.items():
        v = ventana(scores, ancho)
        if not v.empty:
            _, g = metricas(v)
            etiqueta = {0: "acumulado", 1: "24 h", 7: "7 d"}[fold]
            print(f"  {etiqueta:<10} accuracy {g['accuracy']:6.2f} %  "
                  f"wape {g['wape']:.4f}  sesgo {g['bias']:+.2f}  n={len(v)}")

    obs = traer_observaciones(sb)
    print(f"obs     : {len(obs)} filas  (última {obs['observed_at'].max()})")

    run_id = 0 if dry_run else abrir_run(sb)
    filas_m = filas_metricas(scores, ficha["model_id"], ahora)
    filas_s = senales(run_id, scores, obs, ficha, base, key, ahora)

    print(f"\nseñales : {len(filas_s)} calculadas")
    for sig in sorted({f["signal"] for f in filas_s}):
        grupo = [f for f in filas_s if f["signal"] == sig]
        glob = next((f for f in grupo if f["station_id"] is None), None)
        por_est = [f for f in grupo if f["station_id"] is not None]
        rojas = sum(1 for f in por_est if f["breached"])
        # La global puede ir en verde y aun así haber estaciones en rojo: la
        # guía pide mirar "degradación concentrada", no sólo el promedio.
        marca = "ROTO" if glob and glob["breached"] else "ok  "
        print(f"  {sig:<16} {glob['value']:+9.4f}  (umbral {glob['threshold']})"
              f"  {marca}  estaciones en rojo: {rojas}/{len(por_est)}")

    decision = decidir(sb, run_id, filas_s, ficha, obs, ahora)
    print(f"\ndecisión: {decision['decision'].upper()}")
    print(f"motivo  : {decision['reason']}")

    anunciar(decision["decision"])

    if dry_run:
        print(f"\n[dry-run] no se escribió nada "
              f"({len(filas_m)} métricas y {len(filas_s)} señales quedaron sin guardar)")
        return

    try:
        for i in range(0, len(filas_m), 500):
            sb.upsert("model_metrics", filas_m[i:i + 500],
                      conflicto="model_id,split,fold,station_id,metric")
        for i in range(0, len(filas_s), 500):
            sb.upsert("drift_signals", filas_s[i:i + 500],
                      conflicto="run_id,station_id,signal")
        sb.upsert("retrain_decisions", [decision], conflicto="run_id")
        ingestar_leaderboard(sb, base, key)
        cerrar_run(sb, run_id, "success",
                   cutoff_at=scores["target_at"].max().isoformat())
        print(f"\nguardado: run_id {run_id}  ·  {len(filas_m)} métricas  ·  "
              f"{len(filas_s)} señales  ·  decisión {decision['decision']}")
    except Exception as e:
        cerrar_run(sb, run_id, "failed", error_stage="evaluacion",
                   error_message=str(e)[:1000])
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
