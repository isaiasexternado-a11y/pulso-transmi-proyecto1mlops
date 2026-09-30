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

Cada receta se mide como la ENTREGARÍA producción: a su predicción se le
aplican las capas de la ficha del champion vigente (mezcla con persistencia y
corrección de nivel), las mismas que un ganador hereda al registrarse. Medir
recetas crudas comparaba algo que nadie entrega: el 2026-09-28 el champion
daba 85,4 % crudo en el backtest y 86 % en vivo con sus capas.

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
    python3 -m ml.entrenar --dry-run              # compara y reporta, no registra nada
    python3 -m ml.entrenar --dry-run --sin-capas  # la comparación vieja, recetas crudas
    python3 -m ml.entrenar --dry-run --sin-vivo   # sin la prueba de las últimas 24 h
    python3 -m ml.entrenar                        # registra al ganador como candidate

Además de los folds, cada corrida mide las últimas 24 h (`medir_vivo`). Con eso
hay dos rutas más hacia `candidate`: cambiar de receta si otra gana en lo vivo,
o refrescar la receta del champion con los datos más nuevos. `train.yml` corre
cada 2 h, así que el champion ve el régimen actual en horas, no en días.
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
import numpy as np
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
from models import (Columna, GBMconPerfil, Perfil,          # noqa: E402
                    PerfilMasTendencia)

BUCKET = "modelos"
DIAS_VALIDACION = 7
N_FOLDS = 3

# --------------------------------------------------------------- compuertas
# Un candidato no reemplaza al champion por empatarle. El margen existe para
# que el ruido de tres folds no se disfrace de mejora.
MARGEN_CHAMPION = 0.15      # puntos de accuracy sobre el champion
TOLERANCIA_ESTACION = 2.0   # ninguna estación puede empeorar más que esto

# Veto por lo reciente. Los folds pesan igual —ponderar fuerte pocos días de
# drift dejaría ganar por suerte y la promoción es automática—, pero un
# candidato que no le gana al champion en los últimos días, donde vive el
# régimen actual, no entra. Lo reciente puede vetar, no hacer ganar.
DIAS_RECIENTES = 3

PASO = pd.to_timedelta(15, unit="min")
CAPAS = ("mezcla_persistencia", "correccion_nivel")


# ------------------------------------------------------------------ recetas

class Receta:
    """Una forma de entrenar: qué modelo, con qué features, con qué ventana."""

    def __init__(self, etiqueta, crear, dias=None, es_champion=False):
        self.etiqueta = etiqueta
        self.crear = crear
        self.dias = dias                 # None = todo el histórico
        self.es_champion = es_champion


def nombrar(m, nombre: str):
    """Renombra una instancia sin subclasear.

    `nombre` es atributo de clase en algunos modelos; al fijarlo en la
    instancia se distinguen dos ventanas de la misma familia en `models.name`
    sin introducir una clase que `main` no conozca, que rompería la carga del
    artefacto en producción.
    """
    m.nombre = nombre
    return m


# Etiqueta de la receta del champion activo. `main` la fija desde la ficha
# (`hyperparams.receta`); si la ficha no la trae —el champion se empaquetó a
# mano—, la receta marcada `es_champion` en la lista.
RECETA_CHAMPION: str | None = None


def recetas() -> list[Receta]:
    lista = _recetas()
    if RECETA_CHAMPION and any(r.etiqueta == RECETA_CHAMPION for r in lista):
        for r in lista:
            r.es_champion = r.etiqueta == RECETA_CHAMPION
    return lista


def _recetas() -> list[Receta]:
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

        # El perfil corregido por nivel reciente: toma lo que el perfil
        # esperaba para la hora del origen, lo compara con lo que de verdad
        # está pasando, y escala. Perdió por casi 5 puntos en el backtest
        # original —pero ese backtest corrió sobre un periodo SIN drift, donde
        # no había nivel que corregir. Entra ahora porque el problema que se
        # observa es exactamente el suyo: `profile_corr` en 0,98 (la forma del
        # día intacta) con estaciones subestimadas en ~20 % (el nivel movido).
        # Que haya perdido antes no dice nada sobre cómo se comporta ahora;
        # descartarla por eso fue prejuzgar con datos de otro régimen.
        Receta("perfil x ajuste reciente",
               lambda: PerfilMasTendencia(), None),
        Receta("perfil x ajuste reciente · 14d",
               lambda: nombrar(PerfilMasTendencia(),
                               "perfil x ajuste reciente (14d)"), 14),

        # Normalizado por nivel reciente: predice demanda / escala y las
        # variables de nivel entran divididas por la misma escala. Es lo que
        # comparten los dos primeros del leaderboard de 24 h el 2026-09-29: un
        # salto de nivel mueve la escala y el modelo lo acompaña sin esperar a
        # la corrección de la ficha ni a un reentrenamiento.
        Receta("gbm + perfil normalizado 24h",
               lambda: GBMconPerfil("gbm + perfil normalizado 24h (mae)", cols=sc,
                                    escalas=("roll24h",)), None),
        Receta("gbm + perfil normalizado 3 escalas",
               lambda: GBMconPerfil("gbm + perfil normalizado 3 escalas (mae)", cols=sc,
                                    escalas=("roll24h", "roll1h", "roll7d")), None),
        Receta("gbm + perfil normalizado 3 escalas · 14d",
               lambda: GBMconPerfil("gbm + perfil normalizado 3 escalas (mae)", cols=sc,
                                    escalas=("roll24h", "roll1h", "roll7d")), 14),

        # Pesos por recencia (revisión 2 del drift, 30-sep: cambia la FORMA de
        # la demanda). Con 14 días parejos el modelo aprende sobre todo la
        # forma vieja; con semivida cada fila pesa 0,5^(edad / semivida) y lo
        # último manda sin tirar la historia. En un cambio de forma sintético
        # (`ml/experimento_forma.py`) le ganan ~5 puntos a la misma receta
        # sin pesos en el régimen estable.
        Receta("gbm + perfil sin contexto · 14d semivida 12h",
               lambda: GBMconPerfil("gbm + perfil sin contexto semivida 12h (mae)",
                                    cols=sc, semivida_h=12), 14),
        Receta("gbm + perfil sin contexto · 14d semivida 24h",
               lambda: GBMconPerfil("gbm + perfil sin contexto semivida 24h (mae)",
                                    cols=sc, semivida_h=24), 14),
        Receta("gbm + perfil normalizado 3 escalas · 14d semivida 12h",
               lambda: GBMconPerfil("gbm + perfil normalizado 3 escalas semivida 12h (mae)",
                                    cols=sc, escalas=("roll24h", "roll1h", "roll7d"),
                                    semivida_h=12), 14),
        Receta("gbm + perfil normalizado 3 escalas · 14d semivida 24h",
               lambda: GBMconPerfil("gbm + perfil normalizado 3 escalas semivida 24h (mae)",
                                    cols=sc, escalas=("roll24h", "roll1h", "roll7d"),
                                    semivida_h=24), 14),

        # Un modelo por estación. Quedó a 0,24 del champion, lo bastante cerca
        # como para que un cambio de régimen pueda darle la vuelta.
        Receta("gbm + perfil por estacion",
               lambda: GBMconPerfil("gbm + perfil por estacion (mae)",
                                    por_estacion=True), None),
    ]


# ------------------------------------------------------------ capas de la ficha

# El modo quiebre de la corrección de nivel (`correccion_nivel.quiebre`) se
# retira el 2026-09-30. Corrige COMPLETO cuando r de 2 h y 4 h se desvían en el
# mismo sentido, y ante un cambio de forma (la revisión 2 del drift) eso
# amplifica errores de fase como si fueran de nivel. En `experimento_forma`
# la corrección sin quiebre le ganó a la con quiebre en 12 de 12 comparaciones
# (6 escenarios x 2 recetas), incluida la ancla que contiene la continuación
# real del 16 virtual, y a no corregir en 12 de 12.
SIN_QUIEBRE = True


def _sin_quiebre(nivel: dict | None) -> dict | None:
    if not nivel or not SIN_QUIEBRE:
        return nivel
    return {k: v for k, v in nivel.items() if k != "quiebre"}


def capas_vigentes(sb: Supabase) -> dict:
    """Las capas del champion activo: lo que un ganador heredaría."""
    fila = sb.seleccionar("models", select="hyperparams", status="eq.active", limit=1)
    hp = (fila[0]["hyperparams"] or {}) if fila else {}
    capas = {k: hp[k] for k in CAPAS if hp.get(k)}
    if "correccion_nivel" in capas:
        capas["correccion_nivel"] = _sin_quiebre(capas["correccion_nivel"])
    return capas


def nivel_guardado(sb: Supabase) -> dict | None:
    """La corrección de nivel del champion, esté prendida o apagada.

    Apagada vive en `correccion_nivel_apagada`: `pipeline/entregar.py` sólo lee
    `correccion_nivel`, así que no se aplica, pero la receta sigue en la ficha
    para que un reentreno posterior pueda volver a prenderla si en lo vivo
    vuelve a ganar.
    """
    fila = sb.seleccionar("models", select="hyperparams", status="eq.active", limit=1)
    hp = (fila[0]["hyperparams"] or {}) if fila else {}
    return _sin_quiebre(hp.get("correccion_nivel") or hp.get("correccion_nivel_apagada"))


def alternar_nivel(capas: dict, nivel: dict | None) -> dict | None:
    """Las mismas capas con la corrección de nivel en el estado contrario."""
    if not nivel:
        return None
    if "correccion_nivel" in capas:
        return {k: v for k, v in capas.items() if k != "correccion_nivel"}
    return {**capas, "correccion_nivel": nivel}


def aplicar_capas(val: pd.DataFrame, pred, capas: dict) -> np.ndarray:
    """Lo que la receta habría entregado con las capas del champion.

    Es `pipeline/entregar.py::predecir` vectorizado sobre el backtest:

    - mezcla: `(1 − w_h) · pred + w_h · lag0`, con `lag0` el último observado
      en el origen.
    - nivel: por estación, `r = Σ y / Σ max(entregado, 0)` sobre los objetivos
      ya resueltos en (origen − ventana, origen], acotado, y el valor queda en
      `entregado · (1 + α (r − 1))`. Los objetivos de la ventana son las
      predicciones de la propia receta en orígenes previos del mismo fold,
      como los backcasts de producción. Al arrancar cada fold la ventana está
      incompleta y en el primer origen vacía (factor 1): son las primeras
      horas de siete días.

    Sólo usa `y` de objetivos con `target_at <= origen`: nada del futuro.
    """
    out = np.asarray(pred, dtype=float)
    mezcla = capas.get("mezcla_persistencia")
    if mezcla:
        w = val["horizon"].map(lambda h: float(mezcla[str(int(h))])).to_numpy()
        out = (1 - w) * out + w * val["lag0"].to_numpy(dtype=float)

    nivel = capas.get("correccion_nivel")
    if not nivel:
        return out
    d = pd.DataFrame({"station_id": val["station_id"].to_numpy(),
                      "target_at": pd.DatetimeIndex(val["target_at"]),
                      "origen": pd.DatetimeIndex(val["target_at"] - val["horizon"] * PASO),
                      "y": val["y"].to_numpy(dtype=float),
                      "base": np.clip(out, 0, None)})
    L = int(nivel["ventana_h"])
    lo, hi = float(nivel["r_min"]), float(nivel["r_max"])
    alfa = float(nivel["alfa"])
    quiebre = nivel.get("quiebre")
    factor = pd.Series(1.0, index=d.index)
    for est, g in d.groupby("station_id"):
        serie = g.groupby("target_at")[["y", "base"]].first().sort_index()

        def razon(horas):                                          # (t − L, t]
            suma = serie.rolling(pd.to_timedelta(horas, unit="h"), closed="right").sum()
            r = (suma["y"] / suma["base"]).replace([np.inf, -np.inf], np.nan)
            return r.reindex(g["origen"]).to_numpy()

        corto = razon(L)
        f = 1 + alfa * (np.clip(np.nan_to_num(corto, nan=1.0), lo, hi) - 1)
        if quiebre:
            # Mismo criterio que `pipeline/entregar.py::factores_nivel`.
            largo, u = razon(2 * L), float(quiebre["umbral"])
            q = ((np.abs(corto - 1) > u) & (np.abs(largo - 1) > u)
                 & (np.sign(corto - 1) == np.sign(largo - 1)))
            f = np.where(q, np.clip(np.nan_to_num(corto, nan=1.0),
                                    float(quiebre["r_min"]), float(quiebre["r_max"])), f)
        factor[g.index] = f
    return out * factor.to_numpy()


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


def comparar(ancha, ctx, n_folds: int, capas: dict) -> dict[str, dict]:
    todas = recetas()
    recientes = ancha.index[-1] - pd.to_timedelta(DIAS_RECIENTES, unit="D")
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
            entregado = aplicar_capas(val, m.predict(val), capas)
            res = resumen(val.assign(y_pred=entregado))
            ult = val["target_at"] > recientes
            if ult.any():
                res["reciente"] = resumen(val[ult].assign(y_pred=entregado[ult.to_numpy()]))["accuracy"]
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
            "reciente": next((x["reciente"] for x in reversed(rs) if "reciente" in x), None),
            "folds": rs,
        }
    return tabla


# ------------------------------------------------------------ el régimen vivo
# Los folds de 7 días diluyen un cambio de régimen de horas: el 2026-09-29 la
# continuación del escenario era 15 h de 21 días y ningún candidato pasaba,
# aunque reentrenar cada 6 h le ganaba al GBM congelado en 14 de 20 ciclos de
# la continuación (`ml/experimento_reentreno.py`). Esta prueba mide sólo lo
# último: cada receta se entrena con lo resuelto hasta `fin − HORAS_VIVO` y
# predice los ciclos horarios de ahí en adelante, con las capas del champion.

HORAS_VIVO = 24
# Interruptor de la corrección de nivel. Es una corrección de ESCALA, y la
# revisión 2 del drift (30-sep) cambia la FORMA: el profesor advierte que la
# referencia que sólo ajustaba la escala dejó de servir, y en el cambio de
# forma sintético (`ml/experimento_forma.py`) le resta al régimen estable.
# Cada corrida mide cada receta con y sin ella en lo vivo; se cambia de estado
# sólo si el otro gana por este margen, para no parpadear entre corridas.
MARGEN_CAPAS = 0.10
CALENTAR_H = 5              # 2 · ventana de nivel (2 h) + holgura
MARGEN_VIVO = 0.10          # para cambiar de receta hay que ganar esto en lo vivo
# Era 0,30. Se bajó el 2026-09-30: todo elegible ya pasó las compuertas de
# historia, así que cambiar por poco no arriesga; 0,10 filtra el ruido de
# elegir al mejor de ~14 recetas en una sola ventana de 24 h.

# Reserva: las últimas horas no entran al entrenamiento. La corrección de nivel
# compara lo observado contra backcasts del propio modelo en las últimas 2-4 h;
# si el modelo acaba de entrenar con esas horas las reproduce casi exactas, el
# factor sale ~1 y la corrección se apaga. Medido en la continuación con
# reentreno cada 2 h: sin reserva los GBM caían por debajo del congelado
# (`ml/experimento_reentreno.py --cada 2`).
RESERVA_H = 2
# Bajó de 6 a 2 el 2026-09-30 con la revisión 2 del drift. Con pesos por
# recencia las horas más nuevas son las que más enseñan, y en el cambio de
# forma sintético (`ml/experimento_forma.py`, 2 anclas x 3 escenarios) la
# reserva de 2 h le ganó a la de 6 en todas las recetas y capas medidas. El
# riesgo de backcasts in-sample era del modo quiebre, que también se retira.
TOLERANCIA_HISTORIA = 0.30  # y no perder más que esto en los folds


def medir_vivo(sb: Supabase, ancha, ctx, capas: dict, nivel: dict | None = None) -> dict:
    """Accuracy de cada receta en las últimas HORAS_VIVO horas, y la vara.

    La vara es lo que se entregaría sin promover nada: el artefacto activo,
    siempre que se haya entrenado antes del corte. Si se entrenó después (un
    refresco reciente), en esta ventana estaría prediciendo datos que ya vio y
    ganaría por memoria; entonces la vara es su receta entrenada al corte, en
    las mismas condiciones que los rivales.
    """
    from pipeline.entregar import champion

    fin = ancha.index[-1]
    corte = fin - pd.to_timedelta(HORAS_VIVO, unit="h")
    o_val = origenes_por_hora(ancha, corte - pd.to_timedelta(CALENTAR_H, unit="h"), fin)
    val = construir(ancha, ctx, o_val).dropna(subset=["y"])
    evaluar = ((val["target_at"] - val["horizon"] * PASO) >= corte).to_numpy()
    o_train = origenes_por_hora(ancha, ancha.index[0], corte)
    train_full = construir(ancha, ctx, o_train).dropna()
    train_full = train_full[train_full["target_at"]
                            <= corte - pd.to_timedelta(RESERVA_H, unit="h")]

    def medir(pred) -> float:
        entregado = aplicar_capas(val, pred, capas)
        return resumen(val[evaluar].assign(y_pred=entregado[evaluar]))["accuracy"]

    print(f"\nvivo: entrenado hasta {corte:%m-%d %H:%M}, "
          f"{int(evaluar.sum() / 48)} ciclos evaluados")
    otras = alternar_nivel(capas, nivel)
    con_nivel_hoy = "correccion_nivel" in capas

    def medir_con(pred, cs) -> float:
        entregado = aplicar_capas(val, pred, cs)
        return resumen(val[evaluar].assign(y_pred=entregado[evaluar]))["accuracy"]

    por_receta, nivel_por_receta = {}, {}
    for r in recetas():
        tr = train_full
        if r.dias is not None:
            tr = tr[tr["target_at"] > corte - pd.to_timedelta(r.dias, unit="D")]
        pred = r.crear().fit(tr).predict(val)
        hoy = medir(pred)
        por_receta[r.etiqueta], nivel_por_receta[r.etiqueta] = hoy, con_nivel_hoy
        nota = ""
        if otras is not None:
            alt = medir_con(pred, otras)
            nota = f"   {'sin' if con_nivel_hoy else 'con'} nivel {alt:6.2f}"
            if alt >= hoy + MARGEN_CAPAS:
                por_receta[r.etiqueta], nivel_por_receta[r.etiqueta] = alt, not con_nivel_hoy
                nota += "  <- cambia"
        print(f"   {r.etiqueta:40} vivo {hoy:6.2f}{nota}")

    modelo, ficha = champion(sb)
    if pd.Timestamp(ficha["train_end"]) <= corte:
        vara, fuente = medir(modelo.predict(val)), "artefacto activo"
    else:
        etiqueta = next(r.etiqueta for r in recetas() if r.es_champion)
        vara, fuente = por_receta[etiqueta], f"receta del champion al corte ({etiqueta})"
    print(f"   {'vara: ' + fuente:40} vivo {vara:6.2f}")
    return {"corte": str(corte), "ciclos": int(evaluar.sum() / 48),
            "por_receta": por_receta, "nivel_por_receta": nivel_por_receta,
            "vara": vara, "fuente_vara": fuente}


# ------------------------------------------------------------------ compuertas

def elegir(tabla: dict, vivo: dict | None = None) -> tuple[str | None, list[str], str]:
    """Devuelve (etiqueta ganadora o None, razones, ruta).

    Tres rutas deciden quién es ELEGIBLE:

    - `historia`: gana en los folds con margen (las compuertas de siempre).
    - `cambio de receta`: gana en lo vivo por MARGEN_VIVO sin perder más de
      TOLERANCIA_HISTORIA en los folds ni hundir una estación.
    - `refresco`: la receta del champion con los datos más nuevos. Entra si
      en lo vivo no pierde más de TOLERANCIA_HISTORIA contra la vara. Es lo
      que hace el reentreno periódico: el mismo modelo, visto lo último.

    Entre los elegibles gana el mejor en lo vivo. Las rutas ya garantizan que
    nadie empeora la historia; lo vivo decide cuál sirve para el régimen de
    hoy. Sin prueba viva queda sólo la ruta `historia`, como antes.
    """
    etiqueta, razones = _elegir_historia(tabla)
    if not vivo:
        return etiqueta, razones, "historia"

    elegibles = {etiqueta: "historia"} if etiqueta else {}
    champion = next((k for k, v in tabla.items() if v["receta"].es_champion), None)
    if champion is None:
        return etiqueta, razones, "historia"
    ref = tabla[champion]
    pv, vara = vivo["por_receta"], vivo["vara"]

    for k in (k for k in tabla if k != champion and k in pv and k not in elegibles):
        c = tabla[k]
        fallos = []
        if pv[k] < vara + MARGEN_VIVO:
            fallos.append(f"en lo vivo {pv[k]:.2f} vs vara {vara:.2f}, se exigen {MARGEN_VIVO:+.2f}")
        if c["accuracy_media"] < ref["accuracy_media"] - TOLERANCIA_HISTORIA:
            fallos.append(f"en los folds pierde {c['accuracy_media'] - ref['accuracy_media']:+.2f}")
        hundidas = [e for e, v in c["por_estacion"].items()
                    if ref["por_estacion"].get(e, 0) - v > TOLERANCIA_ESTACION]
        if hundidas:
            fallos.append(f"hunde {', '.join(hundidas[:3])}")
        if fallos:
            razones.append(f"{k} [vivo]: {'; '.join(fallos)}")
        else:
            elegibles[k] = "cambio de receta"
            razones.append(f"{k}: elegible por cambio de receta, vivo {pv[k]:.2f}")

    if champion in pv and pv[champion] >= vara - TOLERANCIA_HISTORIA:
        elegibles.setdefault(champion, "refresco")
        razones.append(f"{champion}: elegible por refresco, vivo {pv[champion]:.2f}")
    else:
        razones.append(f"{champion} [refresco]: en lo vivo "
                       f"{pv.get(champion, float('nan')):.2f} vs vara {vara:.2f}")

    if not elegibles:
        return None, razones, "historia"
    mejor = max(elegibles, key=lambda k: pv.get(k, float("-inf")))
    razones.append(f"entre {len(elegibles)} elegibles gana en lo vivo {mejor}: "
                   f"{pv[mejor]:.2f} vs vara {vara:.2f} ({vivo['fuente_vara']})")
    return mejor, razones, elegibles[mejor]


def _elegir_historia(tabla: dict) -> tuple[str | None, list[str]]:
    """Las compuertas de siempre: ganar en los folds con margen."""
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

        if cand["reciente"] is not None and ref["reciente"] is not None \
                and cand["reciente"] <= ref["reciente"]:
            fallos.append(f"no le gana al champion en los últimos {DIAS_RECIENTES} días "
                          f"({cand['reciente']:.2f} vs {ref['reciente']:.2f})")

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

def registrar(sb: Supabase, ancha, ctx, ganador: dict, tabla: dict,
              ruta_promocion: str = "historia", vivo: dict | None = None) -> dict:
    """Reentrena al ganador con todo lo disponible, lo sube y lo inscribe."""
    r = ganador["receta"]
    fin = ancha.index[-1]
    o_full = origenes_por_hora(ancha, ancha.index[0], fin)
    full = construir(ancha, ctx, o_full).dropna()
    full = full[full["target_at"] <= fin - pd.to_timedelta(RESERVA_H, unit="h")]
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

    previo = sb.seleccionar("models", select="model_id,hyperparams",
                            status="eq.active", limit=1)
    # La mezcla con persistencia y la corrección de nivel viven en la ficha,
    # no en el pickle. Si el ganador no las hereda, promoverlo las apaga sin
    # que nadie lo decida. El backtest ya lo midió CON esas capas, así que
    # heredarlas es entregar exactamente lo que ganó.
    hp_previo = (previo[0]["hyperparams"] or {}) if previo else {}
    mezcla = hp_previo.get("mezcla_persistencia")
    nivel = _sin_quiebre(hp_previo.get("correccion_nivel") or hp_previo.get("correccion_nivel_apagada"))
    # Prendida o apagada según lo que ganó en lo vivo (`medir_vivo`); sin
    # prueba viva, como estaba en el champion.
    nivel_on = (vivo or {}).get("nivel_por_receta", {}).get(
        r.etiqueta, "correccion_nivel" in hp_previo)
    sin_contexto = not any(c.startswith(("rain", "temp", "evento")) for c in cols)

    fila = {
        "name": modelo.nombre,
        "kind": "baseline" if modelo.es_baseline else "ml",
        "algorithm": type(modelo).__name__,
        "feature_set": "fs-v2-sin-contexto" if sin_contexto else "fs-v1",
        "feature_list": cols,
        "train_start": str(full["target_at"].min()),
        "train_end": str(full["target_at"].max()),
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
            "reserva_h": RESERVA_H,
            "receta": r.etiqueta,
            "validacion_accuracy": ganador["accuracy_media"],
            "validacion_peor_fold": ganador["accuracy_min"],
            "validacion_reciente": ganador["reciente"],
            "ruta_promocion": ruta_promocion,
            **({"validacion_vivo": {"accuracy": vivo["por_receta"].get(r.etiqueta),
                                    "vara": vivo["vara"], "fuente_vara": vivo["fuente_vara"],
                                    "corte": vivo["corte"], "ciclos": vivo["ciclos"]}}
               if vivo else {}),
            "validacion_con_capas": [k for k in CAPAS if k in (previo[0]["hyperparams"] or {})]
                                    if previo else [],
            "por_fold": ganador["por_fold"],
            "champion_previo_accuracy": next(
                (v["accuracy_media"] for v in tabla.values()
                 if v["receta"].es_champion), None),
            "folds": N_FOLDS,
            "dias_validacion": DIAS_VALIDACION,
            **({"mezcla_persistencia": mezcla} if mezcla else {}),
            **({("correccion_nivel" if nivel_on else "correccion_nivel_apagada"): nivel}
               if nivel else {}),
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

def main(dry_run: bool, n_folds: int, sin_capas: bool, sin_vivo: bool = False) -> None:
    global RECETA_CHAMPION
    sb = Supabase()
    capas = {} if sin_capas else capas_vigentes(sb)
    activo = sb.seleccionar("models", select="hyperparams", status="eq.active", limit=1)
    RECETA_CHAMPION = ((activo[0]["hyperparams"] or {}).get("receta") if activo else None)
    print(f"receta del champion: {RECETA_CHAMPION or 'la marcada en la lista'}")
    print(f"capas    : {', '.join(capas) or 'ninguna (recetas crudas)'}")
    ancha, ctx, _ = cargar(origen="supabase")
    print(f"panel: {ancha.shape[0]:,} periodos x {ancha.shape[1]} estaciones "
          f"({ancha.index[0]:%Y-%m-%d} -> {ancha.index[-1]:%Y-%m-%d %H:%M})")

    tabla = comparar(ancha, ctx, n_folds, capas)

    print("\n" + "=" * 86)
    print(f"{'receta':40} {'media':>7} {'peor fold':>10} {f'últ. {DIAS_RECIENTES} d':>10}")
    print("-" * 86)
    for k, v in sorted(tabla.items(), key=lambda x: -x[1]["accuracy_media"]):
        marca = "<- champion" if v["receta"].es_champion else ""
        rec = f"{v['reciente']:10.2f}" if v["reciente"] is not None else f"{'—':>10}"
        print(f"{k:40} {v['accuracy_media']:7.2f} {v['accuracy_min']:10.2f} {rec}  {marca}")
    print("=" * 86)

    vivo = None if sin_vivo else medir_vivo(sb, ancha, ctx, capas,
                                            None if sin_capas else nivel_guardado(sb))
    etiqueta, razones, ruta = elegir(tabla, vivo)
    print("\ncompuertas:")
    for r in razones:
        print(f"  · {r}")

    if etiqueta is None:
        print("\nNINGÚN CANDIDATO PASA. El champion se queda.")
        anunciar(model_id="", version="")
        return

    print(f"\nGANADOR: {etiqueta}  (ruta: {ruta})")
    if dry_run:
        print("[dry-run] no se registró nada")
        anunciar(model_id="", version="")
        return

    creado = registrar(sb, ancha, ctx, tabla[etiqueta], tabla, ruta, vivo)
    anunciar(model_id=creado["model_id"],
             version=creado["hyperparams"]["version"])
    print("\nlisto: el candidato queda registrado, NO promovido. "
          "La promoción la decide pipeline/promover.py desde main.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--sin-capas", action="store_true",
                    help="comparar recetas crudas, como antes del 2026-09-29")
    ap.add_argument("--sin-vivo", action="store_true",
                    help="sólo las compuertas de los folds, sin cambio de receta ni refresco")
    a = ap.parse_args()
    main(a.dry_run, a.folds, a.sin_capas, a.sin_vivo)
