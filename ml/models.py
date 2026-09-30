"""Candidatos a champion.

Todos exponen la misma interfaz —`fit(train)` y `predict(frame)`— para que el
backtest los mida exactamente igual. Agregar un modelo nuevo es añadirlo a
`CANDIDATOS`; no hay que tocar el evaluador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import clip_no_negativo


class Modelo:
    nombre = "base"
    es_baseline = False

    def fit(self, train: pd.DataFrame) -> "Modelo":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Columna(Modelo):
    """Baselines que son, literalmente, una columna de rezago."""
    es_baseline = True

    def __init__(self, nombre: str, col: str):
        self.nombre = nombre
        self.col = col

    def predict(self, frame):
        return clip_no_negativo(frame[self.col])


class Perfil(Modelo):
    """Promedio histórico por estación, slot y tipo de día."""
    es_baseline = True

    def __init__(self, nombre="perfil (estacion x slot x finde)", claves=("station_id", "slot", "es_finde")):
        self.nombre = nombre
        self.claves = list(claves)

    def fit(self, train):
        self.tabla_ = train.groupby(self.claves)["y"].mean().rename("p")
        self.global_ = float(train["y"].mean())
        return self

    def predict(self, frame):
        p = frame[self.claves].join(self.tabla_, on=self.claves)["p"]
        return clip_no_negativo(p.fillna(self.global_))


class PerfilMasTendencia(Modelo):
    """Perfil ajustado por cuánto se desvía el día de hoy de su propio perfil.

    Captura que una jornada puede venir globalmente más alta o más baja sin que
    cambie la forma del día.
    """
    es_baseline = True
    nombre = "perfil x ajuste reciente"

    def fit(self, train):
        self.perfil_ = Perfil().fit(train)
        return self

    def _esperado_en_origen(self, frame: pd.DataFrame) -> np.ndarray:
        """Lo que el perfil predice para la última hora del propio origen.

        Debe mirarse por estación: comparar el nivel de una estación contra el
        promedio de todas mezcla escalas y destruye el factor.
        """
        claves = frame[["station_id", "slot_origen", "finde_origen"]].rename(
            columns={"slot_origen": "slot", "finde_origen": "es_finde"})
        p = claves.join(self.perfil_.tabla_, on=["station_id", "slot", "es_finde"])["p"]
        return p.fillna(self.perfil_.global_).to_numpy(dtype=float)

    def predict(self, frame):
        base = self.perfil_.predict(frame)
        esperado = self._esperado_en_origen(frame)
        reciente = frame["roll1h"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(esperado > 0, reciente / esperado, 1.0)
        factor = np.clip(np.nan_to_num(factor, nan=1.0), 0.7, 1.3)
        return clip_no_negativo(base * factor)


class GBM(Modelo):
    """Gradient boosting sobre las variables de `features.COLS_NUM`."""
    nombre = "hist gradient boosting"

    def __init__(self, nombre=None, cols=None, **kw):
        if nombre:
            self.nombre = nombre
        self.cols = cols        # None = COLS_NUM completo
        self.kw = {"max_iter": 400, "learning_rate": 0.06,
                   "max_depth": None, "random_state": 20260916, **kw}

    def fit(self, train):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from features import COLS_NUM
        self.cols_ = list(self.cols) if self.cols else COLS_NUM
        self.est_ = HistGradientBoostingRegressor(**self.kw)
        self.est_.fit(train[self.cols_], train["y"])
        return self

    def predict(self, frame):
        return clip_no_negativo(self.est_.predict(frame[self.cols_]))


class GBMconPerfil(Modelo):
    """GBM que recibe el perfil como variable y entrena con pérdida absoluta.

    Dos correcciones sobre el GBM simple:
    - La métrica del reto es WAPE, que es error absoluto relativo. Entrenar con
      error cuadrático optimiza algo distinto y persigue los picos.
    - Darle el perfil como variable evita que tenga que reaprender la forma del
      día desde cero: parte del mejor baseline y solo corrige sobre él.
    """
    nombre = "gbm + perfil (mae)"

    # Variables en unidades de pasajeros: son las que se dividen por la escala
    # cuando el modelo trabaja normalizado.
    COLS_NIVEL = ("lag0", "lag1", "lag2", "lag3", "lag4", "lag_dia", "lag_sem",
                  "roll1h", "roll24h", "roll7d", "perfil")
    PISO_ESCALA = 1.0

    def __init__(self, nombre=None, por_estacion=False, cols=None, escalas=None,
                 semivida_h=None, perfil_dias=None, **kw):
        if nombre:
            self.nombre = nombre
        self.por_estacion = por_estacion
        # Adaptación a un cambio de FORMA (revisión 2 del drift, 30-sep). Una
        # ventana de 14 días aprende sobre todo la forma vieja; estas dos
        # perillas hacen que lo reciente pese más sin tirar la historia:
        #   semivida_h:  cada fila pesa 0,5^(edad / semivida) en el ajuste.
        #   perfil_dias: la variable `perfil` se calcula sólo con los últimos
        #                días (lo que falte se completa con la ventana entera).
        # Sólo actúan en `fit`: `predict` no cambia, así que el artefacto sigue
        # cargando con el `models.py` de main.
        self.semivida_h = semivida_h
        self.perfil_dias = perfil_dias
        self.cols = cols        # None = COLS_NUM completo
        # Normalización por nivel reciente. Con `escalas` (columnas de
        # `features`, p. ej. ("roll24h",)) el modelo no predice pasajeros sino
        # pasajeros / escala, y las variables de nivel entran divididas por la
        # misma escala. Si una estación duplica su demanda, la escala se duplica
        # y el modelo la acompaña sin reentrenar. Se entrena con peso = escala
        # para que la pérdida siga siendo error absoluto en pasajeros (WAPE).
        # Con varias escalas se promedian las predicciones: cada una reacciona
        # a otra velocidad. None = comportamiento de siempre.
        self.escalas = tuple(escalas) if escalas else None
        self.kw = {"max_iter": 500, "learning_rate": 0.06,
                   "loss": "absolute_error", "random_state": 20260916, **kw}

    def _con_perfil(self, frame: pd.DataFrame) -> pd.DataFrame:
        p = frame[self.claves_].join(self.tabla_, on=self.claves_)["p"]
        return frame.assign(perfil=p.fillna(self.global_).to_numpy())

    def _normalizado(self, t: pd.DataFrame, escala: str) -> tuple[pd.DataFrame, np.ndarray]:
        s = np.maximum(t[escala].to_numpy(dtype=float), self.PISO_ESCALA)
        X = t[self.cols_].copy()
        for c in self.COLS_NIVEL:
            if c in X:
                X[c] = X[c].to_numpy(dtype=float) / s
        return X, s

    def _pesos(self, train: pd.DataFrame) -> np.ndarray | None:
        semivida = getattr(self, "semivida_h", None)
        if not semivida:
            return None     # sin pesos: idéntico a como se entrenaba antes
        t = pd.DatetimeIndex(train["target_at"])
        edad_h = (t.max() - t).total_seconds().to_numpy() / 3600.0
        return 0.5 ** (edad_h / float(semivida))

    def _tabla_perfil(self, train: pd.DataFrame) -> pd.Series:
        tabla = train.groupby(self.claves_)["y"].mean().rename("p")
        dias = getattr(self, "perfil_dias", None)
        if not dias:
            return tabla
        t = pd.DatetimeIndex(train["target_at"])
        reciente = train[t > t.max() - pd.to_timedelta(dias, unit="D")]
        corta = reciente.groupby(self.claves_)["y"].mean().rename("p")
        return corta.combine_first(tabla)

    def fit(self, train):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from features import COLS_NUM
        self.claves_ = ["station_id", "slot", "es_finde"]
        self.tabla_ = self._tabla_perfil(train)
        self.global_ = float(train["y"].mean())
        w = self._pesos(train)
        # `cols_` es atributo de instancia, no de clase, y ahí está la gracia:
        # al deserializar se restaura tal cual, así que un artefacto entrenado
        # con menos columnas predice bien aunque el `main` que lo carga tenga
        # una definición de clase más vieja. Por eso la lista de features se
        # parametriza en vez de crear una clase nueva: una clase que main no
        # conoce rompe la inferencia; unos atributos distintos, no.
        base = list(self.cols) if getattr(self, "cols", None) else COLS_NUM
        self.cols_ = base + ["perfil"]

        t = self._con_perfil(train)
        if getattr(self, "escalas", None):
            self.est_ = {}
            for e in self.escalas:
                X, s = self._normalizado(t, e)
                self.est_[e] = HistGradientBoostingRegressor(**self.kw).fit(
                    X, t["y"].to_numpy(dtype=float) / s, sample_weight=s if w is None else s * w)
            return self
        if self.por_estacion:
            self.est_ = {e: HistGradientBoostingRegressor(**self.kw).fit(
                             g[self.cols_], g["y"],
                             sample_weight=None if w is None else w[t.index.get_indexer(g.index)])
                         for e, g in t.groupby("station_id")}
        else:
            self.est_ = HistGradientBoostingRegressor(**self.kw).fit(
                t[self.cols_], t["y"], sample_weight=w)
        return self

    def predict(self, frame):
        f = self._con_perfil(frame).reset_index(drop=True)
        if getattr(self, "escalas", None):
            salidas = []
            for e in self.escalas:
                X, s = self._normalizado(f, e)
                salidas.append(self.est_[e].predict(X) * s)
            return clip_no_negativo(np.mean(salidas, axis=0))
        if not self.por_estacion:
            return clip_no_negativo(self.est_.predict(f[self.cols_]))
        out = np.zeros(len(f))
        for e, g in f.groupby("station_id"):
            out[g.index.to_numpy()] = self.est_[e].predict(g[self.cols_])
        return clip_no_negativo(out)


class GBMporEstacion(Modelo):
    """Un modelo por estación: cada una tiene su propio nivel y forma."""
    nombre = "gbm por estacion"

    def fit(self, train):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from features import COLS_NUM
        self.cols_ = COLS_NUM
        self.est_ = {}
        for est, g in train.groupby("station_id"):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                              random_state=20260916)
            self.est_[est] = m.fit(g[self.cols_], g["y"])
        return self

    def predict(self, frame):
        frame = frame.reset_index(drop=True)
        out = np.zeros(len(frame))
        for est, g in frame.groupby("station_id"):
            out[g.index.to_numpy()] = self.est_[est].predict(g[self.cols_])
        return clip_no_negativo(out)


def candidatos() -> list[Modelo]:
    return [
        Columna("persistencia (ultimo dato)", "lag0"),
        Columna("naive d-1", "lag_dia"),
        Columna("naive s-1", "lag_sem"),
        Perfil(),
        Perfil("perfil (estacion x slot x dow)", ("station_id", "slot", "dow")),
        PerfilMasTendencia(),
        GBM(),
        GBM("gbm (mae)", loss="absolute_error"),
        GBMporEstacion(),
        GBMconPerfil(),
        GBMconPerfil("gbm + perfil por estacion (mae)", por_estacion=True),
    ]
