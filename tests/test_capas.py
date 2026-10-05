"""Capas del champion: detectan el período correcto y no miran el futuro."""
import numpy as np
import pandas as pd
import pytest

from pipeline.entregar import detectar_periodo, onda

from .conftest import ESTACIONES, serie_onda

CFG_ONDA = {"periodo_min": 20, "periodo_max": 44, "ventana": 16, "armonicos": 3, "span": 1.5}


def _objetivos(ancha, origen):
    paso = ancha.index[1] - ancha.index[0]
    return [(s, origen + h * paso) for s in ESTACIONES for h in (1, 2, 3, 4)]


def test_detectar_periodo_encuentra_la_onda_de_4_h():
    # Con algo de ruido, como la demanda real: con una onda perfecta 4 h y 8 h
    # empatan en WAPE ~0 y la tolerancia relativa del desempate no las separa.
    ancha = serie_onda(periodo_pasos=16, n=300, ruido=0.02)
    origen = ancha.index[250]
    p, wape = detectar_periodo(ancha, origen, {"ventana_h": 6, "periodo_min_h": 2,
                                                "periodo_max_h": 12})
    assert p == 16
    assert wape < 0.1


def test_onda_predice_bien_una_onda_de_8_h():
    ancha = serie_onda(periodo_pasos=32, n=300, ruido=0.02)
    origen = ancha.index[200]
    obj = _objetivos(ancha, origen)
    pred = np.array(onda(ancha, origen, obj, CFG_ONDA))
    real = np.array([ancha.at[t, s] for s, t in obj])
    assert len(pred) == 48
    assert (pred >= 0).all()
    wape = np.abs(pred - real).sum() / real.sum()
    assert wape < 0.08


def test_onda_no_usa_informacion_posterior_al_corte():
    ancha = serie_onda(periodo_pasos=32, n=300)
    origen = ancha.index[200]
    obj = _objetivos(ancha, origen)
    antes = onda(ancha, origen, obj, CFG_ONDA)
    alterada = ancha.copy()
    alterada.loc[alterada.index > origen] *= 50     # el futuro cambia por completo
    assert onda(alterada, origen, obj, CFG_ONDA) == pytest.approx(antes)


def test_detectar_periodo_no_usa_informacion_posterior_al_corte():
    ancha = serie_onda(periodo_pasos=16, n=300)
    origen = ancha.index[250]
    cfg = {"ventana_h": 6, "periodo_min_h": 2, "periodo_max_h": 12}
    alterada = ancha.copy()
    alterada.loc[alterada.index > origen] = 0
    assert detectar_periodo(alterada, origen, cfg) == detectar_periodo(ancha, origen, cfg)


def test_sin_historia_suficiente_no_inventa_periodo():
    ancha = serie_onda(periodo_pasos=16, n=40)
    p, wape = detectar_periodo(ancha, ancha.index[-1], {"ventana_h": 6, "periodo_min_h": 2,
                                                         "periodo_max_h": 12})
    assert p is None and wape == float("inf")
