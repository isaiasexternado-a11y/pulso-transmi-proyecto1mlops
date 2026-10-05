"""La métrica oficial: WAPE por estación y promedio sin ponderar."""
import numpy as np
import pandas as pd
import pytest

from ml.metrics import accuracy_oficial, accuracy_por_estacion, clip_no_negativo


def _df(filas):
    return pd.DataFrame(filas, columns=["station_id", "horizon", "y", "y_pred"])


def test_prediccion_perfecta_da_100():
    df = _df([("A", 1, 10, 10), ("A", 2, 20, 20), ("B", 1, 5, 5)])
    assert accuracy_oficial(df) == pytest.approx(100.0)


def test_wape_se_calcula_por_estacion():
    # A: |12-10| + |18-20| = 4 sobre 30 -> WAPE 0,1333 -> 86,67
    df = _df([("A", 1, 10, 12), ("A", 2, 20, 18)])
    assert accuracy_por_estacion(df)["A"] == pytest.approx(100 * (1 - 4 / 30))


def test_promedio_no_ponderado_estacion_grande_no_tapa_a_la_pequena():
    # A es enorme y perfecta; B es pequeña y con 50 % de error.
    df = _df([("A", 1, 100_000, 100_000), ("B", 1, 10, 5)])
    assert accuracy_oficial(df) == pytest.approx((100 + 50) / 2)


def test_accuracy_no_baja_de_cero():
    df = _df([("A", 1, 10, 100)])
    assert accuracy_por_estacion(df)["A"] == 0


def test_target_ausente_cuenta_como_prediccion_cero():
    # La API evalúa un target no entregado como predicción 0: WAPE 1 -> accuracy 0.
    df = _df([("A", 1, 10, 0), ("A", 2, 10, 0)])
    assert accuracy_por_estacion(df)["A"] == 0


def test_clip_deja_valores_finitos_y_no_negativos():
    out = clip_no_negativo(np.array([-3.0, np.nan, np.inf, -np.inf, 7.5]))
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0, 7.5]
