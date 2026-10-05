"""Collector: formatos del stream (v1 y v2) y cursor."""
import base64
import json

import pytest

from collector.recolectar import cursor_de, demanda, en_lotes


def test_demanda_v1_plana():
    assert demanda({"station_id": "02300", "demand": 546}) == 546


def test_demanda_v2_observada():
    o = {"schema_version": 2,
         "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"}}
    assert demanda(o) == 546


@pytest.mark.parametrize("m", [
    {"value": None, "unit": "passengers", "quality": "missing"},
    {"value": None, "unit": "passengers", "quality": "observed"},
])
def test_demanda_v2_faltante_no_es_cero(m):
    # Un faltante no se guarda como 0: entrenar con ceros falsos sesga el modelo.
    assert demanda({"schema_version": 2, "measurement": m}) is None


@pytest.mark.parametrize("o", [
    {"schema_version": 3, "measurement": {"value": "1", "unit": "passengers", "quality": "observed"}},
    {"schema_version": 2, "measurement": {"value": "1", "unit": "kg", "quality": "observed"}},
    {"schema_version": 2, "measurement": {"value": "1", "unit": "passengers", "quality": "estimated"}},
])
def test_formato_desconocido_revienta_antes_de_mover_el_cursor(o):
    with pytest.raises(ValueError):
        demanda(o)


def test_cursor_es_base64_del_trio_released_observed_station():
    fila = {"released_at": "2026-10-04T10:00:00Z", "observed_at": "2026-09-21T08:00:00Z",
            "station_id": "05100"}
    crudo = cursor_de(fila)
    relleno = "=" * (-len(crudo) % 4)
    assert json.loads(base64.urlsafe_b64decode(crudo + relleno)) == [
        "2026-10-04T10:00:00+00:00", "2026-09-21T08:00:00+00:00", "05100"]


def test_cursor_es_determinista():
    fila = {"released_at": "2026-10-04T10:00:00Z", "observed_at": "2026-09-21T08:00:00Z",
            "station_id": "05100"}
    assert cursor_de(fila) == cursor_de(dict(fila))


def test_en_lotes_no_pierde_ni_repite_filas():
    filas = list(range(1234))
    lotes = list(en_lotes(filas, 500))
    assert [len(l) for l in lotes] == [500, 500, 234]
    assert sum(lotes, []) == filas
