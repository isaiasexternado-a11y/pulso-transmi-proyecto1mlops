"""Contrato del batch de 48 valores e idempotencia de la entrega."""
import pytest

from pipeline.entregar import llave_estable, validar, version_en_contrato, ya_entregado

from .conftest import ESTACIONES, SupabaseFalso

ORIGEN = "2026-09-21T08:00:00Z"
TARGETS = [f"2026-09-21T08:{m:02d}:00Z" if m < 60 else "2026-09-21T09:00:00Z"
           for m in (15, 30, 45, 60)]


def _ciclo():
    targets = [{"station_id": s, "target_at": t, "horizon_minutes": 15 * (i + 1)}
               for s in ESTACIONES for i, t in enumerate(TARGETS)]
    return {"cycle_id": "cyc_x", "targets": targets, "expected_predictions": 48}


def _preds(ciclo, valor=100.0):
    return [{"station_id": t["station_id"], "target_at": t["target_at"], "value": valor}
            for t in ciclo["targets"]]


def test_batch_completo_pasa():
    c = _ciclo()
    validar(_preds(c), c, "20261004T011104Z-onda")


def test_son_48_valores_12_estaciones_por_4_horizontes():
    c = _ciclo()
    assert len(c["targets"]) == 48
    assert {t["horizon_minutes"] for t in c["targets"]} == {15, 30, 45, 60}


@pytest.mark.parametrize("version,ok", [
    ("20261004T011104Z-onda", True),
    ("gbm:v1/fase.final_2", True),
    ("gbm+persistencia", False),          # el "+" costó un 422 el 2026-09-25
    ("-empieza-con-guion", False),
    ("x" * 65, False),
])
def test_version_en_contrato(version, ok):
    assert version_en_contrato(version) is ok


def test_falta_un_target_no_se_envia():
    c = _ciclo()
    with pytest.raises(SystemExit, match="no calzan"):
        validar(_preds(c)[:-1], c, "v1")


def test_targets_duplicados_no_se_envian():
    c = _ciclo()
    p = _preds(c)
    p[-1] = dict(p[0])
    with pytest.raises(SystemExit, match="duplicados"):
        validar(p, c, "v1")


@pytest.mark.parametrize("malo,mensaje", [(-1.0, "negativo"), (float("nan"), "no finito"),
                                          (float("inf"), "no finito")])
def test_valores_invalidos_no_se_envian(malo, mensaje):
    c = _ciclo()
    p = _preds(c)
    p[5]["value"] = malo
    with pytest.raises(SystemExit, match=mensaje):
        validar(p, c, "v1")


def test_horizonte_fuera_de_contrato_revienta_antes_del_post():
    c = _ciclo()
    c["targets"][0]["horizon_minutes"] = 75
    with pytest.raises(SystemExit, match="horizonte"):
        validar(_preds(c), c, "v1")


def test_llave_de_idempotencia_estable_ante_reintento():
    c = _ciclo()
    assert llave_estable("cyc_x", "v1", _preds(c)) == llave_estable("cyc_x", "v1", _preds(c))


def test_llave_cambia_si_cambia_el_contenido_o_el_ciclo():
    c = _ciclo()
    base = llave_estable("cyc_x", "v1", _preds(c))
    assert llave_estable("cyc_x", "v1", _preds(c, 101.0)) != base
    assert llave_estable("cyc_y", "v1", _preds(c)) != base
    assert llave_estable("cyc_x", "v2", _preds(c)) != base


def test_ya_entregado_solo_cuenta_recibos_aceptados():
    # Un 422 deja recibo pero no consume intento: no puede bloquear el reintento.
    sb = SupabaseFalso(submissions=[
        {"submission_id": "s1", "cycle_id": "cyc_x", "model_version": "v1", "http_status": 422},
    ])
    assert ya_entregado(sb, "cyc_x", "v1") is None
    sb.tablas["submissions"].append(
        {"submission_id": "s2", "cycle_id": "cyc_x", "model_version": "v1", "http_status": 201})
    assert ya_entregado(sb, "cyc_x", "v1")["submission_id"] == "s2"


def test_ya_entregado_distingue_ciclo_y_version():
    sb = SupabaseFalso(submissions=[
        {"submission_id": "s1", "cycle_id": "cyc_x", "model_version": "v1", "http_status": 201},
    ])
    assert ya_entregado(sb, "cyc_y", "v1") is None
    assert ya_entregado(sb, "cyc_x", "v2") is None
