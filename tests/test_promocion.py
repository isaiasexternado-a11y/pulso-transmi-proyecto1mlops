"""Promoción y rollback: un solo champion, el saliente queda disponible."""
import pytest

from pipeline import promover as prom

from .conftest import SupabaseFalso


def _modelos():
    return SupabaseFalso(models=[
        {"model_id": "m1", "name": "viejo", "status": "retired",
         "retired_at": "2026-10-01T00:00:00Z", "hyperparams": {"version": "v1"}},
        {"model_id": "m2", "name": "champion", "status": "active",
         "retired_at": None, "hyperparams": {"version": "v2"}},
        {"model_id": "m3", "name": "candidato", "status": "candidate",
         "retired_at": None, "hyperparams": {"version": "v3"}},
    ], retrain_decisions=[
        {"run_id": 10, "decision": "retrain", "new_model_id": None,
         "decided_at": "2026-10-04T00:00:00Z"},
    ])


def _estado(sb):
    return {m["model_id"]: m["status"] for m in sb.tablas["models"]}


def test_promover_deja_un_solo_active_y_retira_al_saliente():
    sb = _modelos()
    prom.promover(sb, {"model_id": "m3", "name": "candidato"})
    assert _estado(sb) == {"m1": "retired", "m2": "retired", "m3": "active"}
    assert sb.tablas["models"][1]["retired_at"] is not None


def test_promover_cierra_la_decision_de_reentrenar_con_el_modelo_nuevo():
    sb = _modelos()
    prom.promover(sb, {"model_id": "m3", "name": "candidato"})
    assert sb.tablas["retrain_decisions"][0]["new_model_id"] == "m3"


def test_rollback_vuelve_al_ultimo_retirado_si_pasa_la_inferencia(monkeypatch):
    sb = _modelos()
    monkeypatch.setattr(prom, "inferencia_de_prueba", lambda sb, ficha: (True, "ok"))
    prom.rollback(sb, dry_run=False)
    assert _estado(sb)["m1"] == "active"
    assert _estado(sb)["m2"] == "retired"
    # Un rollback no responde a ninguna decisión de reentrenar.
    assert sb.tablas["retrain_decisions"][0]["new_model_id"] is None


def test_rollback_no_revierte_si_el_anterior_falla_la_inferencia(monkeypatch):
    sb = _modelos()
    monkeypatch.setattr(prom, "inferencia_de_prueba", lambda sb, ficha: (False, "falla"))
    with pytest.raises(SystemExit):
        prom.rollback(sb, dry_run=False)
    assert _estado(sb)["m2"] == "active"


def test_rollback_en_seco_no_cambia_nada(monkeypatch):
    sb = _modelos()
    monkeypatch.setattr(prom, "inferencia_de_prueba", lambda sb, ficha: (True, "ok"))
    prom.rollback(sb, dry_run=True)
    assert _estado(sb)["m2"] == "active"
