"""Regla de reentrenamiento: persistencia, volumen de datos nuevos y enfriamiento."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from pipeline.evaluar import CICLOS_PERSISTENCIA, MIN_OBS_NUEVAS, decidir

from .conftest import SupabaseFalso

FICHA = {"model_id": "champ", "train_end": "2026-09-20T00:00:00Z"}
AHORA = "2026-10-04T12:00:00Z"


def _obs(n):
    return pd.DataFrame({"observed_at": pd.date_range(
        "2026-09-20T00:15Z", periods=n, freq="15min", tz="UTC")})


def _senal(rota=True, signal="wape_24h", station=None):
    return {"signal": signal, "station_id": station, "breached": rota}


def _sb(rojas_previas=0, cooldown_hasta=None):
    """Historial del champion: `rojas_previas` corridas con wape_24h en rojo."""
    decisiones, senales = [], []
    for i in range(rojas_previas):
        decisiones.append({"run_id": 100 + i, "incumbent_model_id": "champ",
                           "decision": "blocked", "cooldown_until": None,
                           "decided_at": f"2026-10-04T0{i}:00:00Z"})
        senales.append({"run_id": 100 + i, "signal": "wape_24h", "station_id": None,
                        "breached": True, "computed_at": f"2026-10-04T0{i}:00:00Z"})
    if cooldown_hasta:
        decisiones.append({"run_id": 99, "incumbent_model_id": "otro", "decision": "retrain",
                           "cooldown_until": cooldown_hasta, "decided_at": "2026-10-03T00:00:00Z"})
    return SupabaseFalso(retrain_decisions=decisiones, drift_signals=senales)


def test_sin_senales_rotas_se_mantiene():
    d = decidir(_sb(), 1, [_senal(False)], FICHA, _obs(1000), AHORA)
    assert d["decision"] == "keep"


def test_alerta_temprana_sin_caida_de_desempeno_no_reentrena():
    d = decidir(_sb(), 1, [_senal(False), _senal(True, "profile_corr")], FICHA, _obs(1000), AHORA)
    assert d["decision"] == "keep"
    assert d["breached_signals"] == ["profile_corr"]


def test_un_solo_periodo_malo_no_dispara():
    d = decidir(_sb(rojas_previas=0), 1, [_senal()], FICHA, _obs(1000), AHORA)
    assert d["decision"] == "blocked"
    assert "Un mal periodo no es drift" in d["reason"]


def test_persistencia_con_datos_nuevos_dispara_reentreno():
    sb = _sb(rojas_previas=CICLOS_PERSISTENCIA - 1)
    d = decidir(sb, 1, [_senal()], FICHA, _obs(MIN_OBS_NUEVAS + 10), AHORA)
    assert d["decision"] == "retrain"
    assert d["cooldown_until"] is not None


def test_sin_suficientes_datos_nuevos_se_bloquea():
    sb = _sb(rojas_previas=CICLOS_PERSISTENCIA - 1)
    d = decidir(sb, 1, [_senal()], FICHA, _obs(MIN_OBS_NUEVAS - 10), AHORA)
    assert d["decision"] == "blocked"
    assert "observaciones nuevas" in d["reason"]


def test_enfriamiento_vigente_bloquea():
    futuro = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    sb = _sb(rojas_previas=CICLOS_PERSISTENCIA - 1, cooldown_hasta=futuro)
    d = decidir(sb, 1, [_senal()], FICHA, _obs(MIN_OBS_NUEVAS + 10), AHORA)
    assert d["decision"] == "blocked"
    assert "enfriamiento" in d["reason"]


def test_la_racha_no_hereda_corridas_de_otro_champion():
    # Corridas en rojo de un modelo anterior no cuentan para el champion vigente.
    sb = _sb(rojas_previas=CICLOS_PERSISTENCIA - 1)
    for d in sb.tablas["retrain_decisions"]:
        d["incumbent_model_id"] = "anterior"
    d = decidir(sb, 1, [_senal()], FICHA, _obs(MIN_OBS_NUEVAS + 10), AHORA)
    assert d["decision"] == "blocked"
