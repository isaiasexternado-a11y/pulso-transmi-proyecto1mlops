"""Piezas compartidas por las pruebas.

Las pruebas no tocan la red: ni la API del profesor ni Supabase. `SupabaseFalso`
imita la parte de `collector.entorno.Supabase` que usa el pipeline, con los
filtros de PostgREST que aparecen en el código (eq, in, is.null, not.is.null,
order, limit), sobre tablas en memoria.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ESTACIONES = ["02300", "03000", "05000", "05100", "06000", "06111",
              "07107", "07111", "09122", "09104", "10000", "11000"]


def _valor(texto: str):
    if texto in ("true", "false"):
        return texto == "true"
    try:
        return int(texto)
    except ValueError:
        return texto


def _cumple(fila: dict, columna: str, filtro: str) -> bool:
    v = fila.get(columna)
    if filtro == "is.null":
        return v is None
    if filtro == "not.is.null":
        return v is not None
    op, _, arg = filtro.partition(".")
    if op == "eq":
        return v == _valor(arg) or str(v) == arg
    if op == "in":
        return v in [_valor(x) for x in arg.strip("()").split(",")] or str(v) in arg.strip("()").split(",")
    raise NotImplementedError(filtro)


class SupabaseFalso:
    def __init__(self, **tablas: list[dict]) -> None:
        self.tablas = {k: [dict(f) for f in v] for k, v in tablas.items()}

    def seleccionar(self, tabla: str, select: str = "*", order: str | None = None,
                    limit: int | None = None, **filtros) -> list[dict]:
        filas = [f for f in self.tablas.get(tabla, [])
                 if all(_cumple(f, c, x) for c, x in filtros.items())]
        if order:
            col, _, sentido = order.partition(".")
            filas.sort(key=lambda f: (f.get(col) is None, f.get(col)),
                       reverse=sentido == "desc")
        filas = filas[:limit] if limit else filas
        if select != "*":
            cols = select.split(",")
            filas = [{c: f.get(c) for c in cols} for f in filas]
        return [dict(f) for f in filas]

    def actualizar(self, tabla: str, filtro: dict, cambios: dict) -> None:
        for f in self.tablas.get(tabla, []):
            if all(_cumple(f, c, x) for c, x in filtro.items()):
                f.update(cambios)

    def insertar(self, tabla: str, filas: list[dict], devolver: bool = False) -> list:
        self.tablas.setdefault(tabla, []).extend(dict(f) for f in filas)
        return filas if devolver else []


@pytest.fixture
def sb_falso():
    return SupabaseFalso


def serie_onda(periodo_pasos: int, n: int = 400, inicio: str = "2026-09-20T12:00Z",
               ruido: float = 0.0, semilla: int = 0) -> pd.DataFrame:
    """Demanda sintética de 12 estaciones que repite una onda de `periodo_pasos`
    pasos de 15 min, con nivel y desfase propios por estación."""
    rng = np.random.default_rng(semilla)
    idx = pd.date_range(inicio, periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    cols = {}
    for j, s in enumerate(ESTACIONES):
        nivel = 200 + 60 * j
        onda = 1 + 0.5 * np.sin(2 * np.pi * (t + 3 * j) / periodo_pasos) \
                 + 0.15 * np.cos(4 * np.pi * (t + 3 * j) / periodo_pasos)
        cols[s] = nivel * onda * (1 + ruido * rng.standard_normal(n))
    return pd.DataFrame(cols, index=idx).clip(lower=0)
