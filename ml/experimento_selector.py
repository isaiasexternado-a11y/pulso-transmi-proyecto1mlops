"""¿Qué promedio periódico conviene cuando la capa estacional está prendida?

La revisión 3 del drift dejó una demanda que se repite cada 4 h. La capa
estacional de `pipeline/entregar.py` (en `main`) la detecta sola y, en su
primera versión, predecía con el promedio de lo observado P y 2P antes
(`n = 2`). El 2026-10-02, de la retroalimentación compartida en el curso,
salieron dos ideas que adaptamos y medimos aquí:

    estK      promedio de lo observado P, 2P, …, KP antes del objetivo (K=1..6).
    plantK    las 12 estaciones repiten la misma onda, desfasada y escalada
              por su nivel: se estima una forma común con las últimas K
              oscilaciones de todas alineadas y cada una la usa con su fase y
              su nivel (12 veces más datos que su propia historia).
    selector  en cada ciclo gana el experto de mejor accuracy en los
              SELECTOR_CICLOS orígenes horarios previos.

Aquí se miden sobre los ciclos reales: en cada origen horario se replica la
puerta de la capa (período detectado con la ventana de 6 h, peso por WAPE) y
se compara `n = 2` con el selector. Si hay credenciales, se agrega lo que el
equipo entregó de verdad en cada ciclo (submissions oficiales en Supabase).

Validación temporal: cada experto y cada puntaje sólo leen observaciones
hasta el origen; los ciclos con que se puntúa tienen sus objetivos resueltos
a esa hora.

Las columnas `n2` y `selector` son los expertos puros, sin mezclar con el
peso de la capa (en producción, el ciclo 18 14:00 llevaba peso 0,77).

Resultado 2026-10-02 (ciclos 18 14:00 - 19 20:00 virtual, capa prendida en 31):
n=2 91,53 -> selector 93,01; últimos 12: 91,31 -> 93,03 (= lo entregado con n=2).
Con el peso de producción, en `main`: 91,03 -> 92,47.
En régimen normal ningún experto pasa de 68 % y la puerta no prende la capa.

Uso:
    python3 -m ml.experimento_selector
    python3 -m ml.experimento_selector --desde 2026-09-18T00:00Z --hasta 2026-09-19T20:00Z
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from ml.data import cargar  # noqa: E402

# La configuración de `hyperparams.estacional` del champion el 2026-10-02.
CFG = {"ventana_h": 6, "periodo_min_h": 2, "periodo_max_h": 12,
       "umbral_pleno": 0.15, "umbral_apagado": 0.25}
EMPATE = 0.25
SELECTOR_K = range(1, 7)
SELECTOR_CICLOS = 3
H = (1, 2, 3, 4)          # +15, +30, +45, +60


def accuracy(real: np.ndarray, pred: np.ndarray) -> float:
    """Promedio no ponderado por estación de 100 · max(0, 1 − WAPE)."""
    wape = np.sum(np.abs(real - pred), axis=0) / np.sum(real, axis=0)
    return float(np.mean(100 * np.maximum(0, 1 - wape)))


def detectar_periodo(Y: np.ndarray, o: int, paso_min: int) -> tuple[int | None, float]:
    """Réplica de `pipeline.entregar.detectar_periodo` sobre Y[:o + 1]."""
    por_h = 60 // paso_min
    ventana = CFG["ventana_h"] * por_h
    pmin, pmax = CFG["periodo_min_h"] * por_h, CFG["periodo_max_h"] * por_h
    v = Y[:o + 1]
    if len(v) < ventana + pmax:
        return None, float("inf")
    reciente = v[-ventana:]
    total = np.nansum(reciente)
    wapes = {p: float(np.nansum(np.abs(reciente - v[-ventana - p:-p])) / total)
             for p in range(pmin, pmax + 1)}
    minimo = min(wapes.values())
    mejor = min(p for p, w in wapes.items() if w <= minimo * (1 + EMPATE))
    return mejor, wapes[mejor]


def peso_capa(wape: float) -> float:
    lo, hi = CFG["umbral_pleno"], CFG["umbral_apagado"]
    return 1.0 if wape <= lo else 0.0 if wape >= hi else (hi - wape) / (hi - lo)


def expertos(Y: np.ndarray, o: int, p: int) -> dict[str, np.ndarray]:
    """{nombre: matriz horizonte × estación} desde el origen `o`."""
    out = {}
    for k in SELECTOR_K:
        if o + 1 - p * k < 0:
            break
        out[f"est{k}"] = np.stack([np.mean([Y[o + h - p * j] for j in range(1, k + 1)], axis=0) for h in H])
        prof = np.nanmean(Y[o - p * k + 1:o + 1].reshape(k, p, -1), axis=0)
        nivel = prof.mean(axis=0)
        if not np.all(nivel > 0):
            continue
        forma = prof / nivel
        ref = forma[:, int(np.argmax(nivel))]
        desfases = [min(range(p), key=lambda s: np.nansum(np.abs(np.roll(forma[:, j], -s) - ref)))
                    for j in range(forma.shape[1])]
        comun = np.nanmean([np.roll(forma[:, j], -s) for j, s in enumerate(desfases)], axis=0)
        out[f"plant{k}"] = np.stack([np.array([comun[(h - 1 - s) % p] for s in desfases]) * nivel for h in H])
    return out


def seleccionar(Y: np.ndarray, o: int, p: int) -> tuple[str, float] | None:
    puntajes: dict[str, list[float]] = {}
    for c in range(1, SELECTOR_CICLOS + 1):
        oc = o - H[-1] * c
        real = Y[oc + 1:oc + 1 + H[-1]]
        for nombre, pred in expertos(Y, oc, p).items():
            puntajes.setdefault(nombre, []).append(accuracy(real, pred))
    validos = {e: float(np.mean(v)) for e, v in puntajes.items()
               if len(v) == SELECTOR_CICLOS and np.isfinite(v).all()}
    if not validos:
        return None
    mejor = max(validos, key=validos.get)
    return mejor, validos[mejor]


def entregado(columnas: list[str]) -> pd.DataFrame | None:
    """Predicciones oficiales por objetivo, si hay credenciales de Supabase."""
    try:
        from collector.entorno import Supabase, cargar_env
        cargar_env()
        sb = Supabase()
        subs = pd.DataFrame(sb.seleccionar("submissions", select="submission_id,cycle_id,http_status",
                                           order="submitted_at"))
        subs = subs[subs.http_status.between(200, 299)].groupby("cycle_id").last()
        filas = []
        for sid in subs.submission_id:
            filas += sb.seleccionar("predictions", select="station_id,target_at,y_pred",
                                    submission_id=f"eq.{sid}")
    except (SystemExit, Exception) as e:  # noqa: BLE001 - sin credenciales se mide sin lo entregado
        print(f"[selector] sin lo entregado: {e}", file=sys.stderr)
        return None
    df = pd.DataFrame(filas)
    df["target_at"] = pd.to_datetime(df.target_at, utc=True)
    return df.pivot_table(index="target_at", columns="station_id", values="y_pred").reindex(columns=columnas)


def main(desde: str, hasta: str) -> None:
    ancha, _, _ = cargar()
    cols = list(ancha.columns)
    Y = ancha.to_numpy(float)
    paso_min = int((ancha.index[1] - ancha.index[0]) / pd.Timedelta(minutes=1))
    ent = entregado(cols)

    filas = []
    for origen in pd.date_range(desde, hasta, freq="1h"):
        origen = origen.tz_convert(ancha.index.tz)
        if origen not in ancha.index:
            continue
        o = ancha.index.get_loc(origen)
        if o + H[-1] >= len(Y) or np.isnan(Y[o + 1:o + 1 + H[-1]]).any():
            continue
        real = Y[o + 1:o + 1 + H[-1]]
        p, wape = detectar_periodo(Y, o, paso_min)
        w = peso_capa(wape) if p else 0.0
        fila = {"origen": origen.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"), "periodo": p,
                "wape_periodo": round(wape, 3), "peso": round(w, 2)}
        if ent is not None:
            objetivos = ancha.index[o + 1:o + 1 + H[-1]].tz_convert("UTC")
            if set(objetivos) <= set(ent.index):
                fila["entregado"] = accuracy(real, ent.loc[objetivos].to_numpy(float))
        if w > 0:
            ex = expertos(Y, o, p)
            fila["n2"] = accuracy(real, ex["est2"])
            sel = seleccionar(Y, o, p)
            if sel:
                fila["experto"], fila["puntaje"] = sel[0], round(sel[1], 2)
                fila["selector"] = accuracy(real, ex[sel[0]])
        filas.append(fila)

    df = pd.DataFrame(filas).set_index("origen")
    pd.set_option("display.width", 200)
    print(df.round(2).to_string())
    prendida = df[df.peso > 0] if "selector" in df else df.iloc[0:0]
    resumen = {}
    if len(prendida):
        cols_m = [c for c in ("entregado", "n2", "selector") if c in prendida]
        resumen = {"ciclos_prendida": len(prendida),
                   "prendida": prendida[cols_m].mean().round(2).to_dict(),
                   "ultimos_12": prendida.tail(12)[cols_m].mean().round(2).to_dict()}
        print(f"\ncapa prendida en {len(prendida)} ciclos: {resumen['prendida']}")
        print(f"últimos 12: {resumen['ultimos_12']}")

    salida = RAIZ / "ml/resultados/experimento_selector.json"
    salida.write_text(json.dumps({
        "calculado": datetime.now(timezone.utc).isoformat(),
        "desde": desde, "hasta": hasta, "cfg": CFG,
        "selector_k": list(SELECTOR_K), "selector_ciclos": SELECTOR_CICLOS,
        "resumen": resumen,
        "ciclos": json.loads(df.reset_index().to_json(orient="records")),
    }, indent=2, ensure_ascii=False))
    print(f"\ndetalle -> {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desde", default="2026-09-17T00:00Z")
    ap.add_argument("--hasta", default="2026-09-19T20:00Z")
    a = ap.parse_args()
    main(a.desde, a.hasta)
