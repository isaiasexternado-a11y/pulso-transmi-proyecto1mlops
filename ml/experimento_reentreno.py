"""¿Reentrenar seguido le gana al GBM congelado bajo el drift?

El GBM del champion se entrenó con datos hasta el 9 virtual: nunca vio una
hora de drift. La corrección de nivel le arregla el nivel, pero no la forma del
día, y en la continuación del escenario (16 virtual, 08:00) la forma también
cambió: 02300 y 05000 corren a 2-4 veces su semana anterior con el pico más
largo. El 2026-09-29 los dos primeros del leaderboard de 24 h reentrenaban cada
pocas horas; el nuestro no había registrado un candidato desde el 22.

Las compuertas de `entrenar.py` validan en folds de 7 días, donde la
continuación es una fracción pequeña. Aquí se simula lo que haría producción
con reentreno periódico:

    cada CADA_H horas (el "reentreno"), cada receta se entrena con todos los
    objetivos resueltos hasta ese instante y predice los ciclos horarios de
    las CADA_H horas siguientes. Se le aplican las capas del champion vigente
    (mezcla con persistencia y corrección de nivel con quiebre), igual que en
    `entrenar.aplicar_capas`. La ventana de nivel se calienta con los ciclos
    de las horas previas al bloque, predichos por el MISMO modelo: en
    producción los backcasts también los hace el modelo recién entrenado.

La vara es el GBM congelado del champion con las mismas capas.

Validación temporal: ningún objetivo de entrenamiento es posterior al primer
origen del bloque que se evalúa.

Uso:
    python3 -m ml.experimento_reentreno
    python3 -m ml.experimento_reentreno --desde 2026-09-13T05:00:00Z --cada 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase                          # noqa: E402
from data import cargar                                          # noqa: E402
from entrenar import aplicar_capas, capas_vigentes               # noqa: E402
from features import COLS_SIN_CONTEXTO, construir, origenes_por_hora  # noqa: E402
from metrics import accuracy_oficial, accuracy_por_estacion      # noqa: E402
from models import GBMconPerfil, Perfil                          # noqa: E402
from pipeline.entregar import champion                           # noqa: E402

INICIO_DRIFT = pd.Timestamp("2026-09-13T05:00:00Z")
CONTINUACION = pd.Timestamp("2026-09-16T13:00:00Z")
CALENTAR_H = 5           # 2 · ventana de nivel (2 h) + holgura
SC = COLS_SIN_CONTEXTO
ESCALAS3 = ("roll24h", "roll1h", "roll7d")


def recetas() -> dict:
    """Receta -> (constructor, días de ventana o None)."""
    return {
        "gbm + perfil (todo)": (lambda: GBMconPerfil(), None),
        "gbm + perfil sin contexto": (lambda: GBMconPerfil(cols=SC), None),
        "gbm + perfil sin contexto · 14d": (lambda: GBMconPerfil(cols=SC), 14),
        "normalizado 24h": (lambda: GBMconPerfil(cols=SC, escalas=("roll24h",)), None),
        "normalizado 3 escalas": (lambda: GBMconPerfil(cols=SC, escalas=ESCALAS3), None),
        "normalizado 3 escalas · 14d": (lambda: GBMconPerfil(cols=SC, escalas=ESCALAS3), 14),
        "perfil": (lambda: Perfil(), None),
        "perfil · 14d": (lambda: Perfil(), 14),
    }


def main(desde: pd.Timestamp, cada_h: int) -> None:
    sb = Supabase()
    ancha, ctx, _ = cargar(origen="supabase")
    congelado, ficha = champion(sb)
    capas = capas_vigentes(sb)
    print(f"champion: {ficha['hyperparams']['version']}  capas: {', '.join(capas)}")

    tz = ancha.index.tz
    fin = ancha.index[-1]
    cortes = pd.date_range(desde.tz_convert(tz), fin, freq=f"{cada_h}h")
    o_todo = origenes_por_hora(ancha, ancha.index[0], fin)
    todo = construir(ancha, ctx, o_todo)
    todo["origen"] = todo["target_at"] - todo["horizon"] * pd.to_timedelta(15, unit="min")

    salidas = []
    for i, corte in enumerate(cortes):
        hasta = min(corte + pd.to_timedelta(cada_h, unit="h"), fin)
        bloque = todo[(todo["origen"] >= corte - pd.to_timedelta(CALENTAR_H, unit="h"))
                      & (todo["origen"] < hasta)].dropna(subset=["y"])
        if bloque.empty:
            continue
        evaluar = (bloque["origen"] >= corte).to_numpy()
        train_full = todo[todo["target_at"] <= corte].dropna()
        preds = {"congelado": congelado.predict(bloque)}
        for nombre, (crear, dias) in recetas().items():
            tr = train_full
            if dias:
                tr = tr[tr["target_at"] > corte - pd.to_timedelta(dias, unit="D")]
            t0 = time.time()
            preds[nombre] = crear().fit(tr).predict(bloque)
            print(f"  {corte:%m-%d %H:%M} {nombre:32} {time.time() - t0:5.1f}s", file=sys.stderr)
        b = bloque.copy()
        for nombre, p in preds.items():
            b[nombre] = aplicar_capas(bloque, p, capas)
        salidas.append(b[evaluar])
        print(f"bloque {i + 1}/{len(cortes)}  {corte:%m-%d %H:%M} -> {hasta:%m-%d %H:%M}  "
              + "  ".join(f"{k[:14]}={accuracy_oficial(b[evaluar], k):.2f}"
                          for k in ("congelado", "gbm + perfil sin contexto",
                                    "normalizado 3 escalas", "perfil")))

    res = pd.concat(salidas)
    nombres = ["congelado", *recetas()]
    o_utc = res["origen"].dt.tz_convert("UTC")
    tramos = {"drift antes de la continuación": o_utc < CONTINUACION,
              "continuación": o_utc >= CONTINUACION,
              "todo": o_utc >= pd.Timestamp(0, tz="UTC")}
    tabla = pd.DataFrame({t: {n: accuracy_oficial(res[m], n) for n in nombres}
                          for t, m in tramos.items() if m.any()}).round(2)
    tabla["ciclos cont."] = int(res.loc[tramos["continuación"], "origen"].nunique())
    print("\n" + tabla.sort_values("continuación", ascending=False).to_string())

    cont = res[tramos["continuación"]]
    mejor = tabla["continuación"].drop("congelado").idxmax()
    est = pd.DataFrame({"congelado": accuracy_por_estacion(cont, "congelado"),
                        mejor: accuracy_por_estacion(cont, mejor)}).round(1)
    est["delta"] = est[mejor] - est["congelado"]
    print(f"\ncontinuación por estación:\n{est.to_string()}")

    # Por ciclo: ¿gana de forma pareja o por un par de ciclos?
    por_ciclo = cont.groupby("origen").apply(
        lambda g: pd.Series({n: accuracy_oficial(g, n) for n in ("congelado", mejor)}),
        include_groups=False)
    gana = int((por_ciclo[mejor] > por_ciclo["congelado"]).sum())
    print(f"\n{mejor} gana en {gana} de {len(por_ciclo)} ciclos de la continuación")

    salida = RAIZ / "ml/resultados/experimento_reentreno.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "calculado": datetime.now(timezone.utc).isoformat(),
        "champion": ficha["hyperparams"]["version"], "capas": list(capas),
        "desde": str(desde), "cada_h": cada_h, "fin": str(fin),
        "tabla": tabla.to_dict(orient="index"),
        "continuacion_por_estacion": est.to_dict(orient="index"),
        "mejor": mejor, "gana_ciclos": [gana, len(por_ciclo)],
    }, indent=2, ensure_ascii=False, default=float))
    print(f"\ndetalle -> {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default=str(INICIO_DRIFT))
    ap.add_argument("--cada", type=int, default=6)
    a = ap.parse_args()
    main(pd.Timestamp(a.desde), a.cada)
