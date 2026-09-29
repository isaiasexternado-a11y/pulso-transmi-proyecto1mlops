"""¿Cómo le va en vivo al champion frente a lo que habría entregado su padre?

Toma las entregas reales del champion vigente cuyos objetivos ya se
resolvieron y, para esos mismos cortes, recalcula lo que habría enviado el
modelo del que desciende (`parent_model_id`): la misma ficha sin la capa que
se le agregó. Mismos ciclos, mismos datos, así que la diferencia es el modelo
y no la dificultad del periodo.

Sólo usa datos hasta cada `data_cutoff`: el contrafactual se calcula con
`predecir`, igual que en producción.

Uso:
    python3 -m ml.medir_vivo
    python3 -m ml.medir_vivo --model-id <uuid>
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ml"))

from collector.entorno import Supabase                            # noqa: E402
from data import cargar                                            # noqa: E402
from experimento_persistencia import todas                         # noqa: E402
from metrics import accuracy_oficial, accuracy_por_estacion        # noqa: E402
from pipeline.entregar import cargar_artefacto, predecir           # noqa: E402

CAPAS = ("correccion_nivel", "mezcla_persistencia")


def ficha_padre(sb: Supabase, ficha: dict) -> dict:
    """El padre registrado; si comparte artefacto, basta quitarle la capa nueva."""
    padre = sb.seleccionar("models", select="*",
                           model_id=f"eq.{ficha['parent_model_id']}")[0]
    if padre["artifact_sha256"] != ficha["artifact_sha256"]:
        raise SystemExit("el padre usa otro artefacto: este script compara capas, "
                         "no modelos distintos")
    return padre


def main(model_id: str | None) -> None:
    sb = Supabase()
    filtro = {"model_id": f"eq.{model_id}"} if model_id else {"status": "eq.active"}
    ficha = sb.seleccionar("models", select="*", limit=1, **filtro)[0]
    padre = ficha_padre(sb, ficha)
    nueva = [c for c in CAPAS if (ficha["hyperparams"] or {}).get(c)
             != (padre["hyperparams"] or {}).get(c)]
    print(f"champion : {ficha['hyperparams']['version']}")
    print(f"padre    : {padre['hyperparams']['version']}")
    print(f"diferencia: {', '.join(nueva) or 'ninguna'}")

    # El padre se evalúa con SU ficha pero el artefacto ya verificado del hijo:
    # comparten huella, así que es el mismo pickle.
    modelo = cargar_artefacto(sb, ficha)
    ancha, ctx, _ = cargar(origen="supabase")

    pred = pd.DataFrame(todas(sb, "predictions", submitted="is.true",
                              model_id=f"eq.{ficha['model_id']}",
                              select="cycle_id,station_id,target_at,issued_at,y_pred"))
    if pred.empty:
        raise SystemExit("todavía no hay entregas de este champion")
    for c in ("target_at", "issued_at"):
        pred[c] = pd.to_datetime(pred[c], utc=True).dt.tz_convert(ancha.index.tz)
    pred = pred[pred["target_at"] <= ancha.index[-1]]
    completos = pred.groupby("cycle_id")["target_at"].transform("size") == 48
    pred = pred[completos]
    if pred.empty:
        raise SystemExit("hay entregas, pero ningún ciclo resuelto completo")

    filas = []
    for o, g in pred.groupby("issued_at"):
        obj = list(zip(g["station_id"], g["target_at"]))
        antes = predecir(modelo, copy.deepcopy(padre), ancha, ctx, o, obj)
        for (e, t), ya, yb in zip(obj, g["y_pred"], antes):
            filas.append({"ciclo": o, "station_id": e, "y": float(ancha.at[t, e]),
                          "champion": ya, "padre": yb})
    d = pd.DataFrame(filas)

    print(f"\nciclos resueltos: {d['ciclo'].nunique()}  ({len(d)} valores)  "
          f"{d['ciclo'].min():%m-%d %H:%M} -> {d['ciclo'].max():%m-%d %H:%M} (virtual)")
    ch, pa = accuracy_oficial(d, "champion"), accuracy_oficial(d, "padre")
    print(f"accuracy  champion {ch:.2f}  ·  padre {pa:.2f}  ·  delta {ch - pa:+.2f}")

    por_ciclo = d.groupby("ciclo").apply(
        lambda g: pd.Series({"champion": accuracy_oficial(g, "champion"),
                             "padre": accuracy_oficial(g, "padre")}),
        include_groups=False)
    por_ciclo["delta"] = por_ciclo["champion"] - por_ciclo["padre"]
    print(f"gana en {int((por_ciclo['delta'] > 0).sum())} de {len(por_ciclo)} ciclos\n")
    print(por_ciclo.round(2).to_string())

    est = pd.DataFrame({"padre": accuracy_por_estacion(d, "padre"),
                        "champion": accuracy_por_estacion(d, "champion")}).round(1)
    est["delta"] = (est["champion"] - est["padre"]).round(1)
    print("\npor estación:\n" + est.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    main(ap.parse_args().model_id)
