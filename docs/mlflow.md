# MLflow — experimentos y linaje de versiones

El almacén `mlflow/` (SQLite + artefactos, ~5 MB) se versiona en esta rama y abre en cualquier clon:

```bash
pip install "mlflow>=3"
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db     # desde la raíz del repo
```

| Dónde | Qué hay |
|---|---|
| Experimento `pulso-experimentos` | Un run por estudio de `ml/resultados/*.json`: backtest de baselines y candidatos (con un run hijo por modelo comparado), persistencia, nivel, quiebre, reentreno, forma, contexto y selector. Parámetros, métricas y el JSON como artefacto. |
| Experimento `pulso-versiones` | Un run por cada una de las 58 versiones de la tabla `models`: algoritmo, ventana de entrenamiento, features, hiperparámetros y capas (`hp.*`), commit, artefacto y hash, padre (`parent_run_id`), estado, métricas de backtest por fold y de producción (acumulada, 24 h, 7 d). La ficha va como artefacto. |
| Model Registry `pulso-transmi-demanda` | 58 versiones en orden de creación, etiquetadas por estado; el alias `champion` apunta a la activa (v58, capa onda). |

Lo genera `python -m ml.registro_mlflow`, que lee los resultados de esta rama y la tabla `models` de Supabase
(`SUPABASE_URL` y `SUPABASE_SECRET_KEY` en `.env`) y regenera el almacén desde cero cada vez.

Supabase sigue siendo la fuente de verdad operativa: el pipeline corre en runners efímeros y no puede depender
de un servidor de MLflow. MLflow es la vista de experimentación y linaje; responde qué se entrenó, con qué datos
y por qué se promovió.
