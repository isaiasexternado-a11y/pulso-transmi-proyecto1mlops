# Pulso TransMi — Proyecto 1 MLOps

Reto de MLOps del curso de Ciencia de Datos, Universidad Externado de Colombia.
Pronosticar la demanda de pasajeros en 12 estaciones de TransMilenio **mientras los patrones cambian**,
sosteniendo el ciclo completo: **datos → modelo → predicción → evaluación → drift → reentrenamiento**.

**Dashboard:** https://pulso-transmi-proyecto1mlops.vercel.app

## Resultado

La competencia corrió del 21 de septiembre al **domingo 4 de octubre de 2026, 23:59 Bogotá**.
Para la nota cuenta el **Corte 1**, que empieza el 25 de septiembre a las 00:00 Bogotá
([decisión docente](https://github.com/uexternadojz/pulso-transmi/blob/main/docs/primer-corte-evaluacion.md));
la semana anterior fue de aprendizaje.

| Indicador (desde el corte, con la regla del profesor) | Valor |
|---|---|
| Ciclos con entrega / ciclos resueltos | **215 / 215 (cobertura 100 %)** |
| Accuracy del corte | **81,37 %** |
| Corte 1 hasta la pausa del 3-oct | 81,47 % |
| Fase final (régimen de demanda nuevo, 3-4 oct) | 77,75 % |
| Últimos 6 ciclos (champion final) | 92,42 % |

`Accuracy = 100 × max(0, 1 − WAPE)` por estación, sumando error y demanda de todos los ciclos del corte,
y promedio **no ponderado** de las 12 estaciones. Cálculo propio sobre Supabase; el número oficial es el
del leaderboard de la API.

La operación quedó apagada tras el cierre (workflows deshabilitados). Supabase, los modelos y el dashboard
se conservan para revisión.

## Arquitectura

| Capa | Herramienta | Pregunta que responde |
|---|---|---|
| Fuente oficial | API Pulso TransMi | ¿Qué sabemos hasta ahora? |
| Memoria operacional | Supabase (PostgreSQL) | ¿Qué observamos y qué predijimos? |
| Operador automático | GitHub Actions | ¿Cuándo y con qué versión se predijo? |
| Tablero (bono) | Vercel | ¿El modelo sigue siendo bueno? |

El estado vive en Supabase, nunca en el disco del runner. El reloj de la API es la autoridad: `cycle_id`,
`data_cutoff` y deadlines se consultan siempre, nunca se calculan desde la hora local.

### Workflows

| Workflow | Qué hace |
|---|---|
| [`predict.yml`](.github/workflows/predict.yml) | Turnos de ~25 min que consultan el ciclo vigente, recolectan el stream y entregan las 48 predicciones (12 estaciones × +15/+30/+45/+60). Cada turno encola su relevo porque el cron de GitHub se salta ejecuciones; el cron queda de red de seguridad. Antes de enviar se consulta en Supabase si el `cycle_id` ya tiene recibo aceptado: despertar varias veces no es entregar varias veces. Tres fallos seguidos ceden el turno a otro runner. |
| [`evaluate.yml`](.github/workflows/evaluate.yml) | Al cerrar cada turno: calcula métricas, señales de drift y la decisión de reentrenar (`retrain` · `blocked` · `keep`), corre la vigilancia de formato de datos y dispara `train.yml` cuando corresponde. |
| [`train.yml`](.github/workflows/train.yml) | Entrena desde la rama `experimentos-ml`, registra candidatos y los promueve sólo si ganan y completan una inferencia de prueba. Separado de la inferencia: un fallo entrenando no bloquea una entrega. |
| [`watchdog.yml`](.github/workflows/watchdog.yml) | Si no hay ningún `predict` vivo, arranca uno. Existe porque el 2026-09-23 la cadena murió y se perdieron dos ciclos. |

Secretos de Actions: `PULSO_API_KEY`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`. Ninguno está en el repositorio,
en tablas legibles desde el dashboard ni en variables públicas de Vercel.

### Código

```
collector/   Ingesta idempotente del stream (upsert + cursor en stream_cursor); soporta stream v1 y v2
pipeline/    entregar.py (inferencia y submission), evaluar.py (métricas, drift, decisión),
             promover.py (candidate → active), vigilar.py (huellas de formato y revisiones del profesor)
ml/          Carga de datos, features, métricas, modelos y empaquetado del champion
dashboard/   Sitio estático servido por Vercel; lee Supabase con la llave publicable
sql/         Migraciones posteriores al esquema inicial (ciclos, cursor, leaderboard, reloj, entrenamientos…)
supabase/    Esquema inicial, alineación con el ERD y RLS
scripts/     Carga del histórico semilla
eda/         Análisis exploratorio
hallazgos/   Reporte EDA, mapa de estaciones, modelo de datos y ERD (HTML)
docs/        Guías del curso y bitácora
```

La experimentación (entrenamiento, comparación de recetas, backtests) vive en la rama
[`experimentos-ml`](https://github.com/isaiasexternado-a11y/pulso-transmi-proyecto1mlops/tree/experimentos-ml); `main` conserva datos, esquema, EDA y operación.

## Modelo: cómo evolucionó el champion

Todo candidato se midió con **validación temporal** (nunca split aleatorio), sólo con información disponible
hasta `data_cutoff`, contra el piso **naive s-1 = 83,11 %**. Entrenar no es promover: una versión reemplaza
al champion sólo si supera los criterios y completa una inferencia de prueba. Cada versión queda en la tabla
`models` con su linaje y sus hiperparámetros, y el anterior pasa a `retired`, de modo que el rollback es
reactivarlo.

| Fecha | Champion | Por qué |
|---|---|---|
| 18-sep | GBM (HistGradientBoosting) con perfil por estación | Supera los baselines en los folds temporales |
| 25-sep | + mezcla con persistencia por horizonte | Bajo drift sintético: 81,41 → 84,44 % |
| 28-29 sep | + corrección de nivel por estación (ventana de 2 h con detección de quiebre) | Responde a saltos de nivel: 76,24 → 85,30 % en la continuación |
| 30-sep | Reentrenamiento automático cada 2 h, pesos por recencia, sin features de contexto congeladas | Revisiones 2 y 3 del drift cambian la forma de la demanda |
| 1-2 oct | + capa estacional que detecta el período (4 h) y selector entre variantes | Ciclos 18-21Z virtuales: ~74 → ~91 % |
| 4-oct | Fase final: persistencia → GBM dinámico reajustado por ciclo → **capa onda de 8 h** | Régimen nuevo sin parecido con el histórico; backtest causal 91,2 % vs 86,4 del GBM |

El champion final es `Onda de 8 h: armónicos + forma común (fase final)`: en cada ciclo detecta el período
de la onda, ajusta armónicos por estación y una plantilla común alineada, y promedia las variantes. Las capas
viven en la ficha del modelo (`hyperparams`), no en el pickle, así que cada una se puede apagar sin reentrenar.

## Monitoreo y drift

- `model_metrics`: accuracy por ciclo, estación y horizonte.
- `drift_signals`: `wape_24h`, `wape_7d`, `residual_bias`, `level_shift_7d`, `profile_corr`, `ingest_gap`.
- `retrain_decisions`: se reentrena sólo si `wape_24h` supera 0,1689 (accuracy < 83,11 %) en 3 corridas seguidas,
  con ≥576 observaciones nuevas y sin enfriamiento de 6 h. El resto de señales son alerta temprana: un único
  periodo malo no decide.
- `data_watch`: huellas del formato de la API y de las revisiones publicadas por el profesor. Así se detectó
  el stream v2 de la fase final (`measurement` con faltantes explícitos), que el collector y el modelo
  manejan sin tratar un faltante como cero.
- `training_runs`: cada entrenamiento, con su candidato y si se promovió.

## Base de datos

Supabase (PostgreSQL 17). El esquema base son 10 tablas y 3 vistas según el ERD en
[`hallazgos/erd.html`](hallazgos/erd.html); la operación agregó ciclos, submissions, cursor del stream, reloj,
entrenamientos, vigilancia y snapshots del leaderboard ([`sql/`](sql/)).

Las vistas `v_prediction_scores`, `v_accuracy_rolling` y `v_active_model` derivan el scoring:
`predictions` y `observations` no tienen FK entre sí porque al emitir la predicción la observación
todavía no existe. Se unen por `(station_id, target_at = observed_at)` cuando la realidad llega.

### Seguridad (RLS)

- La llave publicable (`anon`) sólo puede **leer**; no escribe nada. Es la que usa el dashboard.
- `run_events` no tiene política de lectura y `predictions.submit_response` está restringida por columna.
- El pipeline escribe con la llave de servicio, que vive en GitHub Actions Secrets.

## Reproducir

```bash
pip install -r requirements.txt          # scikit-learn fijado: el pickle del champion lo exige

# 1. Esquema: supabase/migrations/*.sql y luego sql/*.sql, en orden
# 2. Histórico semilla (idempotente)
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_KEY="<service_role key>"
export PULSO_DATA_DIR="ruta/al/sdk/data"
python3 scripts/load_historico.py

export SUPABASE_SECRET_KEY="$SUPABASE_KEY"   # nombre que usan collector/ y pipeline/

# 3. Operación (lo mismo que corre Actions)
export PULSO_API_KEY="<api key>"
python -m pipeline.entregar               # recolecta y entrega si hay ciclo abierto
python -m pipeline.evaluar                # métricas, drift y decisión
python -m pipeline.vigilar                # vigilancia de formato
```

En GitHub: cargar los tres secretos y habilitar los workflows (`gh workflow enable pulso-transmi-predict`, etc.).
El collector usa `upsert` y avanza el cursor sólo tras confirmar la transacción: correrlo dos veces no duplica.

## Baselines

| Baseline | Accuracy promedio |
|---|---|
| Naive s-1 (misma hora, semana pasada) | **83,11 %** |
| Naive d-1 (mismo momento de ayer) | 77,89 % |
| Media móvil 24 h | 37,88 % |

## Fuentes oficiales

- Contrato técnico: https://github.com/uexternadojz/pulso-transmi
- API y documentación: https://pulso-transmi.72-60-245-2.sslip.io/docs
