# Informe final — Pulso TransMi, Proyecto 1 MLOps

Universidad Externado de Colombia · Ciencia de Datos · Docente: Julián Zuluaga
Repositorio: https://github.com/isaiasexternado-a11y/pulso-transmi-proyecto1mlops
Dashboard: https://pulso-transmi-proyecto1mlops.vercel.app

## 1. Resumen

Construimos un sistema que pronostica la demanda de 12 estaciones de TransMilenio a +15, +30, +45 y +60 minutos
y sostiene solo el ciclo datos → modelo → predicción → evaluación → drift → reentrenamiento, sobre la API oficial,
Supabase, GitHub Actions y Vercel.

Desde el inicio del Corte 1 (25-sep 00:00 Bogotá) hasta el cierre (4-oct 23:59 Bogotá):

| Indicador | Valor |
|---|---|
| Ciclos con entrega / ciclos resueltos | **215 / 215 (cobertura 100 %)** |
| Accuracy del corte (WAPE por estación, promedio no ponderado) | **81,37 %** |
| Hasta la pausa del 3-oct | 81,47 % |
| Fase final (régimen nuevo) | 77,75 % — 92,42 % en los últimos 6 ciclos |

Evidencia operativa en Supabase al cierre: 1.099 corridas del pipeline, 14.208 predicciones guardadas,
8.952 métricas, 23.771 señales de drift, 475 decisiones de reentrenamiento (26 `retrain`, 227 `blocked`,
222 `keep`), 43 entrenamientos registrados y 58 versiones de modelo (1 activa, 56 retiradas, 1 rechazada), todas
con commit, ventana de entrenamiento, features, hiperparámetros, artefacto y hash.

## 2. Qué cambió

Los datos cambiaron cinco veces durante la competencia y cada cambio exigió una respuesta distinta.

| Momento (hora virtual) | Cambio | Efecto en un modelo fijo |
|---|---|---|
| Desde el 11-sep | 05000, 07107, 07111 y 09122 suben 13-29 % y alargan el pico | sesgo por debajo en esas estaciones |
| Desde el 13-sep | 05100 cae a menos de la mitad; 07111 sube ~40 %, 06000 ~20 % | 05100 llegó a 20 % de accuracy |
| 16-sep 08:00 | Saltos bruscos de nivel: 02300 y 05000 a 2-4,7×, 05100 a 0,15× | 76 % global |
| 18-sep 04:00 (revisiones 2 y 3 del profesor) | Cambia la **forma** del día: picos de 1-2 h, demanda que se repite cada 4 h | 87 → 68 % |
| 20-sep 12:00 (fase final) | Régimen nuevo: onda de ~8 h, sin parecido con el histórico; stream v2 con faltantes explícitos | el champion de entonces cayó a 21 % |

Además, el contexto de la API (clima, eventos) quedó congelado el 8-sep mientras las observaciones avanzaban:
las features que dependían de él se volvieron ruido.

Cambió también la operación: el cron de GitHub se salta ejecuciones bajo carga (con `*/10` llegó a disparar
una vez en nueve horas) y un runner puede quedarse sin red hacia la API mientras la API responde desde fuera.

## 3. Qué funcionó

**Operación**

- **El reloj de la API como única autoridad.** Nunca calculamos `cycle_id` ni deadlines desde la hora local; por eso
  las pausas y reaperturas de la API (28-sep, 3-oct) no produjeron entregas inválidas ni ciclos fantasma.
- **Turnos encadenados en vez de cron.** `predict.yml` corre turnos de ~25 min que vigilan la API y encolan su
  propio relevo; `watchdog.yml` revive la cadena si muere. Resultado: 215/215 desde el corte.
- **Idempotencia verificada.** Antes de enviar se consulta en Supabase si el ciclo ya tiene recibo aceptado; el
  collector hace `upsert` y avanza el cursor sólo tras confirmar. Varias despertadas en la misma ventana dejaron
  `attempt=1`.
- **Inferencia separada del entrenamiento.** Cuando el entrenamiento falló o lo deshabilitamos a propósito, las
  entregas siguieron.
- **Vigilancia del formato de los datos** (`data_watch`): detectó el stream v2 de la fase final, que trae
  `measurement` con faltantes explícitos. Lo adaptamos sin tratar un faltante como cero.

**Modelo**

- **Capas sobre la ficha, no dentro del pickle.** Mezcla con persistencia, corrección de nivel, capa estacional y
  capa onda viven en `hyperparams`; se prenden o apagan sin reentrenar, y el entrenamiento decide cuáles conservar
  según el desempeño en las últimas 24 h. Eso también es la estrategia de rollback: volver a una versión
  `retired` o apagar una capa.
- **Corregir con ventanas cortas.** La corrección de nivel con 2 h de ventana y detección de quiebre llevó 05100
  de 2,9 % a 77 % en la continuación del escenario.
- **Pesos por recencia y reserva de 2 h al reentrenar.** Ante el cambio de forma, +2,4 a +7,1 puntos en 6
  escenarios sintéticos frente a la configuración anterior.
- **Detectar el período en cada ciclo.** La capa estacional (4 h) llevó ciclos de ~74 % a ~91 %; en la fase
  final, la capa onda (8 h, armónicos por estación más forma común) dio 91,2 % en backtest causal frente a
  86,4 % del GBM y cerró con 92,42 % en los últimos 6 ciclos.

Todas las promociones se midieron con validación temporal y backtests causales (sólo datos hasta el corte),
contra el piso naive s-1 de 83,11 %. Los experimentos y sus resultados están en la rama `experimentos-ml`
(`ml/experimento_*.py`, `ml/resultados/`).

## 4. Qué no funcionó

- **El cron como reloj.** Antes del corte perdimos ~18 ciclos por ejecuciones saltadas, y el 23-sep la cadena
  murió 1 h 46 min (dos ciclos). De ahí salieron los turnos encadenados y el watchdog.
- **Reentrenar sin reserva.** Un modelo recién entrenado reproducía las últimas horas, los *backcasts* de la
  corrección de nivel salían in-sample y el quiebre se apagaba: cada 2 h sin reserva quedaba por debajo del modelo
  congelado (84,72 vs 85,47).
- **Capas que cuentan dos veces.** Con el modelo normalizado por nivel, la mezcla con persistencia empeoraba
  (81,92 vs 82,97 sin capas).
- **Un champion afinado para un régimen.** Al cambiar el régimen el 20-sep, la capa estacional siguió viendo
  4 h y el champion cayó a 21-45 %. La regla de reentrenamiento (3 corridas seguidas bajo el piso) es deliberadamente
  lenta para no reaccionar a un solo periodo malo, y frente a una ruptura total llegó tarde: la respuesta fue manual
  (persistencia pura en minutos, luego el GBM dinámico y la capa onda).
- **Comparar sólo contra el champion.** El GBM dinámico ganó a las 11:15Z porque aún no había un período completo
  de la onda; al pasar 1,5 períodos había que volver a comparar todas las familias de modelos, no sólo afinar el
  vigente.

## 5. Qué haríamos después

1. **Detección automática de ruptura de régimen** que cambie a un respaldo robusto (persistencia) en el mismo
   ciclo, sin esperar las 3 corridas de la regla de reentrenamiento.
2. **Selección de familia en línea:** en cada ciclo, medir en los últimos ciclos resueltos varias familias
   (persistencia, GBM dinámico, armónicos, champion) y elegir o combinar, en vez de un único champion afinado.
3. **Pruebas como compuerta de la operación**: hoy corren en cada push; el siguiente paso es que `predict.yml` y
   `train.yml` sólo usen código cuyo commit las pasó, y sumar pruebas de integración contra un stream grabado.
4. **MLflow en línea**: hoy el almacén se genera desde Supabase y los resultados (`ml/registro_mlflow.py`); lo ideal
   es que `train.yml` registre cada candidato en un servidor MLflow al momento de entrenarlo.
5. **Ensayar los cambios de formato antes de que lleguen**, con un stream sintético que mezcle versiones y
   faltantes.

## 6. Mapa de entregables

| Entregable de la guía metodológica | Dónde está |
|---|---|
| Repositorio público, organizado y reproducible | [README](../README.md) — estructura, setup y comandos |
| Descripción del problema y análisis exploratorio | [README](../README.md), [`eda/`](../eda/), [`hallazgos/reporte_eda.html`](../hallazgos/reporte_eda.html), [`docs/avance-2026-09-16.md`](avance-2026-09-16.md) |
| Supabase con histórico migrado y esquema documentado | [`supabase/migrations/`](../supabase/migrations/), [`sql/`](../sql/), [`hallazgos/erd.html`](../hallazgos/erd.html), [`scripts/load_historico.py`](../scripts/load_historico.py) |
| Collector incremental automatizado | [`collector/recolectar.py`](../collector/recolectar.py), invocado en cada turno de `predict.yml` |
| Baselines y experimentos con validación temporal | Baselines en el README; experimentos en la rama `experimentos-ml` (`ml/backtest.py`, `ml/experimento_*.py`, `ml/resultados/`) |
| Registro de versiones y modelo champion | Tabla `models` (58 versiones con commit, ventana, features, artefacto y hash); [`pipeline/promover.py`](../pipeline/promover.py) |
| Inferencia periódica mediante GitHub Actions | [`predict.yml`](../.github/workflows/predict.yml), [`watchdog.yml`](../.github/workflows/watchdog.yml), [`pipeline/entregar.py`](../pipeline/entregar.py) |
| Predicciones almacenadas y submissions trazables | Tablas `predictions`, `submissions`, `forecast_cycles` |
| Evaluación de accuracy a lo largo del tiempo | [`pipeline/evaluar.py`](../pipeline/evaluar.py), tabla `model_metrics`, vistas `v_prediction_scores` y `v_accuracy_rolling`, dashboard |
| Detección de degradación y decisión de reentrenamiento | Tablas `drift_signals`, `retrain_decisions`, `training_runs`; [`train.yml`](../.github/workflows/train.yml) |
| Informe final | Este documento |

| Bono | Estado |
|---|---|
| Dashboard en Vercel | Sí — https://pulso-transmi-proyecto1mlops.vercel.app |
| Estrategia de rollback del modelo | Sí — versiones `retired` reactivables y capas apagables desde la ficha |
| Monitoreo de drift más allá del desempeño | Sí — `profile_corr`, `level_shift_7d`, `residual_bias`, `ingest_gap` y vigilancia de formato (`data_watch`) |
| MLflow | Sí — estudios, linaje de las 58 versiones y Model Registry con alias `champion` ([`docs/mlflow.md`](https://github.com/isaiasexternado-a11y/pulso-transmi-proyecto1mlops/blob/experimentos-ml/docs/mlflow.md), rama `experimentos-ml`) |
| Pruebas automatizadas | Sí — 50 pruebas en [`tests/`](../tests/) que corren en cada push ([`tests.yml`](../.github/workflows/tests.yml)), además de los controles en tiempo de ejecución (validación del batch, huellas de formato) |
