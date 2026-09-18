# Pulso TransMi — Proyecto 1 MLOps

Reto de MLOps del curso de Ciencia de Datos, Universidad Externado de Colombia.
Pronosticar la demanda de pasajeros en 12 estaciones de TransMilenio mientras los patrones cambian,
sosteniendo el ciclo completo: **datos → modelo → predicción → evaluación → drift → reentrenamiento**.

## Estado

| Fase | Estado |
|---|---|
| 1 · Comprender (EDA, hipótesis) | hecho |
| 2 · Construir la memoria (Supabase, histórico, collector) | esquema y histórico listos; collector pendiente |
| 3 · Experimentar (baselines, champion) | baselines calculados; champion pendiente |
| 4 · Operar (GitHub Actions, submissions) | pendiente |
| 5 · Aprender del error (drift, reentrenamiento) | pendiente |

La competencia **aún no empieza**: el reloj de la API responde `waiting` y el stream incremental está vacío.
Histórico disponible: 45 días, 12 estaciones, 51.840 observaciones (2026-07-26 → 2026-09-08), cada 15 minutos.

## Estructura

```
eda/         Análisis exploratorio y resumen en JSON
hallazgos/   Reporte EDA, mapa de estaciones, modelo de datos y ERD (HTML)
supabase/    Migraciones versionadas del esquema
scripts/     Carga del histórico a Supabase
docs/        Bitácora de avance
```

El SDK del profesor (`uexternadojz/pulso-transmi-sdk`) se usa como dependencia externa y no se copia aquí.

## Base de datos

Supabase (PostgreSQL 17), 10 tablas y 3 vistas según el ERD en [`hallazgos/erd.html`](hallazgos/erd.html).

| Tabla | Propósito |
|---|---|
| `stations` | Catálogo de las 12 estaciones (semilla) |
| `pipeline_runs` | Una fila por ejecución del pipeline |
| `observations` | Demanda real por estación y periodo |
| `context` | Clima y eventos por periodo |
| `run_events` | Log por etapa de cada ejecución |
| `drift_signals` | Señales de drift medidas por corrida |
| `retrain_decisions` | Decisión de conservar o reentrenar |
| `models` | Modelos entrenados, linaje y estado |
| `model_metrics` | Métricas por split, fold y estación |
| `predictions` | Predicciones emitidas |

Las vistas `v_prediction_scores`, `v_accuracy_rolling` y `v_active_model` derivan el scoring:
`predictions` y `observations` **no** tienen FK entre sí porque al emitir la predicción la observación
todavía no existe. Se unen por `(station_id, target_at = observed_at)` cuando la realidad llega.

### Aplicar el esquema

```bash
supabase db push          # o ejecutar supabase/migrations/*.sql en orden
```

### Cargar el histórico

```bash
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_KEY="<publishable key>"
export PULSO_DATA_DIR="ruta/al/sdk/data"
python3 scripts/load_historico.py
```

Es idempotente (`upsert` con `merge-duplicates`): correrlo dos veces no duplica filas.

## Baselines

Accuracy = `100 × max(0, 1 − WAPE)`, promedio no ponderado de las 12 estaciones.

| Baseline | Accuracy promedio |
|---|---|
| Naive s-1 (misma hora, semana pasada) | **83,11 %** |
| Naive d-1 (mismo momento de ayer) | 77,89 % |
| Media móvil 24 h | 37,88 % |

## Pendientes

1. **Activar RLS** en las 10 tablas: hoy la key pública permite leer y escribir todo.
2. **Modelar ciclos y submissions**: falta `cycle_id`, `submission_id` y llave de idempotencia.
   Sin eso, varios intentos sobre el mismo ciclo contarían el mismo target más de una vez al calcular accuracy.
3. Collector incremental, workflows de GitHub Actions, modelo champion y submissions.

## Fuentes oficiales

- Contrato técnico: https://github.com/uexternadojz/pulso-transmi
- API y documentación: https://pulso-transmi.72-60-245-2.sslip.io/docs
