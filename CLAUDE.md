# Pulso TransMi — contexto del proyecto

Reto MLOps del curso de Ciencia de Datos, Universidad Externado (docente: Julián Zuluaga).
Guías oficiales (no se versionan; se bajan de [uexternadojz/pulso-transmi](https://github.com/uexternadojz/pulso-transmi) en `docs/guides/`):
- Metodológica v1.0 — `docs/pulso-transmi-guia-metodologica-v1.0.pdf`
- **Operativa v2.0** (2026-09-21) — `docs/pulso-transmi-guia-operativa-v2.0.pdf` y su `.md`. Manda sobre submissions y Actions.

## El reto

Pronosticar demanda de pasajeros en 12 estaciones de TransMilenio **mientras los patrones cambian**.
No se evalúa una métrica de laboratorio, sino la capacidad de sostener un ciclo operativo completo:
datos → modelo → predicción → evaluación → drift → reentrenamiento.

## Arquitectura (4 capas)

| Capa | Responsable | Pregunta que responde |
|---|---|---|
| API Pulso TransMi | fuente oficial, común a todos | ¿Qué sabemos hasta ahora? |
| Supabase (PostgreSQL) | memoria operacional del equipo | ¿Qué observamos y qué predijimos? |
| GitHub Actions | operador automático (el "reloj") | ¿Cuándo y con qué versión se predijo? |
| Vercel (bono) | dashboard de solo lectura | ¿El modelo sigue siendo bueno? |

El estado vive en Supabase, **nunca** en el disco del runner.

## Reloj operativo

- Granularidad del dato: **15 min virtuales** · Publicación: **2 timestamps cada 30 min reales**
- **Un ciclo por hora real.** Los ticks de datos son cada 30 min, pero sólo uno de cada dos abre ciclo.
- Ventana de entrega: **25 min** desde que abre · máximo **3 intentos *aceptados*** por ciclo
- Cada ciclo pide **48 valores** = 12 estaciones × 4 horizontes (`+15`, `+30`, `+45`, `+60`)
- Un solo envío cubre los 4 horizontes. No son 4 submissions.
- La API acepta el batch **completo** o lo rechaza completo.
- **Un rechazo por ciclo, corte, esquema o targets no consume intento**: los guardrails corren
  antes de contar el intento. Se corrige la causa y se reintenta dentro de la misma ventana.
- El último intento válido reemplaza al anterior como entrega oficial.

### Cómo se automatiza

El cron **no** calcula cuándo entregar: sólo despierta. Actions corre **cada 10 minutos**, consulta
si hay ciclo abierto y actúa sólo si corresponde. `404 no_open_cycle` termina en verde.
Despertar tres veces en una ventana no es entregar tres veces: antes de enviar se consulta en
Supabase si ese `cycle_id` ya tiene recibo para la versión del champion.

Dos workflows separados: inferencia (frecuente, liviana) y entrenamiento (deliberado). Un fallo
entrenando no puede bloquear una entrega.

## Métrica

```
WAPE     = Σ|error| / Σ(valores reales)      (por estación)
Accuracy = 100 × max(0, 1 − WAPE)
```
Oficial = promedio **no ponderado** de las 12 estaciones (una estación grande no puede tapar a las pequeñas).
Target ausente se evalúa como predicción cero.

## Reglas que no se negocian

- **El reloj de la API es la autoridad.** Nunca fabricar `cycle_id`, `data_cutoff` ni deadlines desde la hora local ni desde el cron. Consultar siempre el ciclo vigente.
- **Validación temporal**, nunca split aleatorio: mezclar futuro y pasado da resultados optimistas.
- **Solo información disponible hasta `data_cutoff`.** Nada de fuga temporal.
- **Idempotencia**: el collector usa `upsert` y avanza el cursor solo tras confirmar la transacción. Correrlo dos veces no puede duplicar.
- **Entrenar ≠ promover.** Una versión reemplaza al champion solo si supera los criterios y completa una inferencia de prueba.
- **La API key vive en GitHub Actions Secrets.** Jamás en el repo, en tablas visibles desde el dashboard ni en variables públicas de Vercel.
- Reentrenar no se decide por un único periodo malo: se consideran persistencia, volumen de datos nuevos y tiempo desde el último entrenamiento.

## Trabajo de machine learning — rama `experimentos-ml`

**Todo lo de ML va obligatoriamente en la rama `experimentos-ml`**: entrenamiento, features,
notebooks, comparación de modelos y artefactos. `main` conserva datos, esquema, EDA y operación.
No mezclar experimentos en `main`; el champion se promueve a `main` solo cuando gana.

Para que la comparación signifique algo, todos los candidatos se miden igual:

- **Validación temporal**, nunca split aleatorio.
- Métrica de decisión: `Accuracy = 100 × max(0, 1 − WAPE)`, promedio **no ponderado** de las 12 estaciones.
- Todo candidato se compara contra los baselines. El piso a superar es **naive s-1 = 83,11 %**.
- Cada experimento registra ventana de entrenamiento, features, hiperparámetros y resultado.
- Gana el de mejores métricas, pero además debe completar una inferencia de prueba antes de promoverse.

## Estado actual

**La competencia está corriendo** desde el 2026-09-21 15:30Z (reloj `official-20260921`, estado `running`).
El histórico semilla (45 días, 51.840 obs, 2026-07-26 → 2026-09-08) se extiende ahora por el stream.

- **Corte 1 en curso** desde 2026-09-24 05:00Z (00:00 Bogotá), inicio fijo. Cuenta ciclos resueltos
  abiertos desde ahí; una ausencia es predicción cero. Evidencia para la nota, no nota oficial.
  Definición: `docs/primer-corte-evaluacion.md` del repo del profe. **La fase de drift arranca el 2026-09-25 en la noche.**
- El collector ingesta el stream a Supabase de forma idempotente, con cursor en `stream_cursor`.
- El champion está en Supabase Storage (`modelos/champion/`) y registrado en `models` con `status=active`.
  Desde el 2026-09-25 es `gbm + perfil (mae) x persistencia`: el mismo artefacto GBM más una mezcla
  con el último observado en `data_cutoff`, con pesos por horizonte en `hyperparams.mezcla_persistencia`
  (+15→0,6 · +30→0,4 · +45/+60→0,2). La aplica `pipeline/entregar.py::predecir`. Evidencia en
  `ml/experimento_persistencia.py` (rama ML): 81,41 → 84,44 % bajo drift, −0,16 sin drift.
- **El contexto de la API está congelado** en 2026-09-08 23:45-05 mientras las observaciones avanzan.
  `ml/data.py` arrastra el último valor conocido.
- El monitoreo corre: `evaluate.yml` llena `model_metrics`, `drift_signals` (wape_24h, wape_7d,
  residual_bias, level_shift_7d, profile_corr, ingest_gap) y `retrain_decisions` (retrain · blocked · keep),
  y dispara `train.yml`, que registra candidatos. `promover` los activa tras la inferencia de prueba.
- Desde el viernes 11 virtual (stream), 05000, 07107, 07111 y 09122 subieron 13–29 % y alargaron el pico.

### Trampas ya pisadas

- `model.version` debe cumplir `^[A-Za-z0-9][A-Za-z0-9._:/-]*$` (máx. 64). Un `+` da 422.
- La mezcla vive en la ficha, no en el pickle: un modelo nuevo la hereda en `ml/entrenar.py::registrar`.
  Si se registra un candidato por otro camino, hay que copiarla o se apaga al promover.
- Un runner de Actions puede quedarse sin red hacia la API mientras la API responde desde fuera.
  `predict.yml` cede el turno tras 3 fallos seguidos.
- En SQL, `observations.demand` es `integer`: castear antes de dividir o el WAPE sale 0.

- `pulso-transmi-sdk/` — SDK del profesor con los datos semilla (`data/*.csv`)
- `eda/` — análisis exploratorio. Mejor baseline: naive s-1 (misma hora, semana pasada), accuracy 83,11 %
- `hallazgos/` — reporte EDA, mapa de estaciones, modelo de datos y ERD (10 tablas, 3 vistas)
- `docs/` — guía metodológica del profesor
- Supabase: proyecto `proyecto1` (`twjvjqyqxcfxbacewhdh`, sa-east-1). Histórico cargado y esquema alineado al ERD.

## Pendientes conocidos

1. Medir la mezcla con persistencia en vivo durante la fase de drift y recalibrar los pesos si cambia
   el régimen (`python3 -m ml.experimento_persistencia --particion <ts>`).
2. El backtest de `ml/entrenar.py` compara recetas crudas: un ganador podría rendir distinto con la mezcla.
3. El GBM depende de 5 features de contexto que ya no se publican. Vale la pena un candidato sin contexto.
4. Dashboard en Vercel (bono).

### Vocabularios que impone el esquema

Antes de insertar, consultar los `CHECK`: el esquema ya define los valores válidos y no coinciden
con los nombres obvios.

| Tabla | Columna | Valores |
|---|---|---|
| `pipeline_runs` | `trigger` | `schedule` · `manual` · `retry` |
| `pipeline_runs` | `status` | `running` · `success` · `failed` · `partial` |
| `models` | `status` | `candidate` · `active` · `retired` · `rejected` (el champion es `active`) |
| `models` | `kind` | `baseline` · `ml` |

RLS ya está activo (solo lectura para `anon`, escritura con `service_role`) y el repositorio
público del equipo ya existe: `isaiasexternado-a11y/pulso-transmi-proyecto1mlops`.

## Evaluación

Experimentación y validación temporal 20 % · Comprensión y calidad de datos 15 % · Collector y persistencia 15 % · Inferencia y submissions 15 % · Monitoreo, drift y reentrenamiento 15 % · Versionamiento y promoción 10 % · Reproducibilidad y documentación 10 %

Bonos: dashboard en Vercel, MLflow, pruebas automatizadas, estrategia de rollback, drift más allá del desempeño.

## Fuentes oficiales

- Contrato técnico: https://github.com/uexternadojz/pulso-transmi
- API y docs: https://pulso-transmi.72-60-245-2.sslip.io/docs
