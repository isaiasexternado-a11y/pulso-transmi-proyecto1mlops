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
- Desde el 2026-09-28 13:17Z el champion es `… x persistencia x nivel`
  (`20260918T202534Z-pers-06-04-02-02-niv-12-075`): encima de la mezcla, cada estación se
  multiplica por `1 + α (r − 1)`, con `r = observado / predicho` en las últimas 12 h resueltas
  (α=0,75, r∈[0,4; 2,5]) y lo predicho sacado de *backcasts* del propio champion.
  Vive en `hyperparams.correccion_nivel`. Evidencia en `ml/experimento_nivel.py` (rama ML):
  77,41 → 83,23 % bajo drift, −0,07 sin drift; 05100 pasó de 20 % a 77 %.
- Desde el 2026-09-29 21:25Z el champion es `… x nivel con quiebre`
  (`20260918T202534Z-pers-06-04-02-02-nivq-02-050`): la corrección de nivel mira 2 h (α=0,5) y,
  si r de 2 h y de 4 h se desvían >20 % en el mismo sentido, corrige completo con r∈[0,15; 5]
  (`correccion_nivel.quiebre`). Responde a la continuación del escenario del 16 virtual 08:00
  (02300/05000 a 2–4,7×, 05100 a 0,15×, 03000 a 0,4×). Evidencia en `ml/experimento_nivel_corto.py`:
  en la continuación 76,24 → 85,30 %, sin costo sin drift; 05100 de 2,9 % a 77 %.
  La idea viene de los punteros del leaderboard, que corrigen con ventanas de 1–2 h.
- El cierre de la competencia es el **viernes 2 de octubre 23:59 Bogotá**.
- Desde el 2026-09-30 03:00Z **`train.yml` corre cada 2 h** y el champion se renueva solo. Además de los
  folds, `ml/entrenar.py` mide cada receta en las últimas 24 h (`medir_vivo`) y hay tres rutas a
  `candidate`: historia (folds), cambio de receta (gana ≥0,10 en lo vivo sin perder >0,30 en folds) y
  refresco (la receta del champion con datos nuevos). Entre los elegibles gana el mejor en lo vivo.
  La receta del champion se lee de `hyperparams.receta`. Primer resultado: `gbm + perfil por estacion`
  (`20260930T025733Z`), 87,82 vs 86,03 del GBM congelado en las últimas 24 h.
- **Se entrena con 6 h de reserva** (`RESERVA_H`). Sin ella, un modelo recién entrenado reproduce las
  últimas horas, los backcasts de la corrección de nivel salen in-sample y el quiebre se apaga: cada 2 h
  sin reserva quedaba por debajo del congelado (84,72 vs 85,47); con reserva, 86,87
  (`ml/experimento_reentreno.py --cada 2 --reserva 6`).
- **El contexto de la API está congelado** en 2026-09-08 23:45-05 mientras las observaciones avanzan.
  `ml/data.py` arrastra el último valor conocido.
- El monitoreo corre: `evaluate.yml` llena `model_metrics`, `drift_signals` (wape_24h, wape_7d,
  residual_bias, level_shift_7d, profile_corr, ingest_gap) y `retrain_decisions` (retrain · blocked · keep),
  y dispara `train.yml`, que registra candidatos. `promover` los activa tras la inferencia de prueba.
  Reentrena sólo si `wape_24h` supera 0,1689 (accuracy < 83,11 %, el piso naive s-1) en 3 corridas
  seguidas **del champion vigente**, con ≥576 obs nuevas y sin enfriamiento de 6 h. El resto de las
  señales son alerta temprana (`keep`), no disparan.
- Desde el viernes 11 virtual (stream), 05000, 07107, 07111 y 09122 subieron 13–29 % y alargaron el pico.
- Desde el 13 virtual, 05100 cayó a menos de la mitad (~62k → ~26k/día), 07111 subió ~40 % y 06000 ~20 %.

### Trampas ya pisadas

- `model.version` debe cumplir `^[A-Za-z0-9][A-Za-z0-9._:/-]*$` (máx. 64). Un `+` da 422.
- La mezcla y la corrección de nivel viven en la ficha, no en el pickle: un modelo nuevo las hereda en
  `ml/entrenar.py::registrar`. Si se registra un candidato por otro camino, hay que copiarlas o se apagan al promover.
- No promover con una ventana abierta si el turno de `predict` en curso arrancó antes del código nuevo:
  cargaría la ficha nueva con el `predecir` viejo y entregaría bajo la versión nueva sin el ajuste.
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
2. Recalibrar L y α de la corrección de nivel con los ciclos en vivo del drift
   (`python3 -m ml.experimento_nivel --particion <ts>`; `python3 -m ml.medir_vivo` compara al champion con su padre).
   Desde el 2026-09-29 `ml/entrenar.py` mide cada receta CON las capas del champion y veta al candidato que no le
   gane en los últimos 3 días (`--sin-capas` reproduce la comparación vieja).
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
