# Pulso TransMi — contexto del proyecto

Reto MLOps del curso de Ciencia de Datos, Universidad Externado (docente: Julián Zuluaga).
Guía metodológica oficial v1.0: [`docs/pulso-transmi-guia-metodologica-v1.0.pdf`](docs/pulso-transmi-guia-metodologica-v1.0.pdf) — leerla antes de decisiones de diseño.

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

- Granularidad del dato: **15 min** · Publicación de datos: **30 min** · Ciclo de predicción: **60 min**
- Ventana de entrega: **25 min** desde que abre el ciclo · máximo **3 intentos válidos** por ciclo
- Cada ciclo pide **48 valores** = 12 estaciones × 4 horizontes (`+15`, `+30`, `+45`, `+60`)
- Un solo envío cubre los 4 horizontes. No son 4 submissions.
- La API acepta el batch **completo** o lo rechaza completo.

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

Fase actual: la competencia **no ha empezado** (el reloj responde `waiting`, el stream incremental está vacío).
Histórico fijo disponible: 45 días, 12 estaciones, 51.840 observaciones (2026-07-26 → 2026-09-08).

- `pulso-transmi-sdk/` — SDK del profesor con los datos semilla (`data/*.csv`)
- `eda/` — análisis exploratorio. Mejor baseline: naive s-1 (misma hora, semana pasada), accuracy 83,11 %
- `hallazgos/` — reporte EDA, mapa de estaciones, modelo de datos y ERD (10 tablas, 3 vistas)
- `docs/` — guía metodológica del profesor
- Supabase: proyecto `proyecto1` (`twjvjqyqxcfxbacewhdh`, sa-east-1). Histórico cargado y esquema alineado al ERD.

## Pendientes conocidos

1. **Modelar ciclos y submissions** — faltan `cycle_id`, `submission_id` y llave de idempotencia.
   Sin eso, los hasta 3 intentos de un mismo ciclo cuentan el mismo target varias veces en el accuracy.
2. Collector incremental, workflows de GitHub Actions, modelo champion, submissions.

RLS ya está activo (solo lectura para `anon`, escritura con `service_role`) y el repositorio
público del equipo ya existe: `isaiasexternado-a11y/pulso-transmi-proyecto1mlops`.

## Evaluación

Experimentación y validación temporal 20 % · Comprensión y calidad de datos 15 % · Collector y persistencia 15 % · Inferencia y submissions 15 % · Monitoreo, drift y reentrenamiento 15 % · Versionamiento y promoción 10 % · Reproducibilidad y documentación 10 %

Bonos: dashboard en Vercel, MLflow, pruebas automatizadas, estrategia de rollback, drift más allá del desempeño.

## Fuentes oficiales

- Contrato técnico: https://github.com/uexternadojz/pulso-transmi
- API y docs: https://pulso-transmi.72-60-245-2.sslip.io/docs
