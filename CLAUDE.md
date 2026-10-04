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
  La idea sale de la retroalimentación compartida en el curso (corregir con ventanas de 1–2 h); la adaptamos y medimos aquí.
- El cierre de la competencia es el **viernes 2 de octubre 23:59 Bogotá**.
  **Pausa**: el último tick salió 2026-10-03 04:50Z (23:50 Bogotá) y desde ahí `/v1/clock` responde sólo
  `{state: "waiting", server_time}`, sin ciclo abierto. El profe dijo en clase que el proyecto va hasta el
  domingo, pero no hay commits ni cambios de `meta` que lo confirmen. Los workflows siguen despertando y
  retoman solos cuando el reloj vuelva. Ambas formas del reloj ya están aceptadas en `data_watch`.
- **Fase final** (API 0.9.0, revisión 4, `docs/fase-final.md` del profe): reabrió el 2026-10-03 22:49Z, **cierre
  domingo 4-oct 23:59 Bogotá** (`2026-10-05T04:59Z`), régimen de demanda nuevo. Lo observado después de
  2026-09-20T12:00Z virtual llega como **stream v2**: sin `demand`, con `schema_version: 2` y
  `measurement = {value: "546.00" | null, unit: "passengers", quality: "observed" | "missing"}`; una página
  mezcla v1 y v2. Un faltante no es cero: `collector/recolectar.py::demanda` no lo guarda, `ml/data.py::cargar`
  rellena la grilla con el último valor observado de cada estación y `pipeline/entregar.py` extiende hasta el
  corte si faltan las 12 (hueco ≤ 1 h). Primeras filas v2 ingestadas 23:20Z (run 983), 24/24 `observed`.
  **Régimen nuevo** desde 2026-09-20T12:00Z virtual: se acabó la oscilación de 4 h y la demanda pasó a tendencias
  suaves por estación. El champion con capa estacional cayó a 45 % y 21 % (ciclos 12:00Z y 13:00Z); sin la capa,
  38 %. En los 5 orígenes de 15 min disponibles: persistencia 61,5 %, con tendencia amortiguada (k=1, φ=0,5) 65,2 %.
  2026-10-04 01:58Z se promovió `gbm + persistencia pura (fase final)` (`20261004T011104Z-pers1`): mismo artefacto,
  `mezcla_persistencia` en 1,0 los 4 horizontes y `estacional_apagada`. **No es un reentrenamiento.** Primera entrega:
  ciclo 15:00Z virtual, intento 2, 02:00:52Z. **`train.yml` está deshabilitado** a mano: `medir_vivo` mira 24 h casi
  todas del régimen viejo y devolvería la capa estacional. Reactivar con 6-12 h del régimen nuevo y comparar en lo vivo.
  02:10Z: `-pers2` (`gbm + persistencia pura sin nivel`), con `correccion_nivel_apagada`: en 9 orígenes del régimen
  nuevo, persistencia 66,4 % sin nivel vs 62,6 % con nivel; tendencia amortiguada 69,7 % (no implementada).
  Primera entrega: ciclo 15:00Z virtual, intento 3, 02:11:40Z.
  **Capa AR** (2026-10-04, `pipeline/entregar.py::autorregresivo`, `hyperparams.ar`): con 11 h del régimen nuevo
  se ve que cada estación oscila con su propio período de 7-9 h (02300 7,5 h, 06111 7,25 h, 06000 9,5 h…) y el
  histórico no se parece (correlación 0,2 con d-1 y s-1). Ridge común a las 12 estaciones sobre los cambios de los
  últimos 8 pasos normalizados por el nivel de 4 h, una por horizonte, reajustada en cada ciclo desde
  max(`desde`, origen − 24 h) sólo con datos hasta el corte. Backtest causal en 31 orígenes (15:00Z a 22:30Z
  virtual): **78,5 %** vs persistencia 72,4, tendencia amortiguada 74,7, ondas por período ≤ 64 (sus rezagos de
  7-9 h caían en el régimen viejo). Los rezagos no deben cruzar `desde`: cruzarlos cuesta 3 puntos. Si la ficha
  trae `ar`, reemplaza al modelo y a las capas. `evaluate.yml` ya no falla cuando `train.yml` está deshabilitado.
  Con `ar.modelo = "gbm"` la ridge se cambia por un HistGradientBoosting (MAE) con los mismos rasgos más el nivel
  relativo: **81,1 %** en los mismos 31 orígenes (84,4 en la segunda mitad); entrenado con 3 o 7 días de historia
  da 79,1-79,2 y la mezcla AR+GBM 80,8. La idea de aprender la dinámica de rezagos cortos sale de la
  retroalimentación compartida en el curso. **Champion desde 2026-10-04 11:2xZ**: `GBM dinámico reajustado por ciclo
  (fase final)` (`20261004T011104Z-argbm`), `ar = {rezagos 8, lam 1, desde 2026-09-20T12:00Z, ventana_h 24, modelo gbm}`.
  Promover sólo con un turno de `predict` que ya corra el código de la capa (ver trampas).
- Desde el 2026-09-30 03:00Z **`train.yml` corre cada 2 h** y el champion se renueva solo.
  La cadencia la lleva `evaluate.yml`, no el cron (GitHub se saltó 5 de 6 el 30-sep): en cada corrida
  mira el último run de `train.yml` y lo dispara si pasaron ≥110 min y no hay uno en cola o corriendo.
  Cada corrida queda en `training_runs` (la abre `train.yml` al arrancar y la cierra un job `if: always()`),
  y el dashboard muestra "Último entrenamiento" y la línea de tiempo con los huecos de más de 3 h. Además de los
  folds, `ml/entrenar.py` mide cada receta en las últimas 24 h (`medir_vivo`) y hay tres rutas a
  `candidate`: historia (folds), cambio de receta (gana ≥0,10 en lo vivo sin perder >0,30 en folds) y
  refresco (la receta del champion con datos nuevos). Entre los elegibles gana el mejor en lo vivo.
  La receta del champion se lee de `hyperparams.receta`. Primer resultado: `gbm + perfil por estacion`
  (`20260930T025733Z`), 87,82 vs 86,03 del GBM congelado en las últimas 24 h.
- Desde el 2026-09-30 17:10Z el champion es `gbm + perfil sin contexto · 14d` (`fs-v2-sin-contexto`), por
  cambio de receta: 86,99 vs 86,77 en las últimas 24 h. Ya no depende de las 5 features de contexto congeladas.
- **Revisión 2 del drift** (activada el 2026-09-30 14:52 Bogotá, `docs/drift-operations.md` del profe): cambia la
  FORMA de la demanda (`peak_shift` del generador), 6 h de transición y régimen estable hasta el cierre. Se notó
  desde el ciclo 04:00Z virtual del 18 (87 → 68 %). Respuesta, medida en `ml/experimento_forma.py` (cambio de forma
  sintético, 2 anclas x 3 escenarios, reentreno cada 2 h): recetas con **pesos por recencia** (`semivida_h`),
  **reserva 2 h** y corrección de nivel **sin quiebre** (le gana a con quiebre 12 de 12). Contra la configuración
  anterior, +2,35 a +7,08 en los 6 escenarios. `medir_vivo` además mide cada receta con y sin nivel y el candidato
  hereda el estado que gane (`correccion_nivel_apagada` guarda la apagada). Activo desde 2026-09-30 22:46Z.
- **Revisión 3** (commit del profe 2026-09-30 20:32Z, `private-continuation-v1.2.0`): más exigente, transición de
  2 h virtuales, régimen estable después. Referencias del profe en 48 h: fija 55 %, adaptativa 80 %; las primeras 6 h
  ~55 % incluso adaptativas, y entre las horas 18-30 recuperan ~84 %. Es la que se ve desde el ciclo 04:00Z virtual.
  El repo sigue diciendo cierre viernes 2-oct 23:59; el equipo oyó que podría ser el domingo: la autoridad es el reloj
  de la API. Vigía en la nube cada 2 h (routine `trig_012tAcmDaGKiLSSo1qmwdkxx`): sólo diagnostica y avisa por Gmail.
- Desde el 2026-10-01 el interruptor de capas cubre mezcla y nivel: `medir_vivo` prueba todas las combinaciones y la
  ficha guarda la apagada en `<capa>_apagada`. Con la revisión 3 (picos de 1-2 h en la madrugada) el champion
  normalizado con semivida quedó **sin capas** (82,97 vs 81,92 con mezcla): el modelo normalizado ya sigue la escala
  y las capas contaban dos veces.
- **Capa estacional** (2026-10-01 15:20Z, `hyperparams.estacional`, `pipeline/entregar.py::estacional`): la revisión 3
  dejó una demanda que se repite **cada 4 h** (rezago 4 h: WAPE 0,096; cualquier otro rezago corto 0,7-1,0). En cada
  ciclo la capa busca el período de 2-12 h que mejor explica las últimas 12 h (sólo datos hasta el corte) y predice
  con el promedio de lo observado P y 2P antes: peso 1 bajo WAPE 0,15, 0 sobre 0,25. En régimen normal el mejor
  rezago corto da 0,44-0,56 y no se prende. Ciclos 18-21Z virtuales: 77,7/73,6/72,2/74,2 -> 86,3/91,1/92,6/92,2.
  `ml/entrenar.py::registrar` la hereda tal cual. Promovida sobre el champion semivida 12h (`…-est`).
  Desde las 16:07Z la ventana de detección es de **6 h** (antes 12): se prende 5 h antes tras un cambio de régimen y
  se apaga igual de rápido; en régimen normal el mejor rezago corto con 6 h nunca baja de 0,326 (con 4 h llega a 0,215,
  descartada). Entre períodos casi empatados (4 h y 8 h) gana el más corto (`EMPATE` = 25 %). La idea de los
  rezagos periódicos sale de la retroalimentación compartida en el curso; la adaptamos a nuestra capa y la medimos aquí.
  Desde el 2026-10-02 ~15:30Z, con la capa prendida, un **selector** (`selector_estacional`) elige cada ciclo entre
  `estK` (promedio de K oscilaciones) y `plantK` (forma de onda común a las 12 estaciones, desfasada y escalada por
  nivel), K=1..6, el de mejor accuracy en los 3 ciclos resueltos previos. Idea tomada de la
  retroalimentación compartida en el curso y adaptada a nuestro pipeline. Replay en ciclos reales 18-19 virtual: 91,03 -> 92,47 (últimos 12: 91,31 -> 93,03).
  `hyperparams.estacional.selector = false` lo apaga. Evidencia en `ml/experimento_selector.py` (rama ML).
- **Vigilancia de tipología** (el profe anunció que cambiará la tipología de los datos): `pipeline/vigilar.py` corre en
  cada evaluate.yml y deja en `data_watch` huellas del formato de la API, de las revisiones del profe y de caídas
  bruscas. Dashboard: banner rojo + sección "Vigilancia de datos". El vigía en la nube corre cada hora y manda correo
  '[Pulso TransMi] CAMBIO EN LOS DATOS'. Tras adaptarse a un formato nuevo: `python -m pipeline.vigilar --aceptar`.
- Antes del 2026-09-30 22:46Z **se entrenaba con 6 h de reserva** (`RESERVA_H`). Sin ella, un modelo recién entrenado reproduce las
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
3. ~~Candidato sin contexto~~: champion desde el 2026-09-30 17:10Z. Vigilar que siga ganando en lo vivo.
4. ~~Dashboard en Vercel (bono)~~: en https://pulso-transmi-proyecto1mlops.vercel.app, se redespliega solo con cada push a `main`
   (integración Git de Vercel, sirve `dashboard/` sin build). Lee Supabase con la llave publicable; RLS bloquea escrituras.

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
