-- Una fila por cada modelo que la capa AR entrena al entregar un ciclo.
--
-- Desde el 2026-10-04 (fase final) el champion no es un artefacto fijo: la
-- capa `hyperparams.ar` de `pipeline/entregar.py::autorregresivo` entrena en
-- cada ciclo un modelo por horizonte con los datos del régimen vigente hasta
-- el `data_cutoff`, y predice con él. `train.yml` está pausado. Sin esta
-- tabla esos reentrenamientos pasaban y se perdían: no quedaba con qué datos
-- se entrenó cada uno, que es lo que pide la guía como evidencia.
--
-- La escribe `pipeline/entregar.py` después de guardar el recibo; si falla,
-- la entrega ya quedó y el turno sigue.
create table if not exists public.ajustes_ciclo (
    cycle_id        text        not null,
    model_version   text        not null,
    horizon         smallint    not null check (horizon between 1 and 4),  -- pasos de 15 min
    model_id        uuid        references public.models (model_id),
    submission_id   text,
    data_cutoff     timestamptz not null,
    window_start    timestamptz,          -- primer instante cuyos rezagos entran
    window_end      timestamptz,          -- último origen de entrenamiento (objetivo <= corte)
    n_origins       integer     not null, -- instantes de entrenamiento
    n_rows          integer     not null, -- ejemplos = instantes x estaciones
    algorithm       text        not null, -- ridge · gbm · persistencia (sin datos suficientes)
    params          jsonb,
    train_mae_rel   double precision,     -- MAE dentro de la muestra, en unidades de nivel
    fitted_at       timestamptz not null default now(),
    primary key (cycle_id, model_version, horizon)
);

comment on table public.ajustes_ciclo is
    'Reentrenamiento en línea: cada modelo que la capa AR ajusta al entregar un ciclo, con su ventana de datos.';

alter table public.ajustes_ciclo enable row level security;

-- Mismo criterio que el resto del esquema: anon lee, service_role escribe.
drop policy if exists ajustes_ciclo_lectura on public.ajustes_ciclo;
create policy ajustes_ciclo_lectura
    on public.ajustes_ciclo for select to anon using (true);

create index if not exists ajustes_ciclo_cutoff_idx
    on public.ajustes_ciclo (data_cutoff desc);
