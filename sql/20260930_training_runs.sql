-- Una fila por cada corrida de train.yml, haya candidato o no.
--
-- `ml/entrenar.py` sólo escribe en `pipeline_runs`/`models` cuando registra un
-- candidato. Una corrida que entrena, no encuentra ganador y termina en verde
-- no dejaba rastro, así que "no hubo ganador" y "no se entrenó" se veían
-- igual. Desde el 30-sep la cadencia de 2 h la lleva evaluate.yml porque el
-- cron de GitHub se saltaba corridas; esta tabla es la que permite comprobar
-- que efectivamente se entrena, y que el dashboard lo diga sin abrir Actions.
--
-- La escribe el propio workflow (no el código de ML): se abre al arrancar y se
-- cierra en un job con `if: always()`, así que queda aunque entrenar falle.
-- Una fila que se queda en `running` más de una hora es un runner que murió.
create table if not exists public.training_runs (
    gh_run_id          bigint      not null,          -- GITHUB_RUN_ID
    attempt            smallint    not null default 1, -- GITHUB_RUN_ATTEMPT: re-run no pisa
    event              text        not null,          -- schedule · workflow_dispatch
    started_at         timestamptz not null,
    finished_at        timestamptz,
    status             text        not null
        check (status in ('running', 'success', 'failed', 'cancelled')),
    candidate_model_id uuid references public.models (model_id),
    promoted           boolean,                       -- null: no hubo candidato que promover
    ml_commit          text,                          -- commit de experimentos-ml que entrenó
    run_url            text,
    primary key (gh_run_id, attempt)
);

comment on table public.training_runs is
    'Cada corrida de train.yml con su desenlace. Sirve para verificar la cadencia de reentreno.';

alter table public.training_runs enable row level security;

-- Mismo criterio que el resto del esquema: anon lee, service_role escribe.
drop policy if exists training_runs_lectura on public.training_runs;
create policy training_runs_lectura
    on public.training_runs for select to anon using (true);

create index if not exists training_runs_started_idx
    on public.training_runs (started_at desc);
