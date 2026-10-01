-- Vigilancia de la tipología de los datos.
--
-- El 2026-10-01 el profesor anunció que va a cambiar la "tipología de los
-- datos". Puede ser el formato (campos, tipos, estaciones, horizontes que
-- entrega la API) o el comportamiento (otro tipo de drift). `pipeline/vigilar.py`
-- revisa las dos cosas en cada corrida de evaluate.yml y deja aquí lo que ve:
--
--   kind = 'formato'  huella del contrato de la API (meta, estaciones, ciclo,
--                     filas del stream). La primera huella queda como 'base';
--                     una distinta es 'alerta'.
--   kind = 'profe'    huella de los documentos del repo del profesor
--                     (revisiones del drift, contrato de la API).
--   kind = 'patron'   quiebre del comportamiento: caída brusca de accuracy
--                     o estaciones que se apagan o se disparan.
--
-- Una misma huella no se duplica: se actualiza `last_seen_at`. El dashboard
-- muestra un banner con las alertas recientes y el vigía en la nube manda
-- correo. Aceptar un formato nuevo como normal: `python -m pipeline.vigilar --aceptar`.
create table if not exists public.data_watch (
    kind          text        not null check (kind in ('formato', 'profe', 'patron')),
    fingerprint   text        not null,
    status        text        not null check (status in ('base', 'alerta', 'aceptada')),
    title         text        not null,
    detail        jsonb,
    first_seen_at timestamptz not null default now(),
    last_seen_at  timestamptz not null default now(),
    primary key (kind, fingerprint)
);

comment on table public.data_watch is
    'Huellas del formato de la API, de los documentos del profesor y quiebres de patrón. Alimenta el banner del dashboard.';

alter table public.data_watch enable row level security;

-- Mismo criterio que el resto del esquema: anon lee, service_role escribe.
drop policy if exists data_watch_lectura on public.data_watch;
create policy data_watch_lectura
    on public.data_watch for select to anon using (true);

create index if not exists data_watch_last_seen_idx
    on public.data_watch (last_seen_at desc);
