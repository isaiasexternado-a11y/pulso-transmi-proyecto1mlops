-- Último estado conocido del reloj oficial de la API.
--
-- El dashboard no puede preguntarle a la API (exige PULSO_API_KEY), así que no
-- sabe distinguir "no entregamos porque algo se rompió" de "no entregamos
-- porque la API está en pausa y no hay ciclos". El 2026-09-28 el reloj quedó en
-- `state=waiting` y el panel gritaba "Pipeline detenido" sin serlo.
--
-- Una sola fila (`reloj = 'oficial'`) que reescriben quienes ya hablan con la
-- API: `pipeline/entregar.py` en cada vuelta de `predict` (~2,5 min) y
-- `pipeline/evaluar.py`. `consultado_at` es la hora en que NOSOTROS miramos: si
-- envejece, el operador dejó de mirar y el panel no puede dar la pausa por
-- buena.
create table if not exists public.api_clock (
    reloj          text        primary key default 'oficial',
    state          text        not null,             -- running · waiting · lo que publique la API
    code           text,                              -- p. ej. official-20260921; ausente en pausa
    virtual_now    timestamptz,
    tick_number    integer,
    last_tick_at   timestamptz,
    server_time    timestamptz,
    consultado_at  timestamptz not null default now(),
    cambio_at      timestamptz not null default now() -- desde cuándo está en este `state`
);

comment on table public.api_clock is
    'Último estado del reloj de la API visto por el operador. Una fila.';

alter table public.api_clock enable row level security;

-- Mismo criterio que el resto del esquema: anon lee, service_role escribe.
drop policy if exists api_clock_lectura on public.api_clock;
create policy api_clock_lectura
    on public.api_clock for select to anon using (true);
