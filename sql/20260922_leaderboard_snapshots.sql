-- Fotos del leaderboard a lo largo del tiempo.
--
-- El dashboard no puede consultar la API: eso exige PULSO_API_KEY, y esa llave
-- no puede vivir en variables públicas de Vercel. Así que la trae el evaluador,
-- que ya corre en Actions con la llave, y aquí queda el dato derivado. El
-- dashboard lee de Supabase, nunca de la API.
--
-- Qué se guarda y qué NO. La API devuelve el nombre y el puntaje de cada
-- compañero. Publicar esos nombres en una página web pública es una exposición
-- distinta de verlos dentro de la plataforma del curso, así que aquí sólo se
-- guarda NUESTRA fila con nombre, más estadísticas agregadas del pelotón
-- (cuántos son, cómo va el primero, la mediana). Alcanza para responder "¿cómo
-- cambia mi posición en el tiempo?", que es la pregunta del panel, sin
-- republicar datos de terceros.
create table if not exists public.leaderboard_snapshots (
    captured_at      timestamptz primary key,   -- `calculated_at` de la API: reingerir no duplica
    rank             smallint         not null,
    participantes    smallint         not null,
    accuracy         double precision not null,
    coverage         double precision not null,
    raw_wape         double precision,
    accuracy_at_20   double precision,
    lider_accuracy   double precision,
    lider_coverage   double precision,
    mediana_accuracy double precision,
    ingested_at      timestamptz      not null default now()
);

comment on table public.leaderboard_snapshots is
    'Posición propia y agregados del pelotón. Sin nombres de terceros: el dashboard es público.';

alter table public.leaderboard_snapshots enable row level security;

-- Mismo criterio que el resto del esquema: anon lee, service_role escribe.
drop policy if exists leaderboard_snapshots_lectura on public.leaderboard_snapshots;
create policy leaderboard_snapshots_lectura
    on public.leaderboard_snapshots for select to anon using (true);

create index if not exists leaderboard_snapshots_captured_idx
    on public.leaderboard_snapshots (captured_at desc);
