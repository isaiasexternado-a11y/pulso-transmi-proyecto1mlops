-- Cursor del stream incremental de la API.
-- Vive en la base, no en el disco del runner: un runner nuevo cada ciclo
-- no puede recordar dónde quedó. Se avanza sólo después de confirmar el
-- upsert de las filas.
create table if not exists public.stream_cursor (
  stream            text primary key,
  cursor            text        not null,
  last_released_at  timestamptz,
  rows_total        bigint      not null default 0,
  updated_at        timestamptz not null default now()
);

alter table public.stream_cursor enable row level security;

drop policy if exists stream_cursor_lectura_anon on public.stream_cursor;
create policy stream_cursor_lectura_anon
  on public.stream_cursor for select to anon using (true);
