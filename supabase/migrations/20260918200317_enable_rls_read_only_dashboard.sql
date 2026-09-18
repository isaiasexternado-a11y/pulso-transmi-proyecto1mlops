-- 20260918200317_enable_rls_read_only_dashboard
-- RLS en las 10 tablas.
-- Modelo: el dashboard (anon) solo lee; el pipeline escribe con service_role,
-- que ignora RLS por diseño en Supabase. No se crea ninguna política de escritura.

alter table stations          enable row level security;
alter table pipeline_runs     enable row level security;
alter table observations      enable row level security;
alter table context           enable row level security;
alter table run_events        enable row level security;
alter table models            enable row level security;
alter table model_metrics     enable row level security;
alter table drift_signals     enable row level security;
alter table retrain_decisions enable row level security;
alter table predictions       enable row level security;

-- Lectura pública para lo que el dashboard necesita mostrar
create policy "lectura publica" on stations
  for select to anon, authenticated using (true);
create policy "lectura publica" on pipeline_runs
  for select to anon, authenticated using (true);
create policy "lectura publica" on observations
  for select to anon, authenticated using (true);
create policy "lectura publica" on context
  for select to anon, authenticated using (true);
create policy "lectura publica" on models
  for select to anon, authenticated using (true);
create policy "lectura publica" on model_metrics
  for select to anon, authenticated using (true);
create policy "lectura publica" on drift_signals
  for select to anon, authenticated using (true);
create policy "lectura publica" on retrain_decisions
  for select to anon, authenticated using (true);
create policy "lectura publica" on predictions
  for select to anon, authenticated using (true);

-- run_events son logs de operación: sin política, solo service_role los ve.

-- Nota: este revoke por columna NO surte efecto mientras exista el SELECT a nivel
-- de tabla. Se corrige en 20260918200353_restrict_submit_response_column.
revoke select (submit_response) on predictions from anon, authenticated;
