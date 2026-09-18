-- 20260918200353_restrict_submit_response_column
-- El REVOKE por columna es inútil mientras exista el SELECT a nivel de tabla:
-- en Postgres el permiso de tabla cubre todas las columnas.
-- Hay que quitar el de tabla y volver a otorgar columna por columna.

revoke select on predictions from anon, authenticated;

grant select (run_id, station_id, target_at, model_id, issued_at,
              horizon, y_pred, submitted, created_at)
  on predictions to anon, authenticated;

-- Limpieza de la fila usada para probar RLS en run_events
delete from run_events where message = 'fila temporal para probar RLS';
