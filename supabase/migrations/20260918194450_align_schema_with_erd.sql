-- 20260918194450_align_schema_with_erd
-- Alinea el esquema con el ERD: dominios, NOT NULL, defaults, cascades y vistas.

-- 1. Datos: la fila semilla debe caber en el dominio antes de activar los CHECK
update pipeline_runs
set trigger = 'manual',
    git_commit = 'd113f68b892d7e03939428585b7fcc64534322bb'
where trigger = 'manual_seed';

-- 2. run_id vuelve a ser asignado externamente (GITHUB_RUN_ID)
alter table pipeline_runs alter column run_id drop identity if exists;

-- 3. stations
alter table stations
  alter column station_name set not null,
  alter column corridor     set not null,
  alter column latitude     set not null,
  alter column longitude    set not null;
alter table stations drop column is_portal;
alter table stations add column is_portal boolean
  generated always as (station_name like 'Portal%') stored;
alter table stations add column seeded_at timestamptz not null default now();

-- 4. pipeline_runs
alter table pipeline_runs
  alter column attempt       set not null,
  alter column attempt       set default 1,
  alter column trigger       set not null,
  alter column git_commit    set not null,
  alter column started_at    set not null,
  alter column started_at    set default now(),
  alter column status        set not null,
  alter column status        set default 'running',
  alter column rows_ingested set not null,
  alter column rows_ingested set default 0;
alter table pipeline_runs
  add constraint pipeline_runs_trigger_check
    check (trigger in ('schedule','manual','retry')),
  add constraint pipeline_runs_status_check
    check (status in ('running','success','failed','partial'));

-- 5. observations
alter table observations
  alter column demand      set not null,
  alter column run_id      set not null,
  alter column ingested_at set not null,
  alter column ingested_at set default now();
alter table observations
  add constraint observations_demand_check check (demand >= 0);

-- 6. context
alter table context
  alter column run_id      set not null,
  alter column ingested_at set not null,
  alter column ingested_at set default now();

-- 7. run_events
alter table run_events
  alter column run_id  set not null,
  alter column stage   set not null,
  alter column level   set not null,
  alter column message set not null,
  alter column at      set not null,
  alter column at      set default now();
alter table run_events
  add constraint run_events_stage_check check (stage in
    ('ingest','features','monitor','decide','train','predict','submit')),
  add constraint run_events_level_check check (level in ('info','warning','error'));
alter table run_events drop constraint run_events_run_id_fkey;
alter table run_events add constraint run_events_run_id_fkey
  foreign key (run_id) references pipeline_runs(run_id) on delete cascade;

-- 8. models
alter table models
  alter column name        set not null,
  alter column kind        set not null,
  alter column algorithm   set not null,
  alter column hyperparams set not null,
  alter column hyperparams set default '{}'::jsonb,
  alter column feature_set set not null,
  alter column feature_list set not null,
  alter column git_commit  set not null,
  alter column status      set not null,
  alter column status      set default 'candidate',
  alter column created_at  set not null;
alter table models
  add constraint models_kind_check check (kind in ('baseline','ml')),
  add constraint models_status_check
    check (status in ('candidate','active','retired','rejected'));

-- 9. model_metrics: fuera la PK sintética, entra la clave natural del ERD
alter table model_metrics drop column id;
alter table model_metrics
  alter column model_id    set not null,
  alter column split       set not null,
  alter column fold        set not null,
  alter column fold        set default 0,
  alter column metric      set not null,
  alter column value       set not null,
  alter column computed_at set not null,
  alter column computed_at set default now();
alter table model_metrics
  add constraint model_metrics_split_check
    check (split in ('train','validation','backtest','production')),
  add constraint model_metrics_metric_check
    check (metric in ('wape','accuracy','mae','rmse','bias')),
  add constraint model_metrics_natural_key
    unique nulls not distinct (model_id, split, fold, station_id, metric);
alter table model_metrics drop constraint model_metrics_model_id_fkey;
alter table model_metrics add constraint model_metrics_model_id_fkey
  foreign key (model_id) references models(model_id) on delete cascade;

-- 10. drift_signals
alter table drift_signals
  alter column value       set not null,
  alter column threshold   set not null,
  alter column breached    set not null,
  alter column computed_at set not null,
  alter column computed_at set default now();
alter table drift_signals
  add constraint drift_signals_signal_check check (signal in
    ('wape_24h','wape_7d','level_shift_7d','profile_corr','residual_bias','ingest_gap'));
alter table drift_signals drop constraint drift_signals_run_id_station_id_signal_key;
alter table drift_signals add constraint drift_signals_natural_key
  unique nulls not distinct (run_id, station_id, signal);
alter table drift_signals drop constraint drift_signals_run_id_fkey;
alter table drift_signals add constraint drift_signals_run_id_fkey
  foreign key (run_id) references pipeline_runs(run_id) on delete cascade;

-- 11. retrain_decisions
alter table retrain_decisions
  alter column decision         set not null,
  alter column reason           set not null,
  alter column breached_signals set not null,
  alter column breached_signals set default '{}'::text[],
  alter column decided_at       set not null,
  alter column decided_at       set default now();
alter table retrain_decisions
  add constraint retrain_decisions_decision_check
    check (decision in ('keep','retrain','rollback','blocked'));
alter table retrain_decisions drop constraint retrain_decisions_run_id_fkey;
alter table retrain_decisions add constraint retrain_decisions_run_id_fkey
  foreign key (run_id) references pipeline_runs(run_id) on delete cascade;

-- 12. predictions
alter table predictions
  alter column model_id   set not null,
  alter column issued_at  set not null,
  alter column horizon    set not null,
  alter column y_pred     set not null,
  alter column submitted  set not null,
  alter column submitted  set default false,
  alter column created_at set not null;
alter table predictions
  add constraint predictions_horizon_check check (horizon between 1 and 4),
  add constraint predictions_y_pred_check check (y_pred >= 0),
  add constraint predictions_horizon_coherente
    check (target_at = issued_at + (horizon * interval '15 minutes'));

-- 13. Vistas: el scoring es derivado, nunca una columna
create view v_prediction_scores with (security_invoker = on) as
select p.run_id, p.model_id, p.station_id, s.corridor,
       p.issued_at, p.target_at, p.horizon, p.y_pred,
       o.demand as y_true,
       abs(o.demand - p.y_pred) as abs_error
from predictions p
join observations o
  on o.station_id = p.station_id and o.observed_at = p.target_at
join stations s on s.station_id = p.station_id;

create view v_accuracy_rolling with (security_invoker = on) as
select station_id, model_id,
       100 * greatest(0, 1 - sum(abs_error) / nullif(sum(y_true), 0)) as accuracy_24h,
       count(*) as n
from v_prediction_scores
where target_at >= now() - interval '24 hours'
group by station_id, model_id;

create view v_active_model with (security_invoker = on) as
select * from models where status = 'active';
