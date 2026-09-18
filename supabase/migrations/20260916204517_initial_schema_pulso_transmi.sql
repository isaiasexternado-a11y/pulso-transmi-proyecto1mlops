-- 20260916204517_initial_schema_pulso_transmi
-- Esqueleto inicial: las 10 tablas del ERD con sus PK, FK e índices.
-- Las reglas de negocio (CHECK, NOT NULL, defaults, cascades y vistas)
-- llegan en la migración 20260918194450_align_schema_with_erd.

create extension if not exists pgcrypto;

create table stations (
  station_id   text primary key,
  station_name text not null,
  corridor     text,
  latitude     double precision,
  longitude    double precision,
  vagones      smallint,
  accesos      smallint,
  tipo         smallint,
  is_portal    boolean
);

create table pipeline_runs (
  run_id         bigint generated always as identity primary key,
  attempt        smallint,
  trigger        text,
  git_commit     text,
  git_ref        text,
  run_url        text,
  started_at     timestamptz,
  finished_at    timestamptz,
  status         text,
  cutoff_at      timestamptz,
  rows_ingested  integer,
  error_stage    text,
  error_message  text
);

create table observations (
  station_id   text not null references stations(station_id),
  observed_at  timestamptz not null,
  demand       integer,
  run_id       bigint references pipeline_runs(run_id),
  ingested_at  timestamptz,
  primary key (station_id, observed_at)
);

create table context (
  observed_at            timestamptz primary key,
  rain_mm                double precision,
  rain_forecast          double precision,
  temperature_c          double precision,
  temperature_forecast   double precision,
  event_intensity        double precision,
  run_id                 bigint references pipeline_runs(run_id),
  ingested_at            timestamptz
);

create table run_events (
  event_id  bigserial primary key,
  run_id    bigint references pipeline_runs(run_id),
  stage     text,
  level     text,
  message   text,
  payload   jsonb,
  at        timestamptz
);

create table models (
  model_id           uuid primary key default gen_random_uuid(),
  name               text,
  kind               text,
  algorithm          text,
  hyperparams        jsonb,
  feature_set        text,
  feature_list       text[],
  train_start        timestamptz,
  train_end          timestamptz,
  n_train_rows       integer,
  artifact_uri       text,
  artifact_sha256    text,
  git_commit         text,
  trained_by_run_id  bigint references pipeline_runs(run_id),
  parent_model_id    uuid references models(model_id),
  status             text,
  activated_at       timestamptz,
  retired_at         timestamptz,
  created_at         timestamptz default now()
);

create table drift_signals (
  run_id       bigint not null references pipeline_runs(run_id),
  station_id   text references stations(station_id),
  signal       text not null,
  value        double precision,
  threshold    double precision,
  breached     boolean,
  computed_at  timestamptz,
  unique (run_id, station_id, signal)
);

create table retrain_decisions (
  run_id               bigint primary key references pipeline_runs(run_id),
  decision             text,
  reason               text,
  breached_signals     text[],
  incumbent_model_id   uuid references models(model_id),
  new_model_id         uuid references models(model_id),
  cooldown_until       timestamptz,
  decided_at           timestamptz
);

create table model_metrics (
  id            bigserial primary key,
  model_id      uuid not null references models(model_id),
  split         text,
  fold          smallint,
  station_id    text references stations(station_id),
  metric        text,
  value         double precision,
  window_start  timestamptz,
  window_end    timestamptz,
  computed_at   timestamptz
);

create table predictions (
  run_id            bigint not null references pipeline_runs(run_id),
  station_id        text not null references stations(station_id),
  target_at         timestamptz not null,
  model_id          uuid references models(model_id),
  issued_at         timestamptz,
  horizon           smallint,
  y_pred            double precision,
  submitted         boolean,
  submit_response   jsonb,
  created_at        timestamptz default now(),
  primary key (run_id, station_id, target_at)
);

create index on observations (run_id);
create index on context (run_id);
create index on run_events (run_id);
create index on drift_signals (run_id);
create index on drift_signals (station_id);
create index on models (trained_by_run_id);
create index on models (parent_model_id);
create index on model_metrics (model_id);
create index on model_metrics (station_id);
create index on predictions (model_id);
create index on predictions (station_id);
