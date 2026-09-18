import csv, json, os, sys, urllib.request, urllib.error
from datetime import datetime, timezone

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_KEY"]
DATA = os.environ.get("PULSO_DATA_DIR", "pulso-transmi-sdk/data")

def post(table, rows, upsert=True, return_rep=False):
    body = json.dumps(rows).encode()
    prefer = []
    if upsert:
        prefer.append("resolution=merge-duplicates")
    prefer.append("return=representation" if return_rep else "return=minimal")
    req = urllib.request.Request(
        f"{URL}/rest/v1/{table}",
        data=body,
        method="POST",
        headers={
            "apikey": KEY,
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Prefer": ",".join(prefer),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return json.loads(raw) if return_rep and raw else None
    except urllib.error.HTTPError as e:
        print(f"ERROR {table}: {e.code} {e.read().decode()[:500]}", file=sys.stderr)
        raise

def batched(rows, n):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]

now = datetime.now(timezone.utc).isoformat()
meta = json.load(open(f"{DATA}/metadata.json"))

run = post("pipeline_runs", [{
    "attempt": 1,
    "trigger": "manual_seed",
    "git_ref": "refs/heads/main",
    "started_at": now,
    "status": "running",
    "cutoff_at": meta["history_end"],
}], upsert=False, return_rep=True)
run_id = run[0]["run_id"]
print(f"pipeline_run creado: run_id={run_id}")

with open(f"{DATA}/stations.csv") as f:
    stations = [{
        "station_id": r["station_id"],
        "station_name": r["station_name"],
        "corridor": r["corridor"],
        "latitude": float(r["latitude"]),
        "longitude": float(r["longitude"]),
        "is_portal": r["station_name"].lower().startswith("portal"),
    } for r in csv.DictReader(f)]
post("stations", stations)
print(f"stations: {len(stations)} filas enviadas")

with open(f"{DATA}/observations.csv") as f:
    obs = [{
        "station_id": r["station_id"],
        "observed_at": r["observed_at"],
        "demand": int(r["demand"]),
        "run_id": run_id,
        "ingested_at": now,
    } for r in csv.DictReader(f)]
sent = 0
for chunk in batched(obs, 2000):
    post("observations", chunk)
    sent += len(chunk)
    print(f"  observations: {sent}/{len(obs)}", flush=True)

def fnum(v):
    return float(v) if v not in ("", None) else None

with open(f"{DATA}/context.csv") as f:
    ctx = [{
        "observed_at": r["observed_at"],
        "rain_mm": fnum(r["rain_mm"]),
        "rain_forecast": fnum(r["rain_forecast"]),
        "temperature_c": fnum(r["temperature_c"]),
        "temperature_forecast": fnum(r["temperature_forecast"]),
        "event_intensity": fnum(r["event_intensity"]),
        "run_id": run_id,
        "ingested_at": now,
    } for r in csv.DictReader(f)]
sent = 0
for chunk in batched(ctx, 2000):
    post("context", chunk)
    sent += len(chunk)
    print(f"  context: {sent}/{len(ctx)}", flush=True)

post("pipeline_runs", [{
    "run_id": run_id,
    "attempt": 1,
    "trigger": "manual_seed",
    "git_ref": "refs/heads/main",
    "started_at": now,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "status": "success",
    "cutoff_at": meta["history_end"],
    "rows_ingested": len(obs) + len(ctx) + len(stations),
}])
print(f"OK run_id={run_id} stations={len(stations)} observations={len(obs)} context={len(ctx)}")
