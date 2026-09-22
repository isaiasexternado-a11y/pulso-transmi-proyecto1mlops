# Migraciones

El esquema inicial (10 tablas y 3 vistas del ERD en `hallazgos/`) se aplicó
directamente sobre el proyecto de Supabase y no quedó versionado aquí. De este
punto en adelante, **todo cambio de esquema entra como un archivo en esta
carpeta**, nombrado `AAAAMMDD_descripcion.sql`, y se aplica en orden.

Aplicadas:

| Archivo | Qué hace |
|---|---|
| `20260921_stream_cursor.sql` | Tabla del cursor del stream incremental |
| `20260921_pipeline_runs_identity.sql` | `pipeline_runs.run_id` lo genera la base |
