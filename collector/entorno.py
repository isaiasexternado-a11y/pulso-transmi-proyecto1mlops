"""Credenciales y acceso REST a Supabase.

El collector corre en un runner efímero: no hay disco que recordar ni sesión
que reusar. Todo lo que necesita saber llega por variables de entorno, y todo
lo que aprende lo deja en la base.

Local   : las variables salen de .env (que está en .gitignore).
Actions : salen de los Secrets del repositorio. El archivo .env no existe allá
          y su ausencia no es un error.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RUTA_ENV = Path(".env")

# Un timeout suelto no puede matar una corrida. El entrenamiento pagina más de
# cincuenta peticiones para traer el histórico completo, y basta que una se
# demore para tirar todo abajo: ya pasó. Los reintentos son cortos y pocos —no
# están para disimular que Supabase se cayó, sino para absorber el hipo de red
# de un runner efímero.
REINTENTOS = 3
ESPERA_BASE = 2.0


def _transitorio(e: Exception) -> bool:
    """¿Vale la pena reintentar esto, o es un error de verdad?

    Un 404 o un 400 van a fallar igual la segunda vez. Un timeout o un 5xx, no.
    """
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (429, 500, 502, 503, 504)
    return isinstance(e, (urllib.error.URLError, TimeoutError, OSError))


def cargar_env() -> None:
    """Vuelca .env al entorno sin pisar lo que ya venga de afuera."""
    if not RUTA_ENV.exists():
        return
    for linea in RUTA_ENV.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip())


def exigir(*nombres: str) -> tuple[str, ...]:
    cargar_env()
    faltan = [n for n in nombres if not os.environ.get(n)]
    if faltan:
        raise SystemExit(
            "faltan variables de entorno: " + ", ".join(faltan) +
            "\n  local  -> agrégalas a .env"
            "\n  Actions-> agrégalas a los Secrets del repositorio"
        )
    return tuple(os.environ[n].rstrip("/") if n.endswith("_URL") or n.endswith("_BASE")
                 else os.environ[n] for n in nombres)


class Supabase:
    """Cliente mínimo sobre PostgREST. Sólo lo que el pipeline usa."""

    def __init__(self) -> None:
        self.url, self.key = exigir("SUPABASE_URL", "SUPABASE_SECRET_KEY")

    def _reintentar(self, describir, hacer):
        """Ejecuta `hacer`, reintentando sólo lo que tiene sentido reintentar.

        Quien llama decide si la operación es segura de repetir. No lo decide
        esta función: un GET y un upsert se pueden repetir sin consecuencias,
        un INSERT crearía una fila de más.
        """
        import sys
        for intento in range(1, REINTENTOS + 1):
            try:
                return hacer()
            except Exception as e:
                if not _transitorio(e):
                    raise
                if intento == REINTENTOS:
                    # Se agotaron los intentos. Se re-envuelve para que el
                    # mensaje diga qué operación murió y tras cuántos golpes,
                    # en vez de escupir un HTTPError pelado sin contexto.
                    detalle = ""
                    if isinstance(e, urllib.error.HTTPError):
                        try:
                            detalle = ": " + e.read().decode(errors="replace")[:300]
                        except Exception:
                            detalle = ""
                    raise RuntimeError(
                        f"Supabase {describir} falló tras {REINTENTOS} intentos "
                        f"({type(e).__name__}){detalle}") from None
                espera = ESPERA_BASE * (2 ** (intento - 1))
                print(f"[supabase] {describir} falló ({type(e).__name__}); "
                      f"reintento {intento + 1}/{REINTENTOS} en {espera:.0f}s",
                      file=sys.stderr)
                time.sleep(espera)

    def _pedir(self, metodo: str, ruta: str, params: dict | None = None,
               cuerpo=None, prefer: str | None = None,
               reintentable: bool = False) -> list | dict:
        destino = f"{self.url}/rest/v1/{ruta}"
        if params:
            destino += "?" + urllib.parse.urlencode(params)
        cabeceras = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
        }
        datos = None
        if cuerpo is not None:
            datos = json.dumps(cuerpo).encode()
            cabeceras["Content-Type"] = "application/json"
        if prefer:
            cabeceras["Prefer"] = prefer

        def una_vez():
            req = urllib.request.Request(destino, data=datos, method=metodo,
                                         headers=cabeceras)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    crudo = r.read()
                    return json.loads(crudo) if crudo else []
            except urllib.error.HTTPError as e:
                if _transitorio(e):
                    raise
                detalle = e.read().decode(errors="replace")[:500]
                raise RuntimeError(
                    f"Supabase {metodo} {ruta} -> {e.code}: {detalle}") from None

        if not reintentable:
            return una_vez()
        return self._reintentar(f"{metodo} {ruta}", una_vez)

    def seleccionar(self, tabla: str, **params) -> list:
        return self._pedir("GET", tabla, params, reintentable=True)

    def insertar(self, tabla: str, filas: list[dict], devolver: bool = False) -> list:
        """Sin reintentos, y es deliberado: un INSERT repetido crea una fila de
        más. Lo idempotente es `upsert`; si hace falta tolerancia a fallos,
        se usa ése."""
        prefer = "return=representation" if devolver else "return=minimal"
        return self._pedir("POST", tabla, cuerpo=filas, prefer=prefer)

    def upsert(self, tabla: str, filas: list[dict], conflicto: str) -> None:
        """Upsert por la llave primaria. Correrlo dos veces no duplica."""
        self._pedir("POST", tabla, params={"on_conflict": conflicto}, cuerpo=filas,
                    prefer="resolution=merge-duplicates,return=minimal",
                    reintentable=True)

    def actualizar(self, tabla: str, filtro: dict, cambios: dict) -> None:
        self._pedir("PATCH", tabla, params=filtro, cuerpo=cambios,
                    prefer="return=minimal", reintentable=True)

    def subir(self, bucket: str, ruta: str, datos: bytes) -> None:
        """Sube un objeto a Storage, pisando si ya existe.

        `x-upsert` hace la operación idempotente: reintentar una promoción a
        medias no falla por "el archivo ya está". El nombre lleva versión, así
        que pisar sólo puede ocurrir reintentando la misma versión.
        """
        def una_vez():
            req = urllib.request.Request(
                f"{self.url}/storage/v1/object/{bucket}/{ruta}",
                data=datos, method="POST",
                headers={"apikey": self.key, "Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/octet-stream",
                         "x-upsert": "true"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    r.read()
            except urllib.error.HTTPError as e:
                if _transitorio(e):
                    raise
                detalle = e.read().decode(errors="replace")[:300]
                raise RuntimeError(
                    f"Storage POST {bucket}/{ruta} -> {e.code}: {detalle}") from None

        # Reintentar es seguro justamente por `x-upsert`.
        self._reintentar(f"POST storage/{ruta}", una_vez)

    def descargar(self, bucket: str, ruta: str) -> bytes:
        """Baja un objeto de Storage.

        Storage exige la cabecera `apikey` además del Bearer: intenta decodificar
        el Authorization como JWT y las llaves nuevas (sb_secret_...) no lo son.
        """
        def una_vez():
            req = urllib.request.Request(
                f"{self.url}/storage/v1/object/{bucket}/{ruta}",
                headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if _transitorio(e):
                    raise
                detalle = e.read().decode(errors="replace")[:300]
                raise RuntimeError(
                    f"Storage GET {bucket}/{ruta} -> {e.code}: {detalle}") from None

        # Si esto falla de verdad, la entrega se queda sin champion: es el
        # lugar donde más vale la pena aguantar un hipo de red.
        return self._reintentar(f"GET storage/{ruta}", una_vez)
