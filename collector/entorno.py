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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RUTA_ENV = Path(".env")


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
        self.url, self.key = exigir("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")

    def _pedir(self, metodo: str, ruta: str, params: dict | None = None,
               cuerpo=None, prefer: str | None = None) -> list | dict:
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

        req = urllib.request.Request(destino, data=datos, method=metodo,
                                     headers=cabeceras)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                crudo = r.read()
                return json.loads(crudo) if crudo else []
        except urllib.error.HTTPError as e:
            detalle = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Supabase {metodo} {ruta} -> {e.code}: {detalle}") from None

    def seleccionar(self, tabla: str, **params) -> list:
        return self._pedir("GET", tabla, params)

    def insertar(self, tabla: str, filas: list[dict], devolver: bool = False) -> list:
        prefer = "return=representation" if devolver else "return=minimal"
        return self._pedir("POST", tabla, cuerpo=filas, prefer=prefer)

    def upsert(self, tabla: str, filas: list[dict], conflicto: str) -> None:
        """Upsert por la llave primaria. Correrlo dos veces no duplica."""
        self._pedir("POST", tabla, params={"on_conflict": conflicto}, cuerpo=filas,
                    prefer="resolution=merge-duplicates,return=minimal")

    def actualizar(self, tabla: str, filtro: dict, cambios: dict) -> None:
        self._pedir("PATCH", tabla, params=filtro, cuerpo=cambios,
                    prefer="return=minimal")
