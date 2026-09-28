"""Genera los íconos PNG de la PWA del dashboard a partir del logo del header.

El logo es el mismo SVG de `dashboard/index.html` (cuadro azul con el pulso en
blanco, lienzo de 34×34). Se rasteriza a mano con distancias con signo para no
depender de PIL ni de rsvg: sólo biblioteca estándar.

    python3 scripts/iconos_dashboard.py
"""
import math
import struct
import zlib
from pathlib import Path

AZUL = (0x2A, 0x78, 0xD6)
BLANCO = (0xFF, 0xFF, 0xFF)
# El trazo del header: M6 18 h5 l3-7 5 13 3-8 h6, grosor 2,2.
PULSO = [(6, 18), (11, 18), (14, 11), (19, 24), (22, 16), (28, 16)]
GROSOR = 2.2
LADO = 34

SALIDA = Path(__file__).resolve().parent.parent / "dashboard" / "iconos"


def dist_segmento(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_caja_redondeada(px, py, lado, radio):
    # Distancia con signo a un cuadrado [0, lado]² de esquinas redondeadas.
    c = lado / 2
    qx = abs(px - c) - (c - radio)
    qy = abs(py - c) - (c - radio)
    fuera = math.hypot(max(qx, 0), max(qy, 0))
    return fuera + min(max(qx, qy), 0) - radio


def cobertura(d, escala):
    # d en unidades del logo; medio píxel de suavizado a cada lado del borde.
    return max(0.0, min(1.0, 0.5 - d * escala))


def dibujar(tam, *, esquinas, zona):
    """`esquinas`: radio del fondo en unidades del logo (0 = sangrado completo).
    `zona`: fracción del ícono que ocupa el logo (maskable exige ≤ 0,8)."""
    escala = tam / LADO                    # píxeles por unidad del fondo
    esc_logo = zona                        # el pulso se encoge hacia el centro
    filas = []
    for y in range(tam):
        fila = bytearray([0])              # filtro PNG "none"
        for x in range(tam):
            ux, uy = (x + 0.5) / escala, (y + 0.5) / escala
            fondo = 1.0 if esquinas == 0 else cobertura(dist_caja_redondeada(ux, uy, LADO, esquinas), escala)
            lx = LADO / 2 + (ux - LADO / 2) / esc_logo
            ly = LADO / 2 + (uy - LADO / 2) / esc_logo
            d = min(dist_segmento(lx, ly, PULSO[i], PULSO[i + 1]) for i in range(len(PULSO) - 1))
            trazo = cobertura((d - GROSOR / 2) * esc_logo, escala) * fondo
            color = [round(AZUL[k] + (BLANCO[k] - AZUL[k]) * trazo) for k in range(3)]
            fila += bytes(color + [round(255 * fondo)])
        filas.append(bytes(fila))
    return png(tam, b"".join(filas))


def png(tam, crudo):
    def trozo(tipo, datos):
        return (struct.pack(">I", len(datos)) + tipo + datos
                + struct.pack(">I", zlib.crc32(tipo + datos) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + trozo(b"IHDR", struct.pack(">IIBBBBB", tam, tam, 8, 6, 0, 0, 0))
            + trozo(b"IDAT", zlib.compress(crudo, 9))
            + trozo(b"IEND", b""))


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    iconos = {
        # "any": el logo tal cual, con sus esquinas.
        "icono-192.png": (192, dict(esquinas=9, zona=1.0)),
        "icono-512.png": (512, dict(esquinas=9, zona=1.0)),
        # maskable: fondo a sangre y el pulso dentro de la zona segura; el
        # sistema recorta la forma (círculo, gota, cuadrado) que quiera.
        "maskable-512.png": (512, dict(esquinas=0, zona=0.72)),
        # iOS redondea solo y no admite transparencia: fondo a sangre.
        "apple-touch-icon.png": (180, dict(esquinas=0, zona=0.85)),
        "favicon-32.png": (32, dict(esquinas=9, zona=1.0)),
    }
    for nombre, (tam, opciones) in iconos.items():
        (SALIDA / nombre).write_bytes(dibujar(tam, **opciones))
        print(f"{nombre}: {tam}×{tam}")
