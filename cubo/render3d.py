"""Réplica 3D do cubo desenhada com OpenCV, com a animação do próximo giro."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .modelo import Cubo, NORMAIS
from .visao import BGR

CORPO = (25, 25, 25)


def _rot_eixo(n, ang):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = n
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _vista(yaw=-32, pitch=26):
    a, b = math.radians(yaw), math.radians(pitch)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rx @ ry


VISTA = _vista()


def _giro(mov: str):
    """Eixo, camada e ângulo total de um movimento (horário olhando a face = -90° no eixo da normal)."""
    base, suf = mov[0], mov[1:]
    q = {"": 1, "'": -1, "2": 2}[suf]
    if base in "xyz":
        n = NORMAIS[{"x": "R", "y": "U", "z": "F"}[base]]
        return np.array(n, float), None, -q * math.pi / 2
    n = NORMAIS[base]
    return np.array(n, float), n, -q * math.pi / 2


def desenhar(cubo: Cubo, mov: str | None, fase: float, tam: int = 380, fundo=(30, 30, 30),
             rotulos: bool = False) -> np.ndarray:
    """Imagem tam x tam do cubo; se mov, a camada que gira está a `fase` (0..1) do caminho."""
    img = np.full((tam, tam, 3), fundo, np.uint8)
    eixo = camada = None
    ang = 0.0
    if mov:
        eixo, camada, total = _giro(mov)
        ang = total * fase
    rot_camada = _rot_eixo(eixo, ang) if mov else np.eye(3)

    def gira(p, pos):
        if mov and (camada is None or np.dot(pos, camada) == 1):
            return rot_camada @ p
        return p

    f, dist = tam * 1.35, 9.0
    centro = np.array([tam / 2, tam / 2 + tam * 0.03])

    def projeta(p):
        v = VISTA @ p
        k = f / (dist - v[2])
        return centro + np.array([v[0] * k, -v[1] * k]), v[2]

    polis = []
    pecas = cubo.pecas()
    for pos in [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]:
        if pos == (0, 0, 0):
            continue
        pv = np.array(pos, float)
        for n in [np.array(v, float) for v in NORMAIS.values()]:
            u = np.array([n[1], n[2], n[0]])
            w = np.cross(n, u)
            cfa = pv + n * 0.5
            canto = [cfa + (a * u + b * w) * 0.5 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            nr = VISTA @ gira(n, pos)
            if nr[2] <= 0.02:
                continue
            pts3 = [gira(c, pos) for c in canto]
            prof = np.mean([(VISTA @ p)[2] for p in pts3])
            polis.append((prof, [projeta(p)[0] for p in pts3], CORPO))
            cor = pecas.get(pos, {}).get(tuple(int(x) for x in n))
            if cor:
                adesivo = [cfa + n * 0.01 + (a * u + b * w) * 0.43 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                pts3 = [gira(c, pos) for c in adesivo]
                polis.append((prof + 0.001, [projeta(p)[0] for p in pts3], BGR[cor]))
    for _, pts, cor in sorted(polis, key=lambda x: x[0]):
        cv2.fillConvexPoly(img, np.round(np.array(pts)).astype(np.int32), cor, cv2.LINE_AA)
    if mov:
        _seta(img, eixo, camada, _giro(mov)[2], projeta)
    if rotulos:
        _rotulo(img, projeta(np.array([0, 2.25, 0]))[0], "PRA CIMA")
        _rotulo(img, projeta(np.array([-0.2, -2.2, 1.6]))[0], "PRA VOCE")
    return img


def _rotulo(img, p, texto):
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    x, y = int(p[0] - w / 2), int(p[1] + h / 2)
    cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + 6), (60, 60, 60), -1)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _seta(img, eixo, camada, total, projeta):
    """Seta curva em volta da camada que gira, no sentido do giro."""
    n = eixo / np.linalg.norm(eixo)
    altura = 1.0 if camada is not None else 0.0
    # começa no lado da camada que está mais virado para quem olha
    perp = [np.array(v, float) for v in NORMAIS.values() if abs(np.dot(v, n)) < 0.5]
    frente = max(perp, key=lambda v: (VISTA @ v)[2] + 0.3 * (VISTA @ v)[1])
    raio = 2.05
    sentido = 1 if total > 0 else -1
    span = math.radians(150 if abs(total) > 2 else 85)
    pts = []
    for t in np.linspace(-span / 2, span / 2, 30):
        p = n * altura + _rot_eixo(n, sentido * t) @ (frente * raio)
        pts.append(projeta(p)[0])
    pts = np.round(np.array(pts)).astype(np.int32)
    cv2.polylines(img, [pts], False, (255, 255, 255), 9, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (0, 170, 255), 5, cv2.LINE_AA)
    a, b = pts[-4].astype(float), pts[-1].astype(float)
    d = (b - a) / (np.linalg.norm(b - a) + 1e-6)
    o = np.array([-d[1], d[0]])
    ponta = np.array([b + d * 14, b + o * 11, b - o * 11]).astype(np.int32)
    cv2.fillConvexPoly(img, ponta, (0, 170, 255), cv2.LINE_AA)
    cv2.polylines(img, [ponta], True, (255, 255, 255), 2, cv2.LINE_AA)
