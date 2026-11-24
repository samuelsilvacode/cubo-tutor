"""Acha os 9 adesivos de uma face em qualquer lugar da imagem."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    centros: np.ndarray  # 9 x 2, ordem de leitura (de pé)
    lado: float
    angulo: float

    def amostras(self, frame) -> list[np.ndarray]:
        r = max(2, int(self.lado * 0.22))
        h, w = frame.shape[:2]
        out = []
        for x, y in self.centros.astype(int):
            pedaco = frame[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)].reshape(-1, 3)
            out.append(np.median(pedaco, axis=0).astype(np.uint8) if len(pedaco) else np.zeros(3, np.uint8))
        return out


def _quadrados(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(cinza, 20, 60)
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8))
    conts, _ = cv2.findContours(255 - bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = frame.shape[0] * frame.shape[1]
    out = []
    for c in conts:
        area = cv2.contourArea(c)
        if not area_img * 0.0003 < area < area_img * 0.05:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) == 0 or max(w, h) / min(w, h) > 1.4:
            continue
        if area / (w * h) < 0.75:
            continue
        out.append((cx, cy, (w + h) / 2, ang))
    return out


def achar_face(frame) -> Face | None:
    qs = _quadrados(frame)
    if len(qs) < 6:
        return None
    pts = np.array([(q[0], q[1]) for q in qs])
    lados = np.array([q[2] for q in qs])
    melhor = None
    for i, (cx, cy, lado, ang) in enumerate(qs):
        parecidos = np.abs(lados - lado) < lado * 0.3
        viz = pts[parecidos]
        if len(viz) < 6:
            continue
        # passo da grade: distância até o vizinho mais próximo
        d = np.linalg.norm(viz - (cx, cy), axis=1)
        d = d[d > lado * 0.5]
        if not len(d):
            continue
        passo = d.min()
        if passo > lado * 1.6:
            continue
        t = np.deg2rad(ang)
        ex = np.array([np.cos(t), np.sin(t)])
        ey = np.array([-np.sin(t), np.cos(t)])
        rel = viz - (cx, cy)
        u, v = rel @ ex / passo, rel @ ey / passo
        achados = {}
        for a, b, p in zip(u, v, viz):
            ia, ib = round(a), round(b)
            if abs(ia) <= 1 and abs(ib) <= 1 and abs(a - ia) < 0.25 and abs(b - ib) < 0.25:
                achados[(ia, ib)] = p
        if len(achados) >= 7 and (0, 0) in achados and (melhor is None or len(achados) > melhor[0]):
            melhor = (len(achados), cx, cy, ex, ey, passo, lado, ang, achados)
    if melhor is None:
        return None
    _, cx, cy, ex, ey, passo, lado, ang, achados = melhor
    # deixa "de pé": o eixo y da grade é o que mais aponta para baixo na imagem
    eixos = [(ex, ey), (ey, -ex), (-ex, -ey), (-ey, ex)]
    ex, ey = max(eixos, key=lambda e: e[1][1])
    centros = []
    for lin in (-1, 0, 1):
        for col in (-1, 0, 1):
            centros.append(np.array((cx, cy)) + (col * ex + lin * ey) * passo)
    ang_pe = np.degrees(np.arctan2(ex[1], ex[0]))
    return Face(np.array(centros), lado, ang_pe)


def nitidez(frame, face: Face) -> float:
    x0, y0 = face.centros.min(axis=0).astype(int)
    x1, y1 = face.centros.max(axis=0).astype(int)
    m = int(face.lado)
    roi = frame[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    if roi.size == 0:
        return 0
    return float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
