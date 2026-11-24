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
    achado: list | None = None  # quais dos 9 vieram de adesivo achado na imagem

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
    global ultimas_bordas
    ultimas_bordas = bordas
    conts, _ = cv2.findContours(255 - bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = frame.shape[0] * frame.shape[1]
    out = []
    for c in conts:
        area = cv2.contourArea(c)
        if not area_img * 0.0003 < area < area_img * 0.12:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) == 0 or area / (w * h) < 0.75:
            continue
        razao = max(w, h) / min(w, h)
        n = round(razao)
        if n > 3 or abs(razao - n) > 0.3:
            continue
        if n == 1:
            out.append((cx, cy, (w + h) / 2, ang))
            continue
        # cubo sem borda: peças da mesma cor lado a lado viram um retângulo só
        unidade = min(w, h)
        t = np.deg2rad(ang)
        eixo = np.array([np.cos(t), np.sin(t)]) if w > h else np.array([-np.sin(t), np.cos(t)])
        for k in range(n):
            px, py = np.array((cx, cy)) + eixo * unidade * (k - (n - 1) / 2)
            out.append((px, py, unidade, ang))
    return _divide_blocos(out)


def _divide_blocos(qs):
    """Quadrado com o dobro do tamanho dos outros = 4 peças iguais juntas."""
    if len(qs) < 3:
        return qs
    tipico = np.median([q[2] for q in qs])
    out = []
    for cx, cy, lado, ang in qs:
        if abs(lado / tipico - 2) < 0.35:
            t = np.deg2rad(ang)
            ex, ey = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
            for a in (-0.5, 0.5):
                for b in (-0.5, 0.5):
                    px, py = np.array((cx, cy)) + (ex * a + ey * b) * lado / 2
                    out.append((px, py, lado / 2, ang))
        else:
            out.append((cx, cy, lado, ang))
    return out


ultimos_quadrados = 0
ultimas_bordas = None


def achar_face(frame) -> Face | None:
    global ultimos_quadrados
    qs = _quadrados(frame)
    ultimos_quadrados = len(qs)
    if len(qs) < 4:
        return None
    pts = np.array([(q[0], q[1]) for q in qs])
    lados = np.array([q[2] for q in qs])
    melhor = None
    for cx, cy, lado, ang in qs:
        parecidos = np.abs(lados - lado) < lado * 0.3
        viz = pts[parecidos]
        if len(viz) < 4:
            continue
        d = np.linalg.norm(viz - (cx, cy), axis=1)
        d = d[d > lado * 0.5]
        if not len(d):
            continue
        # passo da grade; em cubo de corpo branco o vizinho mais perto pode estar a 2 casas
        passo = d.min() if d.min() < lado * 1.6 else lado * 1.2
        t = np.deg2rad(ang)
        ex = np.array([np.cos(t), np.sin(t)])
        ey = np.array([-np.sin(t), np.cos(t)])
        rel = viz - (cx, cy)
        u, v = rel @ ex / passo, rel @ ey / passo
        achados = {}
        for a, b, p in zip(u, v, viz):
            ia, ib = round(a), round(b)
            if abs(ia) <= 2 and abs(ib) <= 2 and abs(a - ia) < 0.25 and abs(b - ib) < 0.25:
                achados[(ia, ib)] = p
        cols = {k[0] for k in achados}
        lins = {k[1] for k in achados}
        # a grade só fica definida quando os achados cobrem 3 colunas e 3 linhas
        if max(cols) - min(cols) != 2 or max(lins) - min(lins) != 2 or len(achados) < 4:
            continue
        mc, ml = min(cols) + 1, min(lins) + 1
        n = len(achados)
        if melhor is None or n > melhor[0]:
            centro = np.array((cx, cy)) + (mc * ex + ml * ey) * passo
            melhor = (n, centro, ex, ey, passo, lado, list(achados.values()))
    if melhor is None:
        return None
    _, centro, ex, ey, passo, lado, reais = melhor
    # azulejo, parede, papel: grade sem cor nenhuma não é face de cubo
    if not _tem_cor(frame, centro, ex, ey, passo):
        return None
    eixos =[(ex, ey), (ey, -ex), (-ex, -ey), (-ey, ex)]
    ex, ey = max(eixos, key=lambda e: e[1][1])
    centros = np.array([centro + (col * ex + lin * ey) * passo for lin in (-1, 0, 1) for col in (-1, 0, 1)])
    # troca o ponto da grade pelo centro real do adesivo quando ele foi achado (guarda a perspectiva)
    achado = [False] * 9
    for i, c in enumerate(centros):
        if reais:
            d = [np.linalg.norm(r - c) for r in reais]
            j = int(np.argmin(d))
            if d[j] < passo * 0.35:
                centros[i] = reais[j]
                achado[i] = True
    if not _grade_confere(frame, centros, passo):
        return None
    return Face(centros, lado, float(np.degrees(np.arctan2(ex[1], ex[0]))), achado)


def _grade_confere(frame, centros, passo) -> bool:
    """Descarta grade deslocada (quadrado em cima da divisão) ou na escala errada
    (dois vizinhos da mesma cor sem divisão entre eles = mesma peça)."""
    h, w = frame.shape[:2]
    m = max(2, int(passo * 0.3))
    cores = []
    for x, y in centros.astype(int):
        p = frame[max(0, y - m):y + m, max(0, x - m):x + m].reshape(-1, 3).astype(float)
        if len(p) == 0 or p.std(axis=0).mean() > 30:
            return False
        cores.append(np.median(p, axis=0))
    c = centros.reshape(3, 3, 2)
    sem_divisao = 0
    for i in range(3):
        for j in range(3):
            for di, dj in ((0, 1), (1, 0)):
                a, b = i + di, j + dj
                if a > 2 or b > 2 or np.abs(cores[i * 3 + j] - cores[a * 3 + b]).max() > 40:
                    continue
                pontos = [(c[i, j] * (1 - t) + c[a, b] * t).astype(int) for t in np.linspace(0.3, 0.7, 9)]
                if not any(0 <= y < h and 0 <= x < w and ultimas_bordas[y, x] for x, y in pontos):
                    sem_divisao += 1
    return sem_divisao < 5


def _tem_cor(frame, centro, ex, ey, passo) -> bool:
    h, w = frame.shape[:2]
    coloridos = 0
    for lin in (-1, 0, 1):
        for col in (-1, 0, 1):
            x, y = (centro + (col * ex + lin * ey) * passo).astype(int)
            if 0 <= x < w and 0 <= y < h:
                coloridos += cv2.cvtColor(frame[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0, 1] > 90
    return coloridos >= 2


def nitidez(frame, face: Face) -> float:
    x0, y0 = face.centros.min(axis=0).astype(int)
    x1, y1 = face.centros.max(axis=0).astype(int)
    m = int(face.lado)
    roi = frame[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    if roi.size == 0:
        return 0
    return float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
