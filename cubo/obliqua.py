"""Face do cubo vista de lado: adesivos viram losangos e a grade fica achatada.

O detector principal procura quadrados; aqui cada adesivo é um quadrilátero
qualquer, e a grade 3x3 é montada com os dois lados dele (vetores a e b),
o que aceita a face inclinada até uns 70 graus.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Face


def _bordas(frame):
    cinza = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bordas = cv2.Canny(cinza, 20, 60)
    sat = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    bordas |= cv2.Canny(sat, 30, 90)
    return cv2.dilate(bordas, np.ones((3, 3), np.uint8))


def quadrilateros(frame):
    """(centro, a, b) de cada mancha de quatro lados; a e b são os lados médios."""
    bordas = _bordas(frame)
    conts, _ = cv2.findContours(255 - bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = frame.shape[0] * frame.shape[1]
    out = []
    for c in conts:
        area = cv2.contourArea(c)
        if not area_img * 0.0002 < area < area_img * 0.06:
            continue
        hull = cv2.convexHull(c)
        if area / max(cv2.contourArea(hull), 1) < 0.85:
            continue
        p = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)
        if len(p) != 4:
            continue
        a = ((p[1] - p[0]) + (p[2] - p[3])) / 2
        b = ((p[3] - p[0]) + (p[2] - p[1])) / 2
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if min(la, lb) < 6:
            continue
        # cubo sem borda: duas ou três peças iguais lado a lado viram uma mancha comprida
        n = max(la, lb) / min(la, lb)
        k = round(n)
        if k > 3 or abs(n - k) > 0.3:
            continue
        centro = p.mean(axis=0)
        if k == 1:
            out.append((centro, a, b))
            continue
        longo, curto = (a, b) if la > lb else (b, a)
        passo = longo / k
        for i in range(k):
            out.append((centro + passo * (i - (k - 1) / 2), passo, curto))
    return out


def _parecidos(a1, b1, a2, b2, tol=0.3):
    """Mesmo formato de adesivo, aceitando a e b trocados ou com sinal invertido."""
    def perto(u, v):
        return min(np.linalg.norm(u - v), np.linalg.norm(u + v)) < tol * max(np.linalg.norm(u), 1)
    return (perto(a1, a2) and perto(b1, b2)) or (perto(a1, b2) and perto(b1, a2))


def _meios(idx):
    """Coluna (ou linha) do meio possível para os índices achados."""
    lo, hi = min(idx), max(idx)
    if hi - lo == 2:
        return [lo + 1]
    if hi - lo == 1:
        return [lo, hi]
    return [lo - 1, lo, lo + 1]


def _nota_grade(frame, centro, a, b):
    """Quantos dos 9 pontos caem em adesivo: mancha lisa, colorida ou branca."""
    h, w = frame.shape[:2]
    r = max(2, int(0.2 * min(np.linalg.norm(a), np.linalg.norm(b))))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    n = 0
    for lin in (-1, 0, 1):
        for col in (-1, 0, 1):
            x, y = (centro + col * a + lin * b).astype(int)
            if not (r <= x < w - r and r <= y < h - r):
                continue
            p = frame[y - r:y + r, x - r:x + r].reshape(-1, 3).astype(float)
            s, v = np.median(hsv[y - r:y + r, x - r:x + r].reshape(-1, 3), axis=0)[1:]
            if p.std(axis=0).mean() < 25 and (s > 80 or v > 170):
                n += 1
    return n


def achar_face(frame, minimo: int = 5) -> Face | None:
    """Maior grade 3x3 de adesivos com o mesmo formato. Pede `minimo` adesivos achados de verdade."""
    qs = quadrilateros(frame)
    if len(qs) < minimo:
        return None
    melhor = None
    for c0, a, b in qs:
        base = np.c_[a, b]
        if abs(np.linalg.det(base)) < 1e-6:
            continue
        iguais = [c for c, a2, b2 in qs if _parecidos(a, b, a2, b2)]
        uv = np.array([np.linalg.inv(base) @ (c - c0) for c in iguais])
        # o passo da grade é maior que o adesivo (folga, canto arredondado): mede pelo vizinho mais perto
        escalas = []
        for eixo in (0, 1):
            outro = 1 - eixo
            viz = np.abs(uv[(np.abs(uv[:, outro]) < 0.35) & (np.abs(uv[:, eixo]) > 0.8) & (np.abs(uv[:, eixo]) < 1.7), eixo])
            escalas.append(float(viz.min()) if len(viz) else None)
        if escalas[0] is None and escalas[1] is None:
            continue
        su = escalas[0] or escalas[1]
        sv = escalas[1] or escalas[0]
        a, b = a * su, b * sv
        inv = np.linalg.inv(np.c_[a, b])
        achados = {}
        for c in iguais:
            u, v = inv @ (c - c0)
            iu, iv = round(u), round(v)
            if abs(iu) <= 2 and abs(iv) <= 2 and abs(u - iu) < 0.25 and abs(v - iv) < 0.25:
                achados[(iu, iv)] = c
        if len(achados) < minimo:
            continue
        cols = {k[0] for k in achados}
        lins = {k[1] for k in achados}
        if max(cols) - min(cols) > 2 or max(lins) - min(lins) > 2:
            continue
        # fileira inteira sem adesivo achado (sombra, mancha emendada): testa os dois lados
        for mc in _meios(cols):
            for ml in _meios(lins):
                nota = _nota_grade(frame, c0 + mc * a + ml * b, a, b)
                chave = (len(achados), nota)
                if nota >= 8 and (melhor is None or chave > melhor[0]):
                    melhor = (chave, achados, c0, a, b, mc, ml)
    if melhor is None:
        return None
    _, achados, c0, a, b, mc, ml = melhor
    centro = c0 + mc * a + ml * b
    # de pé: b aponta para baixo na imagem e a para a direita
    if abs(a[1]) > abs(b[1]):
        a, b = b, a
    if b[1] < 0:
        b = -b
    if a[0] < 0:
        a = -a
    grade = np.array([centro + col * a + lin * b for lin in (-1, 0, 1) for col in (-1, 0, 1)])
    reais = np.array(list(achados.values()))
    achado = [False] * 9
    passo = min(np.linalg.norm(a), np.linalg.norm(b))
    for i, p in enumerate(grade):
        d = np.linalg.norm(reais - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < passo * 0.35:
            grade[i] = reais[j]
            achado[i] = True
    return Face(grade, float(passo), float(np.degrees(np.arctan2(a[1], a[0]))), achado)
