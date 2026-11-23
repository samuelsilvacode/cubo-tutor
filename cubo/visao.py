"""Leitura das cores pela webcam."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

CORES = "WYGBRO"

# BGR aproximado de cada cor, usado só para desenhar e nomear os centros
BGR = {
    "W": (235, 235, 235), "Y": (0, 215, 255), "G": (72, 155, 0),
    "B": (173, 70, 0), "R": (46, 16, 200), "O": (19, 106, 255),
}


def grade(frame: np.ndarray, frac: float = 0.45) -> list[tuple[int, int, int, int]]:
    """9 retângulos (x, y, w, h) de uma grade centralizada, em ordem de leitura."""
    h, w = frame.shape[:2]
    lado = int(min(h, w) * frac)
    x0, y0, cel = (w - lado) // 2, (h - lado) // 2, lado // 3
    return [(x0 + c * cel, y0 + l * cel, cel, cel) for l in range(3) for c in range(3)]


def amostras(frame: np.ndarray) -> list[np.ndarray]:
    out = []
    for x, y, w, h in grade(frame):
        m = int(w * 0.3)
        pedaco = frame[y + m:y + h - m, x + m:x + w - m].reshape(-1, 3)
        out.append(np.median(pedaco, axis=0).astype(np.uint8))
    return out


def cor_rapida(bgr: np.ndarray) -> str:
    """Palpite por matiz, para o preview ao vivo antes da calibração."""
    h, s, v = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    if s < 60 and v > 110:
        return "W"
    if h < 5 or h >= 160:
        return "R"
    if h < 19:
        return "O"
    if h < 38:
        return "Y"
    if h < 90:
        return "G"
    if h < 140:
        return "B"
    return "R"


def _lab(bgrs) -> np.ndarray:
    arr = np.array(bgrs, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    lab[:, 0] *= 0.5  # luz varia muito entre faces, pesa menos que a cor
    return lab


def nomear_centros(centros: list[np.ndarray]) -> list[str]:
    palpite = [cor_rapida(c) for c in centros]
    if len(set(palpite)) == 6:
        return palpite
    ref = _lab([BGR[c] for c in CORES])
    lab = _lab(centros)
    custo = np.linalg.norm(lab[:, None] - ref[None], axis=2)
    lin, col = linear_sum_assignment(custo)
    out = [""] * 6
    for i, j in zip(lin, col):
        out[i] = CORES[j]
    return out


def classificar(todas: list[np.ndarray], centros: list[np.ndarray], nomes: list[str]) -> list[str]:
    """Classifica os 54 adesivos usando os centros como referência, 9 de cada cor."""
    lab = _lab(todas)
    ref = _lab(centros)
    dist = np.linalg.norm(lab[:, None] - ref[None], axis=2)
    custo = np.repeat(dist, 9, axis=1)
    lin, col = linear_sum_assignment(custo)
    out = [""] * len(todas)
    for i, j in zip(lin, col):
        out[i] = nomes[j // 9]
    return out


class Referencia:
    """Depois do escaneamento: classifica ao vivo pelos centros calibrados."""

    def __init__(self, centros: list[np.ndarray], nomes: list[str]):
        self.ref = _lab(centros)
        self.nomes = nomes

    def cor(self, bgr: np.ndarray) -> str:
        d = np.linalg.norm(self.ref - _lab([bgr])[0], axis=1)
        return self.nomes[int(np.argmin(d))]
