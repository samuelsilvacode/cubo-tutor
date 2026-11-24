"""CIEDE2000: diferença de cor como o olho percebe (padrão CIE). Vetorizado em numpy.

Referência: Sharma, Wu e Dalal (2005), "The CIEDE2000 color-difference formula".
"""
from __future__ import annotations

import cv2
import numpy as np


def lab(bgr) -> np.ndarray:
    """BGR (0-255) -> CIELAB de verdade (L 0-100, a e b em torno de 0)."""
    a = np.clip(np.asarray(bgr, float).reshape(-1, 3), 0, 255) / 255
    return cv2.cvtColor(a.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Diferença entre cada cor de lab1 (N x 3) e cada cor de lab2 (M x 3): matriz N x M."""
    L1, a1, b1 = [x[:, None] for x in lab1.T]
    L2, a2, b2 = [x[None, :] for x in lab2.T]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lm = (L1 + L2) / 2
    Cmp = (C1p + C2p) / 2
    soma = h1p + h2p
    hm = np.where(np.abs(h1p - h2p) > 180, np.where(soma < 360, soma + 360, soma - 360) / 2, soma / 2)
    hm = np.where(C1p * C2p == 0, soma, hm)
    T = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    dtheta = 30 * np.exp(-(((hm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cmp ** 7 / (Cmp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    Sc = 1 + 0.045 * Cmp
    Sh = 1 + 0.015 * Cmp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
