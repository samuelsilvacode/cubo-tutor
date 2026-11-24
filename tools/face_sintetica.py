"""Desenha uma face de cubo numa imagem (para testar o detector sem câmera)."""
import cv2
import numpy as np

REAL = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
        "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}


def desenha(cores, rng, angulo=0.0, centro=None, lado=48, fundo="ruido", tinta=(1, 1, 1)):
    h, w = 720, 1280
    if fundo == "ruido":
        img = cv2.GaussianBlur(rng.integers(40, 200, (h, w, 3), dtype=np.uint8), (31, 31), 0)
    else:
        img = np.full((h, w, 3), 90, np.uint8)
    cx, cy = centro or (w // 2, h // 2)
    passo = lado * 1.12
    t = np.deg2rad(angulo)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corpo = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]]) * passo * 1.03
    cv2.fillPoly(img, [(corpo @ rot.T + (cx, cy)).astype(np.int32)], (20, 20, 20))
    for i, c in enumerate(cores):
        lin, col = divmod(i, 3)
        base = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * lado
        quad = base + ((col - 1) * passo, (lin - 1) * passo)
        cor = np.clip(np.array(REAL[c]) * tinta + rng.normal(0, 4, 3), 0, 255)
        cv2.fillPoly(img, [(quad @ rot.T + (cx, cy)).astype(np.int32)], cor.tolist(), lineType=cv2.LINE_AA)
    return img
