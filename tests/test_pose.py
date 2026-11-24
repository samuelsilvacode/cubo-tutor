import math
import random

import cv2
import numpy as np

from cubo.modelo import Cubo, embaralhar, grade_da_face
from cubo.pose import CAM_PARA_TELA, Rastreador


def _rot(yaw, pitch, roll):
    a, b, c = map(math.radians, (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


def _cena(cubo, R_tela, rng, w=1280, h=720):
    """Projeta os 9 centros da face que fica virada para a câmera numa pose conhecida."""
    R_cam = CAM_PARA_TELA @ R_tela  # tela -> câmera (a matriz é a própria inversa)
    f = 0.75 * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    # a face cuja normal aponta mais para a câmera
    melhor = max("URFDLB", key=lambda fc: (R_tela @ np.array(grade_da_face(fc)[4][1]))[2])
    grade = grade_da_face(melhor)
    pts3 = np.array([np.array(p) + 0.5 * np.array(n) for p, n in grade], float)
    img, _ = cv2.projectPoints(pts3, cv2.Rodrigues(R_cam)[0], np.array([0, 0, 14.0]), K, None)
    img = img.reshape(9, 2) + rng.normal(0, 1.0, (9, 2))
    # o detector entrega a face "de pé": das 4 rotações da grade, a que tem as linhas mais para baixo
    idx = np.arange(9).reshape(3, 3)
    ordem = max((np.rot90(idx, k).reshape(9) for k in range(4)),
                key=lambda o: (img[o[6]] + img[o[7]] + img[o[8]] - img[o[0]] - img[o[1]] - img[o[2]])[1])
    lida = [cubo.face(melhor)[i] for i in ordem]
    return img[ordem], lida, melhor


def test_rastreia_rotacao():
    cubo = Cubo.resolvido().aplica(embaralhar(25, random.Random(1)))
    rng = np.random.default_rng(0)
    erros = []
    for _ in range(60):
        R = _rot(rng.uniform(-30, 30), rng.uniform(-25, 25), rng.uniform(-20, 20))
        pts, lida, face = _cena(cubo, R, rng)
        r = Rastreador()
        assert r.pela_grade(pts, lida, cubo, (720, 1280))
        assert r.face == face
        est = CAM_PARA_TELA @ r.R_cam
        ang = math.degrees(math.acos(np.clip((np.trace(est.T @ R) - 1) / 2, -1, 1)))
        erros.append(ang)
    assert np.median(erros) < 5


def cena_cubo(cubo, R_tela, dist=11.0, w=1280, h=720, rng=None):
    """Imagem sintética do cubo inteiro numa pose (faces tortas, quinas), com fundo bagunçado."""
    rng = rng or np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(40, 160, (h, w, 3), dtype=np.uint8), (31, 31), 0)
    R_cam = CAM_PARA_TELA @ R_tela
    f = 0.75 * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    t = np.array([0, 0, dist])
    rv = cv2.Rodrigues(R_cam)[0]
    real = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
            "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}
    polis = []
    for fc in "URFDLB":
        n = np.array(grade_da_face(fc)[4][1], float)
        if (R_cam @ n) @ -(R_cam @ (n * 1.5) + t) <= 0:
            continue
        u = np.array([n[1], n[2], n[0]])
        v = np.cross(n, u)
        corpo = [n * 1.5 + (a * u + b * v) * 1.5 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        polis.append((corpo, (20, 20, 20)))
        for (pos, nn), cor in zip(grade_da_face(fc), cubo.face(fc)):
            c = np.array(pos) + 0.5 * np.array(nn)
            polis.append(([c + (a * u + b * v) * 0.44 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))], real[cor]))
    for quad, cor in polis:
        p, _ = cv2.projectPoints(np.array(quad, float), rv, t, K, None)
        cv2.fillConvexPoly(img, p.reshape(-1, 2).astype(np.int32), cor, cv2.LINE_AA)
    return img
