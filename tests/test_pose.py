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
        r = Rastreador(suavizar=1.0)
        assert r.atualiza(pts, lida, cubo, (720, 1280))
        assert r.face == face
        ang = math.degrees(math.acos(np.clip((np.trace(r.R.T @ R) - 1) / 2, -1, 1)))
        erros.append(ang)
    assert np.median(erros) < 5
