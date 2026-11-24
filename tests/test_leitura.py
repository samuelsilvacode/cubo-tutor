import random

import numpy as np

from cubo.leitura import Acumulador, girar, montar
from cubo.modelo import Cubo, ORDEM_FACES, embaralhar

REAL = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
        "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}


def _sessao(k, rng):
    alvo = Cubo.resolvido().aplica(embaralhar(25, random.Random(500 + k)))
    tinta = np.array([rng.uniform(.75, 1), rng.uniform(.9, 1.05), rng.uniform(1, 1.2)])
    acc = Acumulador()
    seq = list(ORDEM_FACES) * 3
    random.Random(k).shuffle(seq)
    giro = {f: int(rng.integers(4)) for f in "URFDLB"}
    for f in seq:
        for _ in range(4):
            v = np.array([REAL[c] for c in alvo.face(f)], float) * tinta * rng.uniform(.85, 1.15)
            v = np.clip(v * rng.uniform(.95, 1.05, (9, 1)) + rng.normal(0, 5, (9, 3)), 0, 255)
            acc.adiciona(girar(v, giro[f]))
        if rng.random() < 0.3:
            acc.adiciona(rng.uniform(0, 255, (9, 3)))
    return alvo, acc


def test_faces_em_qualquer_ordem_e_giradas():
    rng = np.random.default_rng(0)
    certos = 0
    for k in range(25):
        alvo, acc = _sessao(k, rng)
        assert acc.prontas() == 6
        _, certas, _, _ = montar([g.media for g in acc.faces()])
        certos += any(Cubo.de_texto(certas) == alvo.copia().aplica(y) for y in ["", "y", "y2", "y'"])
    assert certos >= 24
