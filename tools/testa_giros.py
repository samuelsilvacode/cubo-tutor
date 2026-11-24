"""Gira camadas de um cubo sintético na frente da 'câmera' e confere se o modo espelho vê cada giro."""
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))
from test_pose import _rot, cena_cubo  # noqa: E402
from cubo.app import App, Tela  # noqa: E402
from cubo.modelo import Cubo, embaralhar  # noqa: E402
from cubo import visao  # noqa: E402

REAL = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
        "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}


PELE = (70, 95, 140)  # BGR de pele


def dedos(img, rng):
    """Desenha 3 ou 4 'dedos' por cima do cubo (pior caso: a mão não é reconhecida como mão)."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    lado = rng.choice([-1, 1])
    for k in range(rng.randint(3, 4)):
        y = cy - 60 + k * 45 + rng.randint(-10, 10)
        x0 = cx + lado * 260
        x1 = cx + lado * rng.randint(40, 140)
        cv2.line(img, (x0, y), (x1, y + rng.randint(-15, 15)), PELE, 32)
        cv2.circle(img, (x1, y), 16, PELE, -1)
    return img


def rodada(semente, n_giros=10, quadros_por_giro=8, com_dedos=False):
    rng = random.Random(semente)
    real = Cubo.resolvido().aplica(embaralhar(25, rng))
    a = App(camera=None, estado=real.texto())
    a.espelho = False
    a.ref = visao.Referencia([np.array(REAL[c], float) for c in "YOGWRB"], list("YOGWRB"),
                             visao.balanco_branco(REAL["W"]))
    a.tecla(ord("e"))
    a.maos.maos = []
    a.maos.ok = False
    feitos, i = [], 0
    # começa com uma face de frente e vai inclinando, como quem pega o cubo
    for k in range(12):
        a.frame = cena_cubo(real, _rot(30 * k / 11, 25 * k / 11, 0))
        a.tela_espelho(Tela())
    giros = [rng.choice([f + s for f in "URFDLB" for s in ("", "'", "2")]) for _ in range(n_giros)]
    for g in [None] + giros:
        if g:
            real.aplica(g)
            feitos.append(g)
        for _ in range(quadros_por_giro):
            a.frame = cena_cubo(real, _rot(30 + 8 * math.sin(i / 5), 25 + 5 * math.cos(i / 4), 0))
            if com_dedos:
                a.frame = dedos(a.frame, rng)
            a.tela_espelho(Tela())
            i += 1
    return feitos, a.giros_feitos, a.cubo_espelho == real


if __name__ == "__main__":
    for com_dedos in (False, True):
        certos, vistos_tot, giros_tot = 0, 0, 0
        for sem in range(5):
            feitos, vistos, igual = rodada(sem, com_dedos=com_dedos)
            certos += igual
            # quantos giros seguidos ele acertou desde o começo
            n = 0
            for f, v in zip(feitos, vistos):
                if f != v:
                    break
                n += 1
            vistos_tot += n
            giros_tot += len(feitos)
        print(f"{'com dedos na frente' if com_dedos else 'sem dedos'}: {certos}/5 rodadas perfeitas | "
              f"{vistos_tot}/{giros_tot} giros vistos em sequência certa", flush=True)
