"""Mede até que velocidade de giro o rastreamento segue o cubo (volta sintética, graus por quadro)."""
import math
import random
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))
from test_pose import _rot, cena_cubo  # noqa: E402
from cubo.modelo import Cubo, embaralhar  # noqa: E402
from cubo.pose import Rastreador, CAM_PARA_TELA  # noqa: E402
from cubo import detector, visao  # noqa: E402

REAL = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
        "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}


def volta(passo, cubo, ref, quadros=50, eixo=0, prever=True, grade=True):
    r = Rastreador()
    r.prever = prever
    seguidos = 0
    for i in range(quadros):
        R = [_rot(i * passo, 15 + 10 * math.sin(i / 7), 0),
             _rot(20 + 10 * math.sin(i / 9), i * passo, 0),
             _rot(i * passo, i * passo * 0.5, 10)][eixo]
        frame = cena_cubo(cubo, R)
        # como no app: a grade recalibra sempre que acha uma face de frente
        face = detector.achar_face(frame) if (i == 0 or grade) else None
        centros = lida = None
        if face is not None:
            centros, lida = face.centros, [ref.cor(a) for a in face.amostras(frame)]
        ok = r.atualiza(frame, centros, lida, cubo, ref)
        if ok:
            est = CAM_PARA_TELA @ r.R_cam
            erro = math.degrees(math.acos(np.clip((np.trace(est.T @ R) - 1) / 2, -1, 1)))
            seguidos += erro < 15
    return seguidos / quadros


if __name__ == "__main__":
    cubo = Cubo.resolvido().aplica(embaralhar(25, random.Random(1)))
    ref = visao.Referencia([np.array(REAL[c], float) for c in "YOGWRB"], list("YOGWRB"),
                           visao.balanco_branco(REAL["W"]))
    for passo in (4, 8, 12, 16, 20):
        res = {}
        for prever in (False, True):
            res[prever] = np.mean([volta(passo, cubo, ref, eixo=e, prever=prever) for e in range(3)])
        print(f"{passo:2d}° por quadro (~{passo * 12}°/s): sem previsão {100 * res[False]:.0f}% | "
              f"com previsão {100 * res[True]:.0f}%", flush=True)
