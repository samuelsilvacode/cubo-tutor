"""Onde o cubo está na mão: qual face está virada para a câmera e a rotação 3D dele."""
from __future__ import annotations

import cv2
import numpy as np

from .modelo import Cubo, ORDEM_FACES, grade_da_face

# câmera (x direita, y baixo, z para dentro) -> réplica (x direita, y cima, z para quem olha)
CAM_PARA_TELA = np.diag([1.0, -1.0, -1.0])


def _ortonormal(m):
    u, _, vt = np.linalg.svd(m)
    return u @ vt


class Rastreador:
    def __init__(self, suavizar: float = 0.45):
        self.R: np.ndarray | None = None
        self.face: str | None = None
        self.acertos = 0
        self.suavizar = suavizar

    def identifica(self, lida: list[str], cubo: Cubo):
        """Qual face do cubo é essa leitura e como ela está girada (np.rot90 k vezes)."""
        melhor = (0, None, 0)
        for f in ORDEM_FACES:
            grade = np.array(cubo.face(f)).reshape(3, 3)
            for k in range(4):
                n = sum(a == b for a, b in zip(np.rot90(grade, k).reshape(9), lida))
                if n > melhor[0]:
                    melhor = (n, f, k)
        return melhor

    def atualiza(self, centros: np.ndarray, lida: list[str], cubo: Cubo, tamanho, espelho=False) -> bool:
        acertos, face, k = self.identifica(lida, cubo)
        self.acertos = acertos
        if acertos < 7:
            return False
        h, w = tamanho
        f = 0.75 * w
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
        idx = np.rot90(np.arange(9).reshape(3, 3), k).reshape(9)
        grade = grade_da_face(face)
        obj = np.array([np.array(grade[i][0]) + 0.5 * np.array(grade[i][1]) for i in idx], float)
        ok, rvecs, tvecs, erros = cv2.solvePnPGeneric(obj, centros.astype(float), K, None,
                                                       flags=cv2.SOLVEPNP_IPPE)
        if not ok or not len(rvecs):
            return False
        opcoes = sorted(zip(np.ravel(erros), [CAM_PARA_TELA @ cv2.Rodrigues(r)[0] for r in rvecs]),
                        key=lambda x: x[0])
        novo = opcoes[0][1]
        # face plana tem duas inclinações quase iguais na imagem: se empatar, fica com a mais perto da anterior
        if self.R is not None and len(opcoes) > 1 and opcoes[1][0] < opcoes[0][0] * 1.5 + 0.5:
            alvo = self.R if not espelho else np.diag([-1.0, 1, 1]) @ self.R
            novo = min((o[1] for o in opcoes[:2]), key=lambda r: np.linalg.norm(r - alvo))
        if espelho:
            novo = np.diag([-1.0, 1, 1]) @ novo
        if self.R is None or self.face != face and np.linalg.norm(novo - self.R) > 1.5:
            self.R = novo
        else:
            self.R = _ortonormal((1 - self.suavizar) * self.R + self.suavizar * novo)
        self.face = face
        return True
