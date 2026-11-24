"""Junta leituras de vários quadros e monta o cubo no final."""
from __future__ import annotations

from itertools import permutations

import cv2
import numpy as np

from . import visao
from .correcao import corrigir

MIN_VOTOS = 6
MESMA_FACE = 20


def _norm(a: np.ndarray) -> np.ndarray:
    return np.clip(a * (128 / max(a.mean(), 1)), 0, 255)


def _lab(a: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor((a / 255).astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    lab = lab.reshape(-1, 3).astype(float)
    lab[:, 0] *= 0.5
    return lab


def girar(face: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(face.reshape(3, 3, -1), k).reshape(9, -1)


def diferenca(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(_lab(_norm(a)) - _lab(_norm(b)), axis=1).mean())


class Grupo:
    def __init__(self, leitura):
        self.leituras = [leitura]

    @property
    def media(self) -> np.ndarray:
        return np.median(np.array(self.leituras), axis=0)

    @property
    def votos(self) -> int:
        return len(self.leituras)


class Acumulador:
    def __init__(self):
        self.grupos: list[Grupo] = []

    def adiciona(self, amostras) -> Grupo:
        a = np.array(amostras, float)
        melhor = None
        for g in self.grupos:
            ref = g.media
            for k in range(4):
                d = diferenca(girar(a, k), ref)
                if melhor is None or d < melhor[0]:
                    melhor = (d, g, k)
        if melhor and melhor[0] < MESMA_FACE:
            _, g, k = melhor
            g.leituras.append(girar(a, k))
            if len(g.leituras) > 40:
                g.leituras.pop(0)
            return g
        g = Grupo(a)
        self.grupos.append(g)
        # descarta grupos de leitura ruim que nunca se repetiram
        if len(self.grupos) > 12:
            self.grupos.remove(min(self.grupos[:-1], key=lambda x: x.votos))
        return g

    def faces(self) -> list[Grupo]:
        return sorted(self.grupos, key=lambda g: -g.votos)[:6]

    def prontas(self) -> int:
        return sum(1 for g in self.faces() if g.votos >= MIN_VOTOS)


def _croma(bgr) -> tuple[float, float, float]:
    L, a, b = _lab(np.array([bgr], float))[0]
    return L, a, b


def montar(faces: list[np.ndarray]):
    """faces: 6 médias (9x3 BGR) em qualquer ordem. Acha qual é qual e devolve o cubo mais provável.

    Devolve (leitura crua, leitura corrigida, Referencia, cor do centro da frente).
    """
    centros = [_croma(f[4]) for f in faces]
    # branco: claro e pouco colorido (com luz amarelada ele fica creme, então o brilho pesa)
    brancos = sorted(range(6), key=lambda i: -(2 * centros[i][0] - np.hypot(*centros[i][1:])))[:2]
    candidatos = []
    for br in brancos:
        resto = [i for i in range(6) if i != br]
        for am in sorted(resto, key=lambda i: -(centros[i][2] - centros[i][1]))[:2]:
            candidatos.append((br, am))

    melhor = None
    for branco, am in candidatos:
        resto = [i for i in range(6) if i != branco]
        lados = [i for i in resto if i != am]
        frente = lados[0]
        for r, b, l in permutations(lados[1:]):
            for ku in range(4):
                for kd in range(4):
                    ordem = {"U": girar(faces[am], ku), "R": faces[r], "F": faces[frente],
                             "D": girar(faces[branco], kd), "L": faces[l], "B": faces[b]}
                    lista = [ordem[f] for f in "URFDLB"]
                    todas = [x for f in lista for x in f]
                    cs = [f[4] for f in lista]
                    wb = visao.balanco_branco(cs[3])
                    nomes = visao.nomes_dos_centros([np.clip(np.array(c) * wb, 0, 255) for c in cs])
                    dist = visao.distancias(todas, cs, wb)
                    certas, custo = corrigir(dist, nomes, com_custo=True)
                    if melhor is None or custo < melhor[0]:
                        melhor = (custo, todas, cs, nomes, wb, certas)
    _, todas, cs, nomes, wb, certas = melhor
    cruas = "".join(visao.classificar(todas, cs, nomes, wb))
    return cruas, certas, visao.Referencia(cs, nomes, wb), nomes[2]
