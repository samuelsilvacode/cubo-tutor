"""Junta leituras de vários quadros e monta o cubo no final."""
from __future__ import annotations

from itertools import permutations, product

import cv2
import numpy as np

from . import visao
from .correcao import INDICE, corrigir
from .modelo import Cubo

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

    def _junta_parecidos(self):
        for i, a in enumerate(self.grupos):
            for b in self.grupos[i + 1:]:
                for k in range(4):
                    if diferenca(girar(b.media, k), a.media) < MESMA_FACE:
                        a.leituras += [girar(x, k) for x in b.leituras]
                        a.leituras = a.leituras[-40:]
                        self.grupos.remove(b)
                        return True
        return False

    def faces(self) -> list[Grupo]:
        while self._junta_parecidos():
            pass
        return sorted(self.grupos, key=lambda g: -g.votos)[:6]

    def prontas(self) -> int:
        return sum(1 for g in self.faces() if g.votos >= MIN_VOTOS)


# onde cada cor de centro fica, no esquema padrão (amarelo em cima, verde na frente)
POSICAO = {"Y": "U", "O": "R", "G": "F", "W": "D", "R": "L", "B": "B"}
_REF = Cubo.resolvido("YOGWRB")
PECAS = [[INDICE[(pos, n)] for n in ads] for pos, ads in _REF.pecas().items() if len(ads) > 1]
VALIDAS = {frozenset(ads.values()) for ads in _REF.pecas().values() if len(ads) > 1}


def _nomeacoes(centros, quantas=6):
    """Jeitos mais prováveis de dar nome às 6 cores de centro."""
    ref = visao._lab([visao.BGR[c] for c in visao.CORES])
    lab = visao._lab(centros)
    custo = np.linalg.norm(lab[:, None] - ref[None], axis=2)
    opcoes = []
    for perm in permutations(range(6)):
        opcoes.append((sum(custo[g, perm[g]] for g in range(6)), [visao.CORES[perm[g]] for g in range(6)]))
    opcoes.sort(key=lambda x: x[0])
    return opcoes[:quantas]


def montar(faces: list[np.ndarray]):
    """faces: 6 médias (9x3 BGR), em qualquer ordem e giradas de qualquer jeito.

    Devolve (leitura crua, leitura corrigida, Referencia, cor do centro da frente).
    """
    centros = [f[4] for f in faces]
    todas_g = [x for f in faces for x in f]
    candidatos = []
    for _, nomes in _nomeacoes(centros):
        wb = visao.balanco_branco(centros[nomes.index("W")])
        perto = visao.distancias(todas_g, centros, wb).argmin(axis=1)
        letras = [[nomes[j] for j in perto[g * 9:(g + 1) * 9]] for g in range(6)]
        giradas = [[girar(np.array(letras[g]).reshape(9, 1), k).reshape(9).tolist() for k in range(4)]
                   for g in range(6)]
        onde = {POSICAO[nomes[g]]: g for g in range(6)}
        ordem = [onde[f] for f in "URFDLB"]
        pontuados = []
        for giros in product(range(4), repeat=6):
            texto = [c for f, g in enumerate(ordem) for c in giradas[g][giros[f]]]
            pontos = sum(frozenset(texto[i] for i in p) in VALIDAS for p in PECAS)
            pontuados.append((pontos, giros))
        top = max(p for p, _ in pontuados)
        # empates vão todos para o desempate pela correção
        candidatos += [(p, nomes, ordem, g) for p, g in pontuados if p == top][:8]

    finais = []
    top = max(c[0] for c in candidatos)
    for pontos, nomes, ordem, giros in candidatos:
        if pontos < top:
            continue
        lista = [girar(faces[g], giros[f]) for f, g in enumerate(ordem)]
        todas = [x for f in lista for x in f]
        cs = [f[4] for f in lista]
        wb = visao.balanco_branco(cs[3])
        nomes_urf = [nomes[g] for g in ordem]
        certas, custo = corrigir(visao.distancias(todas, cs, wb), nomes_urf, com_custo=True)
        finais.append((custo, todas, cs, nomes_urf, wb, certas))
    _, todas, cs, nomes_urf, wb, certas = min(finais, key=lambda x: x[0])
    cruas = "".join(visao.classificar(todas, cs, nomes_urf, wb))
    return cruas, certas, visao.Referencia(cs, nomes_urf, wb), nomes_urf[2]
