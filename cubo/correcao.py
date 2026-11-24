"""Acha o cubo válido mais parecido com o que a câmera leu."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .modelo import Cubo, ORDEM_FACES, grade_da_face

CHAVES = [c for f in ORDEM_FACES for c in grade_da_face(f)]
INDICE = {c: i for i, c in enumerate(CHAVES)}


def _ordem_canto(p):
    ny, nx, nz = (0, p[1], 0), (p[0], 0, 0), (0, 0, p[2])
    # mesma orientação (determinante positivo) em todos os cantos
    return [ny, nx, nz] if p[0] * p[1] * p[2] < 0 else [ny, nz, nx]


def _ordem_meio(p):
    ns = [tuple(p[i] if j == i else 0 for j in range(3)) for i in range(3) if p[i]]
    return sorted(ns, key=lambda n: [1, 0, 2].index(next(i for i in range(3) if n[i])))


class _Pecas:
    def __init__(self, nomes: list[str]):
        ref = Cubo.resolvido("".join(nomes))
        self.cantos, self.meios = [], []
        for pos in ref.pecas():
            if sum(map(abs, pos)) == 3:
                self.cantos.append(pos)
            elif sum(map(abs, pos)) == 2:
                self.meios.append(pos)
        self.cor_canto = [tuple(ref.a[(p, n)] for n in _ordem_canto(p)) for p in self.cantos]
        self.cor_meio = [tuple(ref.a[(p, n)] for n in _ordem_meio(p)) for p in self.meios]


def corrigir(dist: np.ndarray, nomes: list[str], com_custo: bool = False):
    """dist[i, j]: distância do adesivo i (ordem URFDLB) ao centro j. Devolve as 54 letras."""
    pc = _Pecas(nomes)
    col = {c: j for j, c in enumerate(nomes)}

    def custo(pos, ordem, cores):
        return sum(dist[INDICE[(pos, n)], col[c]] for n, c in zip(ordem, cores))

    def gira(cores, o):
        out = [None] * len(cores)
        for k, c in enumerate(cores):
            out[(k + o) % len(cores)] = c
        return tuple(out)

    def tabela(slots, cores, ordem_fn, n_ori):
        t = np.zeros((len(slots), len(cores), n_ori))
        for s, p in enumerate(slots):
            for q, c in enumerate(cores):
                for o in range(n_ori):
                    t[s, q, o] = custo(p, ordem_fn(p), gira(c, o))
        return t

    tc = tabela(pc.cantos, pc.cor_canto, _ordem_canto, 3)
    tm = tabela(pc.meios, pc.cor_meio, _ordem_meio, 2)

    def melhor(t, perm, mod):
        ori = [int(np.argmin(t[s, perm[s]])) for s in range(len(perm))]
        total = sum(t[s, perm[s], ori[s]] for s in range(len(perm)))
        falta = (-sum(ori)) % mod
        if falta:
            # ajusta a peça que menos piora
            s = min(range(len(perm)), key=lambda s: t[s, perm[s], (ori[s] + falta) % mod] - t[s, perm[s], ori[s]])
            total += t[s, perm[s], (ori[s] + falta) % mod] - t[s, perm[s], ori[s]]
            ori[s] = (ori[s] + falta) % mod
        return total, ori

    def atribui(t):
        lin, col_ = linear_sum_assignment(t.min(axis=2))
        return list(col_[np.argsort(lin)])

    perm_c, perm_m = atribui(tc), atribui(tm)
    if _paridade(perm_c) != _paridade(perm_m):
        opcoes = []
        for perm, t, mod, qual in ((perm_c, tc, 3, "c"), (perm_m, tm, 2, "m")):
            for a, b in combinations(range(len(perm)), 2):
                p2 = list(perm)
                p2[a], p2[b] = p2[b], p2[a]
                opcoes.append((melhor(t, p2, mod)[0] - melhor(t, perm, mod)[0], qual, p2))
        _, qual, p2 = min(opcoes, key=lambda x: x[0])
        if qual == "c":
            perm_c = p2
        else:
            perm_m = p2

    _, ori_c = melhor(tc, perm_c, 3)
    _, ori_m = melhor(tm, perm_m, 2)

    letras = [""] * 54
    for f in range(6):
        letras[f * 9 + 4] = nomes[f]
    for s, p in enumerate(pc.cantos):
        for n, c in zip(_ordem_canto(p), gira(pc.cor_canto[perm_c[s]], ori_c[s])):
            letras[INDICE[(p, n)]] = c
    for s, p in enumerate(pc.meios):
        for n, c in zip(_ordem_meio(p), gira(pc.cor_meio[perm_m[s]], ori_m[s])):
            letras[INDICE[(p, n)]] = c
    texto = "".join(letras)
    if com_custo:
        return texto, float(sum(dist[i, col[c]] for i, c in enumerate(texto)))
    return texto


def _paridade(perm) -> int:
    vistos, trocas = set(), 0
    for i in range(len(perm)):
        j, tam = i, 0
        while j not in vistos:
            vistos.add(j)
            j = perm[j]
            tam += 1
        if tam:
            trocas += tam - 1
    return trocas % 2
