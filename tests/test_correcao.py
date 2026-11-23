import random

import numpy as np
import pytest

from cubo.correcao import corrigir
from cubo.modelo import Cubo, ORDEM_FACES, embaralhar
from cubo.otimo import checar_pecas

NOMES = list("YOGWRB")


def _dist(texto, rng, ruido=0.15):
    d = np.ones((54, 6)) + rng.normal(0, ruido, (54, 6))
    for i, c in enumerate(texto):
        d[i, NOMES.index(c)] = rng.uniform(0, 0.3)
    return d


@pytest.mark.parametrize("semente", range(40))
def test_leitura_certa_continua_igual(semente):
    rng = np.random.default_rng(semente)
    certo = Cubo.resolvido().aplica(embaralhar(25, random.Random(semente))).texto()
    assert corrigir(_dist(certo, rng), NOMES) == certo


@pytest.mark.parametrize("semente", range(40))
def test_corrige_um_adesivo_errado(semente):
    rng = np.random.default_rng(semente)
    certo = Cubo.resolvido().aplica(embaralhar(25, random.Random(semente))).texto()
    d = _dist(certo, rng)
    i = int(rng.choice([k for k in range(54) if k % 9 != 4]))
    errada = (NOMES.index(certo[i]) + int(rng.integers(1, 6))) % 6
    d[i, errada] = 0.0
    d[i, NOMES.index(certo[i])] = 0.5
    saida = corrigir(d, NOMES)
    assert checar_pecas(Cubo.de_texto(saida)) is None
    assert saida == certo


def test_sempre_valido_mesmo_com_lixo():
    rng = np.random.default_rng(1)
    for _ in range(20):
        saida = corrigir(rng.uniform(0, 1, (54, 6)), NOMES)
        assert checar_pecas(Cubo.de_texto(saida)) is None
