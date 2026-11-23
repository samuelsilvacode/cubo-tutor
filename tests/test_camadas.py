import random

import pytest

from cubo.modelo import Cubo, embaralhar, inverte
from cubo.camadas import resolver


def test_movimentos_voltam():
    c = Cubo.resolvido()
    s = embaralhar(40, random.Random(1))
    assert c.aplica(s).aplica(inverte(s)).resolvido_ok()


def test_texto_ida_e_volta():
    c = Cubo.resolvido().aplica(embaralhar(30, random.Random(2)))
    assert Cubo.de_texto(c.texto()) == c


@pytest.mark.parametrize("semente", range(300))
def test_resolve_embaralhado(semente):
    rng = random.Random(semente)
    c = Cubo.resolvido().aplica(embaralhar(25, rng))
    etapas = resolver(c)
    assert len(etapas) == 8
    final = c.copia()
    for e in etapas:
        final.aplica(e.movimentos)
    assert final.resolvido_ok()


def test_ja_resolvido():
    etapas = resolver(Cubo.resolvido())
    assert sum(len(e.movimentos) for e in etapas) == 0
