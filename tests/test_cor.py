import numpy as np

from cubo.cor import ciede2000

# pares de teste do artigo de Sharma, Wu e Dalal (2005)
PARES = [((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
         ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
         ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
         ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]


def test_ciede2000_bate_com_o_artigo():
    for a, b, esperado in PARES:
        assert abs(ciede2000(np.array([a]), np.array([b]))[0, 0] - esperado) < 1e-3
