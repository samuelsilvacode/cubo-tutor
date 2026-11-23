import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from estresse_cores import rodada  # noqa: E402


def test_luz_quente_e_cores_parecidas():
    rng = np.random.default_rng(42)
    r = [rodada(rng, i) for i in range(80)]
    assert sum(ok for _, ok in r) / len(r) >= 0.97
