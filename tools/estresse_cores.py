"""Mede quantos cubos são lidos 100% certos com luz ruim e cores parecidas."""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from cubo.modelo import Cubo, ORDEM_FACES, embaralhar  # noqa: E402
from cubo import visao  # noqa: E402

# cubo "difícil": vermelho e laranja próximos, amarelo puxado pro laranja
DIFICIL = {"W": (215, 220, 215), "Y": (35, 185, 230), "G": (70, 150, 25),
           "B": (150, 70, 15), "R": (35, 35, 175), "O": (25, 80, 215)}


def rodada(rng, semente):
    real = Cubo.resolvido().aplica(embaralhar(25, random.Random(semente)))
    tinta = np.array([rng.uniform(0.65, 1.0), rng.uniform(0.85, 1.05), rng.uniform(1.0, 1.25)])
    faces = []
    for f in ORDEM_FACES:
        exp = rng.uniform(0.6, 1.15)
        face = []
        for c in real.face(f):
            v = np.array(DIFICIL[c]) * tinta * exp * rng.uniform(0.9, 1.1) + rng.normal(0, 5, 3)
            face.append(np.clip(v, 0, 255).astype(np.uint8))
        faces.append(face)
    cruas, certas, _ = visao.ler_cubo(faces)
    t = real.texto()
    return cruas == t, certas == t


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(0)
    r = [rodada(rng, i) for i in range(n)]
    print(f"leitura crua 100% certa: {sum(a for a, _ in r) / n:.0%}")
    print(f"depois da correção:      {sum(b for _, b in r) / n:.0%}")
