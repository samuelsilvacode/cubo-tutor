"""Solução curta (algoritmo de duas fases de Kociemba, ~20 giros).

Usa o pacote RubikTwoPhase. Na primeira execução ele gera tabelas (~70 MB,
pode levar de 20 a 30 minutos) na pasta `twophase/` da raiz do projeto.
"""
from __future__ import annotations

import os
from pathlib import Path

from .modelo import Cubo, ORDEM_FACES

RAIZ = Path(__file__).resolve().parent.parent
_solver = None


def _carrega():
    global _solver
    if _solver is None:
        antes = os.getcwd()
        os.chdir(RAIZ)  # o pacote lê/gera as tabelas relativas à pasta atual
        try:
            import twophase.solver as sv
            _solver = sv
        finally:
            os.chdir(antes)
    return _solver


def tabelas_prontas() -> bool:
    return (RAIZ / "twophase" / "phase2_edgemerge").exists()


def resolver_curto(cubo: Cubo, tempo: float = 2.0) -> list[str]:
    """Devolve a solução em notação padrão (R, U', F2...)."""
    mapa = {cubo.centro(f): f for f in ORDEM_FACES}
    texto = "".join(mapa[c] for c in cubo.texto())
    res = _carrega().solve(texto, 0, tempo)
    if res.startswith("Error"):
        raise ValueError(_traduz_erro(res))
    movs = []
    for m in res.split():
        if m.startswith("("):
            continue
        face, n = m[0], int(m[1])
        movs.append(face + {1: "", 2: "2", 3: "'"}[n])
    return movs


def _traduz_erro(msg: str) -> str:
    if "corner" in msg and "twist" in msg:
        return "um canto está torcido (leitura errada de cor ou peça girada à mão)"
    if "edge" in msg and "flip" in msg:
        return "um meio está virado (leitura errada de cor ou peça trocada à mão)"
    if "parity" in msg:
        return "duas peças parecem trocadas de lugar (provável erro de leitura)"
    if "exactly once" in msg or "9 facelets" in msg:
        return "alguma cor está faltando ou sobrando"
    return msg
