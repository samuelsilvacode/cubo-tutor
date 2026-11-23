"""Solução curta (duas fases de Kociemba). Tabelas geradas em twophase/ na 1ª execução."""
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
        os.chdir(RAIZ)  # o pacote acha as tabelas pela pasta atual
        try:
            import twophase.solver as sv
            _solver = sv
        finally:
            os.chdir(antes)
    return _solver


def tabelas_prontas() -> bool:
    return (RAIZ / "twophase" / "phase2_edgemerge").exists()


def resolver_curto(cubo: Cubo, tempo: float = 2.0) -> list[str]:
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
    if "undefined" in msg:
        return "tem uma peça com combinação de cores que não existe"
    if "9 facelets" in msg:
        return "alguma cor está faltando ou sobrando"
    return msg


def checar_pecas(cubo: Cubo) -> str | None:
    """Confere torção de cantos, meios virados e paridade sem carregar as tabelas."""
    antes = os.getcwd()
    os.chdir(RAIZ)
    try:
        from twophase.face import FaceCube
    finally:
        os.chdir(antes)
    mapa = {cubo.centro(f): f for f in ORDEM_FACES}
    fc = FaceCube()
    res = fc.from_string("".join(mapa[c] for c in cubo.texto()))
    if res is not True:
        return _traduz_erro(res)
    res = fc.to_cubie_cube().verify()
    return None if res is True else _traduz_erro(res)
