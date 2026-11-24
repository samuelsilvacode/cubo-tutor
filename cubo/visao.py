"""Leitura das cores pela webcam."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .correcao import corrigir

CORES = "WYGBRO"

# BGR aproximado de cada cor, usado só para desenhar e nomear os centros
BGR = {
    "W": (235, 235, 235), "Y": (0, 215, 255), "G": (72, 155, 0),
    "B": (173, 70, 0), "R": (46, 16, 200), "O": (19, 106, 255),
}


def grade(frame: np.ndarray, frac: float = 0.45) -> list[tuple[int, int, int, int]]:
    """9 retângulos (x, y, w, h) de uma grade centralizada, em ordem de leitura."""
    h, w = frame.shape[:2]
    lado = int(min(h, w) * frac)
    x0, y0, cel = (w - lado) // 2, (h - lado) // 2, lado // 3
    return [(x0 + c * cel, y0 + l * cel, cel, cel) for l in range(3) for c in range(3)]


def amostras(frame: np.ndarray) -> list[np.ndarray]:
    out = []
    for x, y, w, h in grade(frame):
        m = int(w * 0.3)
        pedaco = frame[y + m:y + h - m, x + m:x + w - m].reshape(-1, 3)
        out.append(np.median(pedaco, axis=0).astype(np.uint8))
    return out


def uniforme(frame: np.ndarray) -> bool:
    """Cada quadrado da grade tem uma cor só (tem um cubo encaixado ali)."""
    desv = []
    for x, y, w, h in grade(frame):
        m = int(w * 0.3)
        desv.append(frame[y + m:y + h - m, x + m:x + w - m].reshape(-1, 3).std(axis=0).mean())
    return max(desv) < 22


def parece_face(amostras) -> bool:
    """Evita ler parede ou fundo liso como se fosse uma face."""
    hsv = cv2.cvtColor(np.array(amostras, np.uint8).reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    coloridos = (hsv[:, 1] > 80).sum()
    return coloridos >= 4 and len({cor_rapida(a) for a in amostras}) >= 3


def distancias(todas, centros, wb=None, nomes=None) -> np.ndarray:
    d = np.linalg.norm(_lab(todas, wb)[:, None] - _lab(centros, wb)[None], axis=2)
    if nomes is not None and "R" in nomes and "O" in nomes:
        # a calibração decide entre vermelho e laranja
        r, o = nomes.index("R"), nomes.index("O")
        for i, bgr in enumerate(todas):
            v = vermelho_ou_laranja(bgr, centros[r], centros[o])
            if v == "R":
                d[i, o] += 500
            elif v == "O":
                d[i, r] += 500
    return d


def cor_rapida(bgr: np.ndarray) -> str:
    """Palpite por matiz, para o preview ao vivo antes da calibração."""
    h, s, v = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    if s < 60 and v > 110:
        return "W"
    if h < 5 or h >= 160:
        return "R"
    if h < 19:
        return "O"
    if h < 38:
        return "Y"
    if h < 90:
        return "G"
    if h < 140:
        return "B"
    return "R"


def _lab(bgrs, wb=None) -> np.ndarray:
    """LAB com pouco peso no brilho e muito no tom (é o que separa vermelho, laranja e amarelo)."""
    a = np.array(bgrs, float).reshape(-1, 3)
    if wb is not None:
        a = np.clip(a * wb, 0, 255)
    lab = cv2.cvtColor((a / 255).astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    L, A, B = lab.reshape(-1, 3).astype(float).T
    croma = np.hypot(A, B) + 1e-6
    peso = 250 * np.minimum(croma, 25) / 25  # cor quase branca não tem tom confiável
    return np.stack([0.25 * L, A, B, peso * A / croma, peso * B / croma], axis=1)


def balanco_branco(branco) -> np.ndarray:
    b = np.maximum(np.array(branco, float), 1)
    return b.mean() / b


def nomear_centros(centros: list[np.ndarray]) -> list[str]:
    palpite = [cor_rapida(c) for c in centros]
    if len(set(palpite)) == 6:
        return palpite
    ref = _lab([BGR[c] for c in CORES])
    lab = _lab(centros)
    custo = np.linalg.norm(lab[:, None] - ref[None], axis=2)
    lin, col = linear_sum_assignment(custo)
    out = [""] * 6
    for i, j in zip(lin, col):
        out[i] = CORES[j]
    return out


def classificar(todas, centros, nomes, wb=None) -> list[str]:
    """Classifica os 54 adesivos usando os centros como referência, 9 de cada cor."""
    custo = np.repeat(distancias(todas, centros, wb, nomes), 9, axis=1)
    lin, col = linear_sum_assignment(custo)
    out = [""] * len(todas)
    for i, j in zip(lin, col):
        out[i] = nomes[j // 9]
    return out


class Referencia:
    """Depois do escaneamento: classifica ao vivo pelos centros calibrados."""

    def __init__(self, centros, nomes, wb=None):
        self.wb = wb
        self.centros = [np.array(c, float) for c in centros]
        self.ref = _lab(centros, wb)
        self.nomes = nomes

    def cor(self, bgr: np.ndarray) -> str:
        d = np.linalg.norm(self.ref - _lab([bgr], self.wb)[0], axis=1)
        c = self.nomes[int(np.argmin(d))]
        if c in "RO" and "R" in self.nomes and "O" in self.nomes:
            c = vermelho_ou_laranja(bgr, self.centros[self.nomes.index("R")],
                                    self.centros[self.nomes.index("O")]) or c
        return c


# com amarelo em cima, as laterais em volta seguem essa ordem no esquema padrão
VOLTA = "GOBR"


def nomes_dos_centros(centros) -> list[str]:
    """Centros na ordem URFDLB. U é amarelo e D é branco; as laterais saem da melhor rotação de VOLTA."""
    lab = _lab(centros)
    ref = _lab([BGR[c] for c in VOLTA])
    lados = [2, 1, 5, 4]  # F R B L dentro de URFDLB
    k = min(range(4), key=lambda k: sum(np.linalg.norm(lab[lados[i]] - ref[(i + k) % 4]) for i in range(4)))
    nomes = ["Y", "", "", "W", "", ""]
    for i, idx in enumerate(lados):
        nomes[idx] = VOLTA[(i + k) % 4]
    return nomes


def ler_cubo(faces: list[list[np.ndarray]]):
    """faces: 6 listas de 9 amostras BGR na ordem URFDLB. Devolve (leitura crua, leitura corrigida, Referencia)."""
    todas = [a for f in faces for a in f]
    centros = [f[4] for f in faces]
    wb = balanco_branco(faces[3][4])  # centro da base é branco
    nomes = nomes_dos_centros([np.clip(np.array(c, float) * wb, 0, 255) for c in centros])
    cruas = "".join(classificar(todas, centros, nomes, wb))
    certas = corrigir(distancias(todas, centros, wb), nomes)
    return cruas, certas, Referencia(centros, nomes, wb)


def _hsv(bgr):
    return cv2.cvtColor(np.clip(np.array(bgr, float), 0, 255).astype(np.uint8).reshape(1, 1, 3),
                        cv2.COLOR_BGR2HSV)[0, 0].astype(float)


def _quanto_laranja(bgr) -> float:
    b, g, r = np.array(bgr, float)
    return (g - b) / max(r, 1)


def confere_centro(pedida: str, centro, lidos: dict) -> tuple[str | None, str | None]:
    """Confere se o centro mostrado é da cor pedida.

    Devolve (mensagem de problema ou None, cor já lida que precisa ser refeita ou None).
    Branco/amarelo e vermelho/laranja são comparados entre si, porque a câmera confunde.
    """
    from .modelo import NOME_COR
    vista = cor_rapida(np.clip(np.array(centro), 0, 255).astype(np.uint8))
    familia = {"W": "WY", "Y": "YW", "R": "RO", "O": "OR", "G": "G", "B": "B"}[pedida]
    if vista not in familia:
        return f"O centro dessa face parece {NOME_COR[vista]}. Mostre a do centro {NOME_COR[pedida]}.", None
    par = {"W": "Y", "Y": "W", "R": "O", "O": "R"}.get(pedida)
    if par is None or par not in lidos:
        if pedida == "W" and _hsv(centro)[1] > 110:
            return "Esse centro está colorido demais para ser branco. Mostre a do centro branco.", None
        return None, None
    if pedida in "WY":
        sat_nova, sat_par = _hsv(centro)[1], _hsv(lidos[par])[1]
        mais_colorido = sat_nova > sat_par
        if (pedida == "Y") != mais_colorido or abs(sat_nova - sat_par) < 25:
            return (f"Esse centro não parece {NOME_COR[pedida]}: está muito parecido com o "
                    f"{NOME_COR[par]} que você já mostrou. Talvez os dois estejam trocados: "
                    f"mostre de novo a do centro {NOME_COR[par]}."), par
        return None, None
    lar_nova, lar_par = _quanto_laranja(centro), _quanto_laranja(lidos[par])
    if (pedida == "O") != (lar_nova > lar_par):
        return (f"Esse centro parece mais {NOME_COR[par]} do que {NOME_COR[pedida]}, comparando com a face "
                f"que você mostrou antes. Talvez as duas estejam trocadas: mostre de novo a do centro "
                f"{NOME_COR[par]}."), par
    return None, None


# ---------- calibração do vermelho e do laranja (tools/calibrar.py) ----------
CALIBRACAO = Path(__file__).resolve().parent.parent / "calibracao" / "cores.json"
_limiar_cache: list = []


def _proporcao_laranja(bgr) -> float:
    """Quanto verde vem junto com o vermelho: separa laranja de vermelho mesmo com o vermelho estourado."""
    b, g, r = np.array(bgr, float)
    return (g - b) / max(r, 1.0)


def _parece_vermelho_ou_laranja(bgr) -> bool:
    b, g, r = np.array(bgr, float)
    return r > 120 and r > 1.6 * g and r > 1.6 * b


def limiar_vermelho_laranja() -> float | None:
    if not _limiar_cache:
        limiar = None
        if CALIBRACAO.exists():
            dados = json.loads(CALIBRACAO.read_text(encoding="utf8"))
            vermelhos = [_proporcao_laranja(a) for a in dados.get("vermelho", []) if _parece_vermelho_ou_laranja(a)]
            laranjas = [_proporcao_laranja(a) for a in dados.get("laranja", []) if _parece_vermelho_ou_laranja(a)]
            if vermelhos and laranjas and max(vermelhos) < min(laranjas):
                limiar = (max(vermelhos) + min(laranjas)) / 2
        _limiar_cache.append(limiar)
    return _limiar_cache[0]


def vermelho_ou_laranja(bgr, centro_r=None, centro_o=None) -> str | None:
    """'R' ou 'O' pela proporção de verde. O limite é o meio do caminho entre o centro vermelho
    e o centro laranja da própria leitura (a webcam muda o brilho sozinha, então valor fixo não serve).
    Sem os centros, usa o limite da calibração, se houver."""
    if not _parece_vermelho_ou_laranja(bgr):
        return None
    if centro_r is not None and centro_o is not None:
        pr, po = _proporcao_laranja(centro_r), _proporcao_laranja(centro_o)
        if po - pr < 0.04:
            return None  # centros parecidos demais: não dá pra confiar
        limiar = (pr + po) / 2
        # perto do meio não decide: deixa a comparação normal de cores resolver
        if abs(_proporcao_laranja(bgr) - limiar) < 0.25 * (po - pr):
            return None
    else:
        limiar = limiar_vermelho_laranja()
        if limiar is None:
            return None
    return "O" if _proporcao_laranja(bgr) > limiar else "R"
