"""Modelo do cubo 3x3.

Cada adesivo guarda sua posição 3D (x, y, z em -1..1), a direção para onde
aponta (normal) e a cor. Girar uma camada é só rotacionar os vetores dos
adesivos daquela camada, o que deixa o código curto e fácil de conferir.

Eixos: x aponta para a direita (R), y para cima (U), z para a frente (F).
Cores são letras: W branco, Y amarelo, G verde, B azul, R vermelho, O laranja.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

Vec = tuple[int, int, int]

# Normal de cada face
NORMAIS: dict[str, Vec] = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "R": (1, 0, 0), "L": (-1, 0, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
}
FACE_DA_NORMAL = {v: k for k, v in NORMAIS.items()}

# Ordem das faces no formato de texto padrão (Kociemba): U R F D L B
ORDEM_FACES = "URFDLB"


def _rot(v: Vec, eixo: str, quartos: int) -> Vec:
    """Rotaciona v em torno do eixo, 90° por quarto, sentido anti-horário (regra da mão direita)."""
    x, y, z = v
    for _ in range(quartos % 4):
        if eixo == "x":
            x, y, z = x, -z, y
        elif eixo == "y":
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


def _eixo_sinal(n: Vec) -> tuple[str, int]:
    for i, eixo in enumerate("xyz"):
        if n[i]:
            return eixo, n[i]
    raise ValueError(n)


def grade_da_face(face: str) -> list[tuple[Vec, Vec]]:
    """Posições dos 9 adesivos de uma face na ordem de leitura padrão.

    A ordem é a da planificação clássica: U vista de cima com F embaixo,
    D vista de baixo com F em cima, faces laterais vistas de frente com U em cima.
    """
    n = NORMAIS[face]
    # vetor "direita" e "baixo" da face como vista na planificação
    dirs = {
        "U": ((1, 0, 0), (0, 0, 1)),
        "D": ((1, 0, 0), (0, 0, -1)),
        "F": ((1, 0, 0), (0, -1, 0)),
        "B": ((-1, 0, 0), (0, -1, 0)),
        "R": ((0, 0, -1), (0, -1, 0)),
        "L": ((0, 0, 1), (0, -1, 0)),
    }
    dx, dy = dirs[face]
    out = []
    for lin in (-1, 0, 1):
        for col in (-1, 0, 1):
            pos = tuple(n[i] + col * dx[i] + lin * dy[i] for i in range(3))
            out.append((pos, n))
    return out


@dataclass(frozen=True)
class Adesivo:
    pos: Vec
    normal: Vec
    cor: str


class Cubo:
    def __init__(self, adesivos: dict[tuple[Vec, Vec], str]):
        # chave (pos, normal) -> cor
        self.a = dict(adesivos)

    # ---------- construção ----------
    @classmethod
    def resolvido(cls, cores: str = "YOGWRB") -> "Cubo":
        """Cubo montado. `cores` dá a cor de U R F D L B (padrão: amarelo em cima, verde na frente)."""
        return cls.de_texto("".join(c * 9 for c in cores))

    @classmethod
    def de_texto(cls, s: str) -> "Cubo":
        """Lê 54 letras na ordem U R F D L B (9 por face, ordem de leitura)."""
        s = s.replace(" ", "")
        if len(s) != 54:
            raise ValueError(f"esperava 54 adesivos, recebi {len(s)}")
        a = {}
        for fi, face in enumerate(ORDEM_FACES):
            for k, chave in enumerate(grade_da_face(face)):
                a[chave] = s[fi * 9 + k]
        return cls(a)

    def texto(self) -> str:
        return "".join(self.cor(c) for face in ORDEM_FACES for c in grade_da_face(face))

    def face(self, face: str) -> list[str]:
        return [self.a[c] for c in grade_da_face(face)]

    def copia(self) -> "Cubo":
        return Cubo(self.a)

    def cor(self, chave: tuple[Vec, Vec]) -> str:
        return self.a[chave]

    def centro(self, face: str) -> str:
        n = NORMAIS[face]
        return self.a[(n, n)]

    def __eq__(self, outro: object) -> bool:
        return isinstance(outro, Cubo) and self.a == outro.a

    def __hash__(self) -> int:
        return hash(self.texto())

    # ---------- movimentos ----------
    def _gira(self, filtro, eixo: str, quartos: int) -> None:
        novo = {}
        for (pos, nor), cor in self.a.items():
            if filtro(pos):
                novo[(_rot(pos, eixo, quartos), _rot(nor, eixo, quartos))] = cor
            else:
                novo[(pos, nor)] = cor
        self.a = novo

    def aplica(self, seq: str | list[str]) -> "Cubo":
        """Aplica movimentos em notação padrão: R U R' U2, e rotações x y z."""
        movs = seq.split() if isinstance(seq, str) else seq
        for m in movs:
            base, suf = m[0], m[1:]
            vezes = {"": 1, "'": 3, "2": 2, "2'": 2}[suf]
            if base in "xyz":
                face = {"x": "R", "y": "U", "z": "F"}[base]
                eixo, sinal = _eixo_sinal(NORMAIS[face])
                self._gira(lambda p: True, eixo, (-sinal * vezes) % 4)
            elif base in NORMAIS:
                n = NORMAIS[base]
                eixo, sinal = _eixo_sinal(n)
                i = "xyz".index(eixo)
                self._gira(lambda p, i=i, s=sinal: p[i] == s, eixo, (-sinal * vezes) % 4)
            else:
                raise ValueError(f"movimento desconhecido: {m}")
        return self

    # ---------- consultas ----------
    def pecas(self) -> dict[Vec, dict[Vec, str]]:
        """Agrupa adesivos por peça: posição -> {normal: cor}."""
        out: dict[Vec, dict[Vec, str]] = {}
        for (pos, nor), cor in self.a.items():
            out.setdefault(pos, {})[nor] = cor
        return out

    def acha(self, cores: set[str]) -> tuple[Vec, dict[Vec, str]]:
        """Posição atual da peça (meio ou canto) com exatamente essas cores."""
        for pos, ads in self.pecas().items():
            if len(ads) == len(cores) and set(ads.values()) == cores:
                return pos, ads
        raise ValueError(f"peça {cores} não existe")

    def resolvido_ok(self) -> bool:
        return all(len(set(self.face(f))) == 1 for f in ORDEM_FACES)


def inverte(seq: str) -> str:
    out = []
    for m in reversed(seq.split()):
        if m.endswith("'"):
            out.append(m[:-1])
        elif m.endswith("2"):
            out.append(m)
        else:
            out.append(m + "'")
    return " ".join(out)


def embaralhar(n: int = 25, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    out, ult = [], ""
    for _ in range(n):
        f = rng.choice([f for f in "URFDLB" if f != ult])
        out.append(f + rng.choice(["", "'", "2"]))
        ult = f
    return " ".join(out)


def simplifica(movs: list[str]) -> list[str]:
    """Junta giros seguidos da mesma face (U U -> U2, U U' -> nada)."""
    pilha: list[str] = []
    q = {"": 1, "2": 2, "'": 3}
    for m in movs:
        if pilha and pilha[-1][0] == m[0]:
            t = (q[pilha[-1][1:]] + q[m[1:]]) % 4
            pilha.pop()
            if t:
                pilha.append(m[0] + {1: "", 2: "2", 3: "'"}[t])
        else:
            pilha.append(m)
    return pilha


def validar(cubo: Cubo) -> list[str]:
    """Lista problemas no estado lido pela câmera (vazio = estado possível)."""
    erros = []
    cont: dict[str, int] = {}
    for c in cubo.a.values():
        cont[c] = cont.get(c, 0) + 1
    for c, n in sorted(cont.items()):
        if n != 9:
            erros.append(f"a cor {NOME_COR.get(c, c)} aparece {n} vezes (deveria ser 9)")
    if erros:
        return erros
    centros = {cubo.centro(f) for f in ORDEM_FACES}
    if len(centros) != 6:
        return ["dois centros com a mesma cor"]
    ref = Cubo.resolvido("".join(cubo.centro(f) for f in ORDEM_FACES))
    esperadas = {frozenset(ads.values()) for ads in ref.pecas().values() if len(ads) > 1}
    for pos, ads in cubo.pecas().items():
        if len(ads) > 1 and frozenset(ads.values()) not in esperadas:
            erros.append("peça impossível: " + "-".join(NOME_COR.get(c, c) for c in ads.values()))
    if erros:
        return erros
    # paridade e torções: confere com o solver ótimo (que valida isso), se disponível
    return []


NOME_COR = {"W": "branco", "Y": "amarelo", "G": "verde", "B": "azul", "R": "vermelho", "O": "laranja"}
