"""Solver pelo método de camadas, passo a passo, com amarelo em cima."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .modelo import Cubo, NORMAIS, FACE_DA_NORMAL, simplifica

FACES_LADO = ["F", "R", "B", "L"]
MOV_FACE = [f + s for f in "URFDLB" for s in ("", "'", "2")]
SETUP_U = ["", "U", "U2", "U'"]
SETUP_Y = ["", "y", "y2", "y'"]

ALG_CANTO = "R U R' U'"
ALG_MEIO_DIR = "U R U' R' U' F' U F"
ALG_MEIO_ESQ = "U' L' U L U F U' F'"
ALG_CRUZ = "F R U R' U' F'"
ALG_ALINHA = "R U R' U R U2 R' U"
ALG_CANTOS_POS = "U R U' L' U R' U' L"
ALG_CANTOS_VIRA = "R' D' R D"


@dataclass
class Passo:
    movs: list[str]
    dica: str


@dataclass
class Etapa:
    numero: int
    nome: str
    objetivo: str
    porque: str
    passos: list[Passo] = field(default_factory=list)

    @property
    def movimentos(self) -> list[str]:
        return [m for p in self.passos for m in p.movs]


class _Gravador:
    def __init__(self, cubo: Cubo):
        self.c = cubo.copia()
        self.etapas: list[Etapa] = []

    def etapa(self, *args) -> Etapa:
        e = Etapa(len(self.etapas) + 1, *args)
        self.etapas.append(e)
        return e

    def faz(self, seq: str, dica: str) -> None:
        movs = simplifica(seq.split())
        if not movs:
            return
        self.c.aplica(movs)
        self.etapas[-1].passos.append(Passo(movs, dica))


def _cor_frente(c: Cubo, pos, normal) -> str:
    return c.a[(pos, normal)]


def _meio_resolvido(c: Cubo, pos) -> bool:
    ads = c.pecas()[pos]
    return all(cor == c.centro(FACE_DA_NORMAL[n]) for n, cor in ads.items())


def _petalas(c: Cubo) -> int:
    return sum(1 for f in FACES_LADO
               if c.a[(_soma(NORMAIS["U"], NORMAIS[f]), NORMAIS["U"])] == "W")


def _soma(*vs):
    return tuple(sum(v[i] for v in vs) for i in range(3))


def _cruz_branca_ok(c: Cubo) -> bool:
    return all(_meio_resolvido(c, _soma(NORMAIS["D"], NORMAIS[f])) for f in FACES_LADO)


def _cantos_d_ok(c: Cubo) -> int:
    return sum(1 for f1, f2 in _pares_lado()
               if _meio_resolvido(c, _soma(NORMAIS["D"], NORMAIS[f1], NORMAIS[f2])))


def _meios_meio_ok(c: Cubo) -> int:
    return sum(1 for f1, f2 in _pares_lado() if _meio_resolvido(c, _soma(NORMAIS[f1], NORMAIS[f2])))


def _pares_lado():
    return [("F", "R"), ("R", "B"), ("B", "L"), ("L", "F")]


def _amarelos_cruz(c: Cubo) -> set[str]:
    u = NORMAIS["U"]
    return {f for f in FACES_LADO if c.a[(_soma(u, NORMAIS[f]), u)] == "Y"}


def _alinhados_topo(c: Cubo) -> set[str]:
    u = NORMAIS["U"]
    return {f for f in FACES_LADO
            if c.a[(_soma(u, NORMAIS[f]), NORMAIS[f])] == c.centro(f)}


def _canto_no_lugar(c: Cubo, pos) -> bool:
    ads = c.pecas()[pos]
    return set(ads.values()) == {c.centro(FACE_DA_NORMAL[n]) for n in ads}


def _testa(c: Cubo, seq: str) -> Cubo:
    return c.copia().aplica(seq.split()) if seq.strip() else c.copia()


def _margarida(g: _Gravador) -> None:
    g.etapa("Margarida",
            "Deixar os 4 meios brancos em volta do centro amarelo, com o branco virado pra cima.",
            "Em volta do amarelo dá pra girar à vontade sem estragar nada. É um estacionamento "
            "para as peças brancas antes de levar cada uma para o lugar certo.")
    if _cruz_branca_ok(g.c):
        return
    while _petalas(g.c) < 4:
        alvo = _petalas(g.c) + 1
        seq = _busca(g.c, lambda c: _petalas(c) >= alvo, prof_max=4)
        g.faz(" ".join(seq), "Leve mais um meio branco para perto do amarelo, com o branco pra cima. "
                             "Se a vaga estiver ocupada, gire o U antes para não derrubar uma pétala.")


def _busca(c: Cubo, objetivo, prof_max: int) -> list[str]:
    for prof in range(1, prof_max + 1):
        for seq in product(MOV_FACE, repeat=prof):
            if any(seq[i][0] == seq[i + 1][0] for i in range(prof - 1)):
                continue
            if objetivo(_testa(c, " ".join(seq))):
                return list(seq)
    raise RuntimeError("busca não achou solução")


def _cruz_branca(g: _Gravador) -> None:
    g.etapa("Cruz branca",
            "Descer cada pétala para baixo, encaixada no centro da mesma cor.",
            "Girando o U, a cor lateral da pétala fica em cima do centro igual. "
            "O giro duplo desce a peça reto, sem tirar ela do lado desse centro.")
    u = NORMAIS["U"]
    while _petalas(g.c) > 0:
        feito = False
        for s in SETUP_U:
            t = _testa(g.c, s)
            for f in FACES_LADO:
                pos = _soma(u, NORMAIS[f])
                if t.a[(pos, u)] == "W" and t.a[(pos, NORMAIS[f])] == t.centro(f):
                    nome = t.centro(f)
                    g.faz(f"{s} {f}2", f"Gire o U até a pétala com {_nome(nome)} ficar em cima do centro "
                                       f"{_nome(nome)}, depois faça {f}2 para descer.")
                    feito = True
                    break
            if feito:
                break


def _cantos_brancos(g: _Gravador) -> None:
    g.etapa("Cantos brancos",
            "Completar a face branca e fechar a primeira camada.",
            "R U R' U' desce o canto, gira ele e devolve o que tirou do lugar. Repetido 6 vezes "
            "volta tudo ao normal, então repetir até encaixar não estraga a cruz.")
    while _cantos_d_ok(g.c) < 4:
        cantos_topo = [(pos, ads) for pos, ads in g.c.pecas().items()
                       if len(ads) == 3 and pos[1] == 1 and "W" in ads.values()]
        if cantos_topo:
            cores = set(cantos_topo[0][1].values())
            outras = cores - {"W"}
            for y in SETUP_Y:
                t = _testa(g.c, y)
                if {t.centro("F"), t.centro("R")} == outras:
                    break
            g.faz(y, "Gire o cubo inteiro para a vaga do canto ficar na frente-direita.")
            for s in SETUP_U:
                t = _testa(g.c, s)
                if t.acha(cores)[0] == (1, 1, 1):
                    break
            g.faz(s, f"Gire o U até o canto {_nomes(cores)} ficar em cima da vaga (frente-direita).")
            vezes = 0
            while not _meio_resolvido(g.c, (1, -1, 1)):
                g.faz(ALG_CANTO, "Repita R U R' U' até o canto encaixar com o branco pra baixo.")
                vezes += 1
                assert vezes <= 6
        else:
            for y in SETUP_Y:
                t = _testa(g.c, y)
                if not _meio_resolvido(t, (1, -1, 1)):
                    break
            g.faz(y, "Gire o cubo para o canto branco preso ficar na frente-direita, embaixo.")
            g.faz(ALG_CANTO, "Faça R U R' U' uma vez para tirar o canto de lá e levar ele pra cima.")


def _segunda_camada(g: _Gravador) -> None:
    g.etapa("Segunda camada",
            "Colocar os 4 meios da camada do meio.",
            "A primeira metade do algoritmo tira do caminho o canto de baixo e junta ele com o meio. "
            "A segunda metade devolve os dois juntos. A face branca sai e volta inteira.")
    voltas = 0
    while _meios_meio_ok(g.c) < 4:
        voltas += 1
        assert voltas < 20
        u = NORMAIS["U"]
        achou = None
        for y in SETUP_Y:
            for s in SETUP_U:
                t = _testa(g.c, f"{y} {s}")
                pos = (0, 1, 1)
                topo, frente = t.a[(pos, u)], t.a[(pos, NORMAIS["F"])]
                if "Y" not in (topo, frente) and frente == t.centro("F"):
                    achou = (y, s, topo, t)
                    break
            if achou:
                break
        if achou:
            y, s, topo, t = achou
            g.faz(y, "Gire o cubo inteiro para trabalhar com outra face na frente.")
            g.faz(s, f"Gire o U até o meio formar um T de cabeça pra baixo com o centro {_nome(t.centro('F'))}.")
            if topo == t.centro("R"):
                g.faz(ALG_MEIO_DIR, f"A cor de cima é {_nome(topo)}, igual ao centro da direita: algoritmo da direita.")
            else:
                g.faz(ALG_MEIO_ESQ, f"A cor de cima é {_nome(topo)}, igual ao centro da esquerda: algoritmo da esquerda.")
        else:
            for y in SETUP_Y:
                t = _testa(g.c, y)
                if not _meio_resolvido(t, (1, 0, 1)):
                    break
            g.faz(y, "Gire o cubo para o meio preso ficar na frente-direita.")
            g.faz(ALG_MEIO_DIR, "Faça o algoritmo da direita para tirar o meio preso e levar ele pra cima.")


def _cruz_amarela(g: _Gravador) -> None:
    g.etapa("Cruz amarela",
            "Fazer uma cruz amarela em cima, sem se preocupar com as laterais.",
            "F tira um pedaço da frente do caminho, R U R' U' vira os meios de cima e F' devolve "
            "a frente. As duas primeiras camadas voltam iguais.")
    voltas = 0
    while len(_amarelos_cruz(g.c)) < 4:
        voltas += 1
        assert voltas < 6
        am = _amarelos_cruz(g.c)
        if len(am) == 2:
            for s in SETUP_U:
                a2 = _amarelos_cruz(_testa(g.c, s))
                if a2 == {"L", "R"}:
                    g.faz(s, "É uma linha: gire o U até ela ficar deitada (esquerda-direita).")
                    break
                if a2 == {"B", "L"}:
                    g.faz(s, "É um L: gire o U até as pontas ficarem para trás e para a esquerda.")
                    break
        nome = {0: "ponto", 2: "L ou linha"}.get(len(am), "")
        g.faz(ALG_CRUZ, f"Faça F R U R' U' F' (formato atual: {nome}).")


def _alinha_cruz(g: _Gravador) -> None:
    g.etapa("Alinhar a cruz",
            "Cada meio amarelo combinar com o centro do lado.",
            "O algoritmo troca o meio da frente com o da esquerda e devolve as duas primeiras "
            "camadas. Por isso os que já estão certos ficam atrás e à direita, fora da troca.")
    voltas = 0
    while True:
        voltas += 1
        assert voltas < 6
        melhor = max(SETUP_U, key=lambda s: len(_alinhados_topo(_testa(g.c, s))))
        g.faz(melhor, "Gire o U até o máximo de meios combinarem com os centros dos lados.")
        al = _alinhados_topo(g.c)
        if len(al) == 4:
            return
        if al in ({"F", "B"}, {"L", "R"}):
            g.faz(ALG_ALINHA, "Os certos estão um na frente do outro: faça o algoritmo uma vez de qualquer jeito.")
            continue
        for y in SETUP_Y:
            if _alinhados_topo(_testa(g.c, y)) >= {"B", "R"}:
                break
        g.faz(y, "Gire o cubo inteiro para os 2 meios certos ficarem atrás e à direita.")
        g.faz(ALG_ALINHA, "Faça R U R' U R U2 R' U.")


def _posiciona_cantos(g: _Gravador) -> None:
    g.etapa("Posicionar cantos",
            "Cada canto amarelo no lugar certo, mesmo que ainda esteja virado.",
            "O algoritmo gira os outros 3 cantos de cima em roda e deixa parado o da "
            "frente-direita. Por isso o que já está certo fica ali.")
    topo = [(1, 1, 1), (-1, 1, 1), (1, 1, -1), (-1, 1, -1)]
    voltas = 0
    while not all(_canto_no_lugar(g.c, p) for p in topo):
        voltas += 1
        assert voltas < 6
        certos = [p for p in topo if _canto_no_lugar(g.c, p)]
        if certos:
            for y in SETUP_Y:
                if _canto_no_lugar(_testa(g.c, y), (1, 1, 1)):
                    break
            g.faz(y, "Gire o cubo inteiro para o canto que já está no lugar ficar na frente-direita, em cima.")
            g.faz(ALG_CANTOS_POS, "Faça U R U' L' U R' U' L.")
        else:
            g.faz(ALG_CANTOS_POS, "Nenhum canto no lugar: faça o algoritmo uma vez e procure de novo.")


def _vira_cantos(g: _Gravador) -> None:
    g.etapa("Virar cantos",
            "Virar os cantos amarelos pra cima e terminar o cubo.",
            "R' D' R D feito 6 vezes volta tudo ao normal. Cada canto precisa de 2 ou 4, e o total "
            "sempre dá múltiplo de 6. Por isso a bagunça de baixo se desfaz sozinha no fim.")
    ufr, u = (1, 1, 1), NORMAIS["U"]
    cantos = [(1, 1, 1), (-1, 1, 1), (1, 1, -1), (-1, 1, -1)]
    falta = any(g.c.a[(p, u)] != "Y" for p in cantos)
    for _ in range(4 if falta else 0):
        n = 0
        while g.c.a[(ufr, u)] != "Y":
            g.faz(ALG_CANTOS_VIRA, "Repita R' D' R D até o amarelo desse canto ficar pra cima. "
                                   "As camadas de baixo vão parecer bagunçadas: é normal.")
            n += 1
            assert n <= 4
        g.faz("U", "Gire só o U para trazer o próximo canto para a frente-direita.")
    for s in SETUP_U:
        if _testa(g.c, s).resolvido_ok():
            g.faz(s, "Gire o U para alinhar a última camada.")
            break


def _nome(c: str) -> str:
    from .modelo import NOME_COR
    return NOME_COR.get(c, c)


def _nomes(cores) -> str:
    return "-".join(_nome(c) for c in sorted(cores, key="WYGBRO".index))


def resolver(cubo: Cubo) -> list[Etapa]:
    if cubo.centro("U") != "Y" or cubo.centro("D") != "W":
        raise ValueError("segure o cubo com o amarelo em cima e o branco embaixo")
    g = _Gravador(cubo)
    for f in (_margarida, _cruz_branca, _cantos_brancos, _segunda_camada,
              _cruz_amarela, _alinha_cruz, _posiciona_cantos, _vira_cantos):
        f(g)
    if not g.c.resolvido_ok():
        raise RuntimeError("o solver terminou sem montar o cubo (estado lido deve ter erro)")
    return g.etapas
