from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import detector, leitura, render3d, visao
from .mao import Maos
from .pose import ESCALA, Rastreador
from .registro import Registro
from .camadas import resolver
from .modelo import Cubo, ORDEM_FACES, NOME_COR, validar
from .otimo import checar_pecas, resolver_curto, tabelas_prontas

LARG, ALT = 1360, 720
CAM_W, CAM_H = 800, 450
PAINEL_X = CAM_W + 20

DESC = {
    "R": "direita sobe", "R'": "direita desce", "R2": "direita 2 vezes",
    "L": "esquerda desce", "L'": "esquerda sobe", "L2": "esquerda 2 vezes",
    "U": "cima pra esquerda", "U'": "cima pra direita", "U2": "cima 2 vezes",
    "D": "baixo pra direita", "D'": "baixo pra esquerda", "D2": "baixo 2 vezes",
    "F": "frente no sentido do relógio", "F'": "frente contra o relógio", "F2": "frente 2 vezes",
    "B": "trás (o topo dela vai pra esquerda)", "B'": "trás (o topo dela vai pra direita)", "B2": "trás 2 vezes",
    "y": "gire o cubo inteiro: a direita vem pra frente",
    "y'": "gire o cubo inteiro: a esquerda vem pra frente",
    "y2": "meia volta no cubo inteiro",
    "U2": "cima 2 vezes",
}

PEDIR = "WYGBRO"

# posição de cada face na planificação (coluna, linha)
NET = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}

_FONTES = {}


def fonte(tam: int, negrito=False):
    chave = (tam, negrito)
    if chave not in _FONTES:
        nome = "segoeuib.ttf" if negrito else "segoeui.ttf"
        for p in (Path("C:/Windows/Fonts") / nome, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
            if p.exists():
                _FONTES[chave] = ImageFont.truetype(str(p), tam)
                break
        else:
            _FONTES[chave] = ImageFont.load_default()
    return _FONTES[chave]


class Tela:
    """Junta os textos e desenha tudo de uma vez com PIL (o putText não tem acento)."""

    def __init__(self):
        self.img = np.full((ALT, LARG, 3), 30, np.uint8)
        self.textos = []

    def t(self, x, y, s, tam=20, cor=(230, 230, 230), negrito=False, larg=None):
        if larg:
            s = quebra(s, fonte(tam, negrito), larg)
        self.textos.append((x, y, s, tam, cor, negrito))
        return y + (s.count("\n") + 1) * int(tam * 1.35)

    def final(self):
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, s, tam, cor, neg in self.textos:
            d.multiline_text((x, y), s, font=fonte(tam, neg), fill=cor[::-1], spacing=int(tam * 0.35))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def quebra(s, f, larg):
    linhas = []
    for par in s.split("\n"):
        atual = ""
        for p in par.split():
            teste = (atual + " " + p).strip()
            if f.getlength(teste) > larg and atual:
                linhas.append(atual)
                atual = p
            else:
                atual = teste
        linhas.append(atual)
    return "\n".join(linhas)


FACE_CAMERA = "B"


def iguais(a, b) -> int:
    return sum(x == y for x, y in zip(a, b))


def posicao_errada(lida, esperado, cubo: Cubo) -> str | None:
    """Diz como corrigir a posição do cubo na mão, olhando só o lado que a câmera vê."""
    if iguais(lida, esperado) >= 7:
        return None
    if lida[4] != esperado[4]:
        oposto = {cubo.centro(a): cubo.centro(b) for a, b in
                  (("U", "D"), ("D", "U"), ("F", "B"), ("B", "F"), ("R", "L"), ("L", "R"))}
        voce = NOME_COR[oposto[lida[4]]]
        return (f"O centro {voce} está virado pra você. Vire o cubo até o centro "
                f"{NOME_COR[cubo.centro('F')]} ficar virado pra você.")
    for k in (1, 2, 3):
        if iguais(np.rot90(np.array(lida).reshape(3, 3), k).reshape(9).tolist(), esperado) >= 8:
            return (f"O cubo está girado de lado. Gire como um volante até o "
                    f"{NOME_COR[cubo.centro('U')]} ficar pra cima.")
    return None


def desenha_face(img, cores, x, y, cel, borda=(20, 20, 20)):
    cv2.rectangle(img, (x - 2, y - 2), (x + 3 * cel + 1, y + 3 * cel + 1), borda, -1)
    for i, c in enumerate(cores):
        cx, cy = x + (i % 3) * cel, y + (i // 3) * cel
        cor = visao.BGR.get(c, (90, 90, 90))
        cv2.rectangle(img, (cx + 2, cy + 2), (cx + cel - 3, cy + cel - 3), cor, -1)


def desenha_net(img, faces: dict[str, list[str]], x, y, cel):
    for f, (col, lin) in NET.items():
        if f in faces:
            desenha_face(img, faces[f], x + col * (3 * cel + 6), y + lin * (3 * cel + 6), cel)


class App:
    def __init__(self, camera: int | None = 0, estado: str | None = None, duracao: float | None = None):
        self.fim = time.time() + duracao if duracao else None
        self.cap = None
        if camera is not None:
            self.cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
            # MJPG deixa a webcam mandar 720p sem travar o USB
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.espelho = True
        self.frame = None
        self.acc = leitura.Acumulador()
        self.lidas: dict[str, np.ndarray] = {}
        self.montando = False
        self.resultado = None
        self.frente = None
        self.ref: visao.Referencia | None = None
        self.cubo: Cubo | None = None
        self.erros: list[str] = []
        self.aviso = ""
        self.modo = "escanear"
        self.otimo: list[str] | None = None
        self.otimo_msg = ""
        self.mostra_otimo = False
        self.auto = True
        self.batidas = 0
        self.corrigidos: set[int] = set()
        self.inicio_auto = None
        self.info: dict = {}
        self.reg = Registro() if camera is not None else None
        if estado:
            self.cubo = Cubo.de_texto(estado)
            self.revisar()

    # ---------- câmera ----------
    def le_camera(self):
        if not self.cap:
            return None
        ok, fr = self.cap.read()
        if not ok:
            return None
        if fr.shape[1] > 1280:
            fr = cv2.resize(fr, (1280, int(fr.shape[0] * 1280 / fr.shape[1])), interpolation=cv2.INTER_AREA)
        self.frame = fr
        return fr

    def ler_face(self):
        """Face vista agora: pelo detector ou, se ele não achar, pelo quadrado fixo."""
        if self.frame is None:
            return None, None
        f = detector.achar_face(self.frame)
        self.info = {"quadrados": detector.ultimos_quadrados, "fonte": None}
        if f is not None:
            nit = detector.nitidez(self.frame, f)
            self.info["nitidez"] = round(nit)
            if nit > 40:
                amos = f.amostras(self.frame)
                self.info.update(fonte="detector", amostras=[a.tolist() for a in amos],
                                 cores="".join(visao.cor_rapida(a) for a in amos))
                return amos, f.centros
        if visao.uniforme(self.frame):
            amos = visao.amostras(self.frame)
            if visao.parece_face(amos):
                self.info.update(fonte="grade", amostras=[a.tolist() for a in amos],
                                 cores="".join(visao.cor_rapida(a) for a in amos))
                pts = np.array([(x + w / 2, y + h / 2) for x, y, w, h in visao.grade(self.frame)])
                return amos, pts
        return None, None

    def desenha_camera(self, tela: Tela, pontos=None, cores=None):
        fr = self.frame
        if fr is None:
            cv2.rectangle(tela.img, (0, 0), (CAM_W, CAM_H), (45, 45, 45), -1)
            tela.t(20, 20, "sem câmera")
            return
        vis = cv2.flip(fr, 1) if self.espelho else fr.copy()
        for x, y, w, h in visao.grade(fr):
            if self.espelho:
                x = fr.shape[1] - x - w
            cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1)
        if pontos is not None and cores:
            for (x, y), c in zip(pontos, cores):
                if self.espelho:
                    x = fr.shape[1] - x
                cv2.circle(vis, (int(x), int(y)), 22, visao.BGR[c], -1, cv2.LINE_AA)
                cv2.circle(vis, (int(x), int(y)), 22, (0, 0, 0), 3, cv2.LINE_AA)
        tela.img[:CAM_H, :CAM_W] = cv2.resize(vis, (CAM_W, CAM_H))

    # ---------- escanear ----------
    def tela_escanear(self, tela: Tela):
        if self.resultado:
            self._aplica_resultado()
            return
        amos, pts = self.ler_face()
        cores = [visao.cor_rapida(a) for a in amos] if amos else None
        self.desenha_camera(tela, pts, cores)
        larg = LARG - PAINEL_X - 20
        falta = [c for c in PEDIR if c not in self.lidas]
        pedida = falta[0] if falta else None
        if amos and pedida and not self.montando:
            self._le_pedida(np.array(amos, float), pedida)

        y = 20
        if pedida:
            y = tela.t(PAINEL_X, 20, f"Face {len(self.lidas) + 1} de 6", 20, (170, 170, 170))
            y = tela.t(PAINEL_X, y + 4, f"Mostre a face com o centro {NOME_COR[pedida].upper()}", 28,
                       negrito=True, larg=larg)
            cv2.rectangle(tela.img, (PAINEL_X, y + 10), (PAINEL_X + 70, y + 80), visao.BGR[pedida], -1)
            cv2.rectangle(tela.img, (PAINEL_X, y + 10), (PAINEL_X + 70, y + 80), (0, 0, 0), 3)
            prog = min(1.0, max((g.votos for g in self.acc.grupos), default=0) / leitura.MIN_VOTOS)
            cv2.rectangle(tela.img, (PAINEL_X + 90, y + 40), (PAINEL_X + 90 + int(300 * prog), y + 56), (120, 220, 120), -1)
            cv2.rectangle(tela.img, (PAINEL_X + 90, y + 40), (PAINEL_X + 390, y + 56), (90, 90, 90), 1)
            y = tela.t(PAINEL_X, y + 95, "Pode girar a face como quiser, só o centro importa.\n"
                                         "BACKSPACE refaz a anterior · M espelha · Q sai", 17, (170, 170, 170), larg=larg)
        if self.aviso:
            y = tela.t(PAINEL_X, y + 10, self.aviso, 18, (80, 180, 255), larg=larg)
        if self.montando:
            tela.t(PAINEL_X, y + 15, "Montando o cubo...", 22, (120, 220, 120), negrito=True)
        cel = 22
        for i, c in enumerate(PEDIR):
            x0 = 20 + i * (3 * cel + 40)
            tela.t(x0, CAM_H + 12, NOME_COR[c], 15, (170, 170, 170))
            if c in self.lidas:
                desenha_face(tela.img, [visao.cor_rapida(a.astype(np.uint8)) for a in self.lidas[c]],
                             x0, CAM_H + 36, cel, (60, 200, 60))
            else:
                desenha_face(tela.img, ["X"] * 4 + [c] + ["X"] * 4, x0, CAM_H + 36, cel)
        if not falta and not self.montando:
            self.montando = True
            threading.Thread(target=self._montar, args=(dict(self.lidas),), daemon=True).start()

    def _le_pedida(self, amos, pedida):
        for cor, media in self.lidas.items():
            if any(leitura.diferenca(leitura.girar(amos, k), media) < leitura.MESMA_FACE for k in range(4)):
                self.aviso = f"Essa é a face {NOME_COR[cor]}, que já foi lida. Mostre a do centro {NOME_COR[pedida]}."
                return
        g = self.acc.adiciona(amos)
        if g.votos < leitura.MIN_VOTOS:
            return
        # só confere quando a leitura estabiliza, para não reclamar de quadro solto
        problema, refazer = visao.confere_centro(pedida, g.media[4], {c: m[4] for c, m in self.lidas.items()})
        if problema:
            self.aviso = problema
            self.acc = leitura.Acumulador()
            if refazer:
                self.lidas.pop(refazer, None)
            return
        self.aviso = ""
        self.lidas[pedida] = g.media
        self.acc = leitura.Acumulador()
        if self.reg:
            self.reg.evento("face_lida", cor=pedida, media=[a.tolist() for a in g.media])

    def _montar(self, faces):
        try:
            self.resultado = leitura.montar_com_nomes(faces)
        except Exception as e:
            self.resultado = e

    def _aplica_resultado(self):
        res, self.resultado, self.montando = self.resultado, None, False
        if isinstance(res, Exception):
            self.aviso = f"Não consegui montar ({res}). Mostre as faces de novo."
            self.acc, self.lidas = leitura.Acumulador(), {}
            return
        cruas, certas, self.ref, self.frente = res
        if self.reg:
            self.reg.evento("montagem", cruas=cruas, certas=certas,
                            medias={c: [a.tolist() for a in m] for c, m in self.lidas.items()})
        self.corrigidos = {i for i, (a, b) in enumerate(zip(cruas, certas)) if a != b}
        self.cubo = Cubo.de_texto(certas)
        self.revisar()
        if len(self.corrigidos) > 8:
            self.erros = [f"A câmera errou {len(self.corrigidos)} adesivos, é muita coisa. "
                          "Confira a planificação ou aperte R para mostrar de novo com mais luz."]
        elif not self.erros:
            self.inicio_auto = time.time() + 4

    # ---------- revisar ----------
    def revisar(self):
        self.modo = "revisar"
        self.erros = validar(self.cubo)
        if not self.erros:
            e = checar_pecas(self.cubo)
            self.erros = [e] if e else []

    def tela_revisar(self, tela: Tela):
        self.desenha_camera(tela)
        y = tela.t(PAINEL_X, 20, "Confira as cores", 28, negrito=True)
        y = tela.t(PAINEL_X, y + 10, "Clique num quadrado da planificação para trocar a cor.\n"
                                     "ENTER começa a aula · R escaneia de novo", 18, larg=LARG - PAINEL_X - 20)
        if self.erros:
            y = tela.t(PAINEL_X, y + 15, "Problemas:", 20, (80, 180, 255), negrito=True)
            for e in self.erros[:6]:
                y = tela.t(PAINEL_X, y, "• " + e, 18, (80, 180, 255), larg=LARG - PAINEL_X - 20)
        else:
            y = tela.t(PAINEL_X, y + 15, "Tudo certo, o cubo é válido.", 20, (120, 220, 120), negrito=True)
        if self.corrigidos:
            y = tela.t(PAINEL_X, y + 5, f"Corrigi sozinho {len(self.corrigidos)} adesivo(s) que a câmera leu errado "
                                       "(marcados em amarelo).", 18, (120, 220, 255), larg=LARG - PAINEL_X - 20)
        if self.inicio_auto and not self.erros:
            falta = self.inicio_auto - time.time()
            if falta <= 0:
                self.inicio_auto = None
                self.comecar_aula()
                return
            tela.t(PAINEL_X, y + 5, f"A aula começa em {int(falta) + 1}... (clique em algo para parar)", 18)
        faces = {f: self.cubo.face(f) for f in ORDEM_FACES}
        x0, y0, cel = self._net_revisar()
        desenha_net(tela.img, faces, x0, y0, cel)
        for k in self.corrigidos:
            f, i = ORDEM_FACES[k // 9], k % 9
            col, lin = NET[f]
            fx = x0 + col * (3 * cel + 6) + (i % 3) * cel
            fy = y0 + lin * (3 * cel + 6) + (i // 3) * cel
            cv2.rectangle(tela.img, (fx, fy), (fx + cel - 1, fy + cel - 1), (0, 230, 255), 3)

    def _net_revisar(self):
        return PAINEL_X, 330, 30

    def clique(self, ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN or self.modo != "revisar":
            return
        self.inicio_auto = None
        x0, y0, cel = self._net_revisar()
        for f, (col, lin) in NET.items():
            fx, fy = x0 + col * (3 * cel + 6), y0 + lin * (3 * cel + 6)
            if fx <= x < fx + 3 * cel and fy <= y < fy + 3 * cel:
                i = (y - fy) // cel * 3 + (x - fx) // cel
                if i == 4:
                    return
                texto = list(self.cubo.texto())
                k = ORDEM_FACES.index(f) * 9 + i
                texto[k] = visao.CORES[(visao.CORES.index(texto[k]) + 1) % 6]
                self.cubo = Cubo.de_texto("".join(texto))
                self.corrigidos.discard(k)
                self.revisar()
                return

    # ---------- aula ----------
    def comecar_aula(self):
        if self.erros:
            return
        self.etapas = resolver(self.cubo)
        self.lista = []  # (etapa, passo, mov)
        for ei, e in enumerate(self.etapas):
            for pi, p in enumerate(e.passos):
                for m in p.movs:
                    self.lista.append((ei, pi, m))
        self.estados = [self.cubo.copia()]
        for _, _, m in self.lista:
            self.estados.append(self.estados[-1].copia().aplica(m))
        self.pos = 0
        self.modo = "aula"
        self.posicionado = False
        self.certo_desde = None
        if self.reg:
            self.reg.evento("aula", giros=len(self.lista))
        self.otimo, self.otimo_msg = None, ""
        threading.Thread(target=self._calc_otimo, daemon=True).start()

    def _calc_otimo(self):
        if not tabelas_prontas():
            self.otimo_msg = "gerando tabelas na primeira vez (pode levar 30 min)..."
        try:
            self.otimo = resolver_curto(self.cubo)
        except Exception as e:
            self.otimo_msg = f"erro: {e}"

    def tela_aula(self, tela: Tela):
        larg = LARG - PAINEL_X - 20
        fim = self.pos >= len(self.lista)
        atual = self.estados[self.pos]
        # a câmera vê o lado de trás: a frente do cubo fica virada para quem está montando
        esperado = atual.face(FACE_CAMERA)
        lida, pts = None, None
        if self.ref:
            amos, pts = self.ler_face()
            if amos:
                lida = [self.ref.cor(a) for a in amos]
        self.desenha_camera(tela, pts, lida)
        problema = posicao_errada(lida, esperado, atual) if lida else None
        if self.posicionado:
            self._auto_avanco(lida)
        self.info.update(confere=(iguais(lida, esperado) >= 8) if lida else None, giro=self.pos,
                         esperado="".join(esperado), lida="".join(lida) if lida else None,
                         posicao=problema)

        frente, cima = NOME_COR[atual.centro("F")], NOME_COR[atual.centro("U")]
        faixa = f"Centro {frente} virado pra você · {cima} pra cima"
        cv2.rectangle(tela.img, (0, 0), (CAM_W, 44), (40, 90, 40) if not problema else (40, 40, 150), -1)
        tela.t(14, 8, problema or faixa, 20, (255, 255, 255), negrito=True, larg=CAM_W - 28)

        if not self.posicionado and not fim:
            y = tela.t(PAINEL_X, 15, "Antes de começar, segure assim", 26, negrito=True)
            y = tela.t(PAINEL_X, y + 6, f"O centro {frente.upper()} virado pra você e o {cima.upper()} pra cima. "
                                        f"A câmera fica olhando o lado de trás (o centro "
                                        f"{NOME_COR[atual.centro(FACE_CAMERA)]}).", 19, larg=larg)
            tam = 330
            img = render3d.desenhar(atual, None, 0, tam, rotulos=True)
            x0 = PAINEL_X + (larg - tam) // 2
            tela.img[y + 8:y + 8 + tam, x0:x0 + tam] = img
            y += tam + 16
            if lida is None:
                tela.t(PAINEL_X, y, "Mostre o cubo pra câmera nessa posição.", 18, (170, 170, 170), larg=larg)
                self.certo_desde = None
            elif problema:
                tela.t(PAINEL_X, y, problema, 18, (80, 180, 255), larg=larg)
                self.certo_desde = None
            else:
                self.certo_desde = self.certo_desde or time.time()
                tela.t(PAINEL_X, y, "Isso! Segura assim...", 20, (120, 220, 120), negrito=True)
                if time.time() - self.certo_desde > 1.2:
                    self.posicionado = True
                    if self.reg:
                        self.reg.evento("posicionado")
            tela.t(PAINEL_X, ALT - 30, "ENTER pula esta etapa · Q sai", 15, (150, 150, 150))
        elif fim:
            y = tela.t(PAINEL_X, 20, "Cubo montado!", 34, (120, 220, 120), negrito=True)
            tela.t(PAINEL_X, y + 10, f"{len(self.lista)} giros pelo método de camadas.\n"
                                     "← volta · R escaneia outro · Q sai", 20)
        else:
            ei, pi, m = self.lista[self.pos]
            e, p = self.etapas[ei], self.etapas[ei].passos[pi]
            y = tela.t(PAINEL_X, 15, f"Etapa {e.numero} de 8 · {e.nome}", 26, negrito=True)
            y = tela.t(PAINEL_X, y + 2, e.objetivo, 18, (190, 190, 190), larg=larg)
            y = tela.t(PAINEL_X, y + 12, p.dica, 20, larg=larg)
            idx = sum(1 for k in range(self.pos) if self.lista[k][:2] == (ei, pi))
            y = tela.t(PAINEL_X, y + 6, f"movimento {idx + 1} de {len(p.movs)} desta sequência", 16, (170, 170, 170))
            # réplica do cubo fazendo o giro, em loop: 1 s girando, 0,6 s parado no fim
            t = time.time() % 1.6
            fase = min(1.0, t / 1.0)
            fase = fase * fase * (3 - 2 * fase)
            tam = 330
            img = render3d.desenhar(atual, m, fase, tam, rotulos=True)
            x0 = PAINEL_X + (larg - tam) // 2
            tela.img[y + 4:y + 4 + tam, x0:x0 + tam] = img
            y = tela.t(PAINEL_X, y + tam + 8, DESC.get(m, "").capitalize(), 24, (255, 210, 120), negrito=True, larg=larg)
            tela.t(PAINEL_X, y + 4, "Por quê: " + e.porque, 15, (160, 160, 160), larg=larg)
            tela.t(PAINEL_X, ALT - 50, f"giro {self.pos + 1} de {len(self.lista)} · →/ESPAÇO próximo · ← volta · "
                                       f"N/P etapa · A auto ({'ligado' if self.auto else 'desligado'}) · "
                                       f"O solução curta · Q sai", 15, (150, 150, 150), larg=larg)

        y0 = CAM_H + 15
        tela.t(20, y0, "A câmera deve ver", 16, (170, 170, 170))
        desenha_face(tela.img, esperado, 20, y0 + 28, 34)
        if lida:
            ok = iguais(lida, esperado) >= 8
            tela.t(170, y0, "Câmera vê", 16, (170, 170, 170))
            desenha_face(tela.img, lida, 170, y0 + 28, 34, (60, 200, 60) if ok else (40, 40, 200))
            tela.t(20, y0 + 145, "confere" if ok else "diferente do esperado", 16,
                   (120, 220, 120) if ok else (80, 180, 255))
        tela.t(330, y0, "Cubo inteiro agora", 16, (170, 170, 170))
        desenha_net(tela.img, {f: atual.face(f) for f in ORDEM_FACES}, 330, y0 + 28, 13)

        if self.mostra_otimo:
            cv2.rectangle(tela.img, (540, CAM_H + 10), (CAM_W, ALT - 10), (55, 45, 30), -1)
            y = tela.t(550, CAM_H + 18, "Solução mais curta (curiosidade)", 17, negrito=True)
            if self.otimo:
                tela.t(550, y + 5, f"{len(self.otimo)} giros: " + " ".join(self.otimo), 15, larg=CAM_W - 560)
            else:
                tela.t(550, y + 5, self.otimo_msg or "calculando...", 15)

    def tela_espelho(self, tela: Tela):
        """A réplica segue a posição do cubo na mão."""
        cubo = self.estados[self.pos] if getattr(self, "estados", None) else self.cubo
        amos, pts = self.ler_face()
        lida = [self.ref.cor(a) for a in amos] if amos else None
        self.desenha_camera(tela, pts, lida)
        mao = None
        if self.frame is not None and self.maos.ok:
            # a mão muda menos que o cubo: o esqueleto roda a cada 2 quadros
            self.n_quadro = getattr(self, "n_quadro", 0) + 1
            if self.n_quadro % 2 == 0 or not self.maos.maos:
                self.maos.atualiza(self.frame)
            peq = (int(self.frame.shape[0] * ESCALA), int(self.frame.shape[1] * ESCALA))
            mao = self.maos.mascara(peq, ESCALA, self.frame)
        achou = self.frame is not None and self.rastreador.atualiza(
            self.frame, pts, lida, cubo, self.ref, self.espelho, mao, self.maos.maos)
        larg = LARG - PAINEL_X - 20
        y = tela.t(PAINEL_X, 15, "Modo espelho", 26, negrito=True)
        y = tela.t(PAINEL_X, y + 4, "Gire o cubo na mão: a réplica acompanha.", 18, (190, 190, 190), larg=larg)
        tam = 500
        img = render3d.desenhar(cubo, None, 0, tam, vista=self.rastreador.R)
        x0 = PAINEL_X + (larg - tam) // 2
        tela.img[y + 10:y + 10 + tam, x0:x0 + tam] = img
        y += tam + 18
        if achou:
            cor = NOME_COR[cubo.centro(self.rastreador.face)]
            como = {"grade": "pela grade", "busca": "achou agora"}.get(self.rastreador.fonte,
                                                                         "pelas cores do cubo inteiro")
            tela.t(PAINEL_X, y, f"Face {cor} mais virada pra câmera · rastreando {como} "
                                f"({100 * max(0, self.rastreador.qualidade):.0f}% das cores batem)", 17,
                   (120, 220, 120), larg=larg)
        else:
            tela.t(PAINEL_X, y, "Pra começar, mostre uma face do cubo de frente pra câmera. "
                                "Depois pode girar à vontade.", 18, (80, 180, 255), negrito=True, larg=larg)
        tela.t(PAINEL_X, ALT - 30, "E volta · M espelha a câmera · Q sai", 15, (150, 150, 150))
        self.info.update(face_pose=self.rastreador.face if achou else None,
                         qualidade=round(float(self.rastreador.qualidade), 2), fonte_pose=self.rastreador.fonte,
                         mao=round(self.rastreador.mao, 3), maos=len(self.maos.maos))
        if self.reg and self.frame is not None:
            self.reg.foto_espelho(self.frame, self.rastreador, self.maos, achou, img)

    def _auto_avanco(self, lida):
        if not self.auto or lida is None or self.pos >= len(self.lista):
            self.batidas = 0
            return
        # olha alguns giros à frente: pega quem fez vários seguidos ou um giro que não muda a frente (B)
        # aceita 1 quadrado diferente (vermelho/laranja oscilando), desde que a leitura
        # se pareça mais com o giro seguinte do que com o estado atual
        atual = self.estados[self.pos].face(FACE_CAMERA)
        alvo = None
        for k in range(1, 5):
            if self.pos + k >= len(self.estados):
                break
            f = self.estados[self.pos + k].face(FACE_CAMERA)
            if f != atual and iguais(lida, f) >= 8 and iguais(lida, f) > iguais(lida, atual):
                alvo = self.pos + k
                break
        if alvo is not None:
            self.batidas += 1
            if self.batidas >= 6:
                if self.reg:
                    self.reg.evento("avanco", de=self.pos, para=alvo)
                self.pos = alvo
                self.batidas = 0
        else:
            self.batidas = 0

    def pula_etapa(self, d):
        if not self.lista:
            return
        atual = self.lista[min(self.pos, len(self.lista) - 1)][0]
        alvo = atual + d
        for k, (ei, _, _) in enumerate(self.lista):
            if ei == alvo:
                self.pos = k
                return
        if d > 0:
            self.pos = len(self.lista)

    # ---------- loop ----------
    def rodar(self):
        cv2.namedWindow("cubo-tutor")
        cv2.setMouseCallback("cubo-tutor", self.clique)
        while True:
            self.le_camera()
            tela = Tela()
            if self.modo == "escanear":
                self.tela_escanear(tela)
            elif self.modo == "revisar":
                self.tela_revisar(tela)
            elif self.modo == "espelho":
                self.tela_espelho(tela)
            else:
                self.tela_aula(tela)
            img = tela.final()
            cv2.imshow("cubo-tutor", img)
            if self.reg:
                self.info["modo"] = self.modo
                self.info["aviso"] = self.aviso
                if self.modo == "escanear":
                    self.info["pedida"] = next((c for c in PEDIR if c not in self.lidas), None)
                    self.info["grupos"] = [{"votos": g.votos, "cores": "".join(
                        visao.cor_rapida(a.astype(np.uint8)) for a in g.media)} for g in self.acc.faces()]
                    self.info["lidas"] = "".join(self.lidas)
                self.reg.quadro(self.info, img, self.frame)
            k = cv2.waitKeyEx(15)
            if self.fim and time.time() > self.fim:
                break
            if k == -1:
                continue
            if k in (ord("q"), 27) or cv2.getWindowProperty("cubo-tutor", cv2.WND_PROP_VISIBLE) < 1:
                break
            self.tecla(k)
        if self.cap:
            self.cap.release()
        if self.reg:
            self.reg.fecha()
        cv2.destroyAllWindows()

    def tecla(self, k):
        direita, esquerda = (2555904, 65363), (2424832, 65361)
        c = chr(k).lower() if 0 <= k < 256 else ""
        if c == "m":
            self.espelho = not self.espelho
        if c == "e" and self.cubo is not None and self.ref is not None:
            if self.modo == "espelho":
                self.modo = self.modo_antes
            else:
                self.modo_antes, self.modo = self.modo, "espelho"
                self.rastreador = Rastreador()
                if not hasattr(self, "maos"):
                    self.maos = Maos()
            return
        if self.modo == "escanear":
            if k == 8 and self.lidas:
                self.lidas.pop(list(self.lidas)[-1])
                self.acc, self.aviso = leitura.Acumulador(), ""
        elif self.modo == "revisar":
            if k == 13:
                self.comecar_aula()
            elif c == "r":
                self.reiniciar()
        elif not self.posicionado and k == 13:
            self.posicionado = True
        else:
            if c == " " or k in direita:
                self.pos = min(self.pos + 1, len(self.lista))
            elif k in esquerda:
                self.pos = max(self.pos - 1, 0)
            elif c == "n":
                self.pula_etapa(1)
            elif c == "p":
                self.pula_etapa(-1)
            elif c == "a":
                self.auto = not self.auto
            elif c == "o":
                self.mostra_otimo = not self.mostra_otimo
            elif c == "r":
                self.reiniciar()

    def reiniciar(self):
        if self.reg:
            self.reg.evento("reinicio")
        self.acc, self.cubo, self.ref, self.aviso = leitura.Acumulador(), None, None, ""
        self.lidas = {}
        self.montando, self.resultado = False, None
        self.corrigidos, self.inicio_auto = set(), None
        self.modo = "escanear"
