from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import detector, leitura, visao
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
}

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.espelho = True
        self.frame = None
        self.acc = leitura.Acumulador()
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
        if amos and not self.montando:
            self.acc.adiciona(amos)
        self.desenha_camera(tela, pts, cores)
        larg = LARG - PAINEL_X - 20
        prontas = self.acc.prontas()
        y = tela.t(PAINEL_X, 20, f"Mostre as 6 faces ({prontas} de 6)", 28, negrito=True)
        y = tela.t(PAINEL_X, y + 10, "Em qualquer ordem e girando o cubo como quiser na frente da câmera. "
                                     "O programa descobre sozinho onde fica cada face.", 20, larg=larg)
        y = tela.t(PAINEL_X, y + 10, "BACKSPACE recomeça · M espelha · Q sai", 16, (150, 150, 150))
        if self.aviso:
            y = tela.t(PAINEL_X, y + 10, self.aviso, 18, (80, 180, 255), larg=larg)
        if self.montando:
            tela.t(PAINEL_X, y + 15, "Montando o cubo...", 22, (120, 220, 120), negrito=True)
        cel = 22
        for i, g in enumerate(self.acc.faces()):
            x0 = 20 + i * (3 * cel + 40)
            desenha_face(tela.img, [visao.cor_rapida(a.astype(np.uint8)) for a in g.media], x0, CAM_H + 30, cel)
            prog = min(1.0, g.votos / leitura.MIN_VOTOS)
            cor = (120, 220, 120) if prog >= 1 else (0, 200, 255)
            cv2.rectangle(tela.img, (x0, CAM_H + 30 + 3 * cel + 8), (x0 + int(3 * cel * prog), CAM_H + 30 + 3 * cel + 16), cor, -1)
        if prontas == 6 and not self.montando:
            self.montando = True
            faces = [g.media for g in self.acc.faces()]
            threading.Thread(target=self._montar, args=(faces,), daemon=True).start()

    def _montar(self, faces):
        try:
            self.resultado = leitura.montar(faces)
        except Exception as e:
            self.resultado = e

    def _aplica_resultado(self):
        res, self.resultado, self.montando = self.resultado, None, False
        if isinstance(res, Exception):
            self.aviso = f"Não consegui montar ({res}). Mostre as faces de novo."
            self.acc = leitura.Acumulador()
            return
        cruas, certas, self.ref, self.frente = res
        if self.reg:
            self.reg.evento("montagem", cruas=cruas, certas=certas,
                            medias=[[a.tolist() for a in g.media] for g in self.acc.faces()])
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
        esperado = self.estados[self.pos].face("F")
        lida, pts = None, None
        if self.ref:
            amos, pts = self.ler_face()
            if amos:
                lida = [self.ref.cor(a) for a in amos]
        self.desenha_camera(tela, pts, lida)
        self._auto_avanco(lida)
        self.info.update(confere=(lida == esperado) if lida else None, giro=self.pos,
                         esperado="".join(esperado), lida="".join(lida) if lida else None)

        if fim:
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
            seq = "  ".join(f"[{mv}]" if k == idx else mv for k, mv in enumerate(p.movs))
            y = tela.t(PAINEL_X, y + 12, seq, 22, (255, 210, 120), negrito=True, larg=larg)
            y = tela.t(PAINEL_X, y + 12, m, 64, (255, 210, 120), negrito=True)
            y = tela.t(PAINEL_X, y + 8, DESC.get(m, ""), 24, larg=larg)
            y = tela.t(PAINEL_X, y + 12, "Por quê: " + e.porque, 16, (160, 160, 160), larg=larg)

        tela.t(PAINEL_X, ALT - 50, f"giro {min(self.pos + 1, len(self.lista))} de {len(self.lista)} · "
                                   f"→/ESPAÇO próximo · ← volta · N/P etapa · A auto ({'ligado' if self.auto else 'desligado'}) "
                                   f"· O solução curta · Q sai", 15, (150, 150, 150), larg=larg)

        if self.pos == 0 and self.cubo:
            cor = NOME_COR.get(self.cubo.centro("F"), "")
            tela.t(20, CAM_H - 40, f"Segure com o centro {cor} na frente e o amarelo em cima", 20,
                   (255, 255, 255), negrito=True)
        y0 = CAM_H + 15
        tela.t(20, y0, "Frente esperada", 16, (170, 170, 170))
        desenha_face(tela.img, esperado, 20, y0 + 28, 34)
        if lida:
            ok = lida == esperado
            tela.t(160, y0, "Câmera vê", 16, (170, 170, 170))
            desenha_face(tela.img, lida, 160, y0 + 28, 34, (60, 200, 60) if ok else (40, 40, 200))
            tela.t(20, y0 + 145, "confere" if ok else "diferente do esperado", 16,
                   (120, 220, 120) if ok else (80, 180, 255))
        tela.t(330, y0, "Cubo inteiro agora", 16, (170, 170, 170))
        atual = self.estados[self.pos]
        desenha_net(tela.img, {f: atual.face(f) for f in ORDEM_FACES}, 330, y0 + 28, 13)

        if self.mostra_otimo:
            cv2.rectangle(tela.img, (PAINEL_X - 10, 520), (LARG - 10, 655), (55, 45, 30), -1)
            y = tela.t(PAINEL_X, 528, "Solução mais curta (curiosidade)", 20, negrito=True)
            if self.otimo:
                tela.t(PAINEL_X, y + 5, f"{len(self.otimo)} giros: " + " ".join(self.otimo), 18, larg=larg)
            else:
                tela.t(PAINEL_X, y + 5, self.otimo_msg or "calculando...", 18)

    def _auto_avanco(self, lida):
        if not self.auto or lida is None or self.pos >= len(self.lista):
            self.batidas = 0
            return
        # olha alguns giros à frente: pega quem fez vários seguidos ou um giro que não muda a frente (B)
        atual = self.estados[self.pos].face("F")
        alvo = None
        for k in range(1, 5):
            if self.pos + k >= len(self.estados):
                break
            f = self.estados[self.pos + k].face("F")
            if f != atual and lida == f:
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
            else:
                self.tela_aula(tela)
            img = tela.final()
            cv2.imshow("cubo-tutor", img)
            if self.reg:
                self.info["modo"] = self.modo
                if self.modo == "escanear":
                    self.info["grupos"] = [{"votos": g.votos, "cores": "".join(
                        visao.cor_rapida(a.astype(np.uint8)) for a in g.media)} for g in self.acc.faces()]
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
        if self.modo == "escanear":
            if k == 8:
                self.reiniciar()
        elif self.modo == "revisar":
            if k == 13:
                self.comecar_aula()
            elif c == "r":
                self.reiniciar()
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
        self.montando, self.resultado = False, None
        self.corrigidos, self.inicio_auto = set(), None
        self.modo = "escanear"
