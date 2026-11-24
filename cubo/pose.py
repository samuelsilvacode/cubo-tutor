"""Onde o cubo está na mão: qual face está virada para a câmera e a rotação 3D dele.

Dois modos que se completam:
- grade: quando uma face aparece de frente, solvePnP nos 9 adesivos dá a pose do zero;
- modelo: entre um quadro e outro, ajusta a pose anterior até as cores projetadas
  de todos os adesivos visíveis baterem com a imagem (funciona com quina e aresta).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import visao
from .modelo import Cubo, ORDEM_FACES, grade_da_face

# câmera (x direita, y baixo, z para dentro) -> réplica (x direita, y cima, z para quem olha)
CAM_PARA_TELA = np.diag([1.0, -1.0, -1.0])
ESPELHA = np.diag([-1.0, 1.0, 1.0])
ESCALA = 0.25  # o modelo compara cores numa imagem reduzida


def _ortonormal(m):
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def _camera(tamanho):
    h, w = tamanho
    f = 0.75 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def _pontos_adesivos():
    """5 pontos por adesivo (centro e 4 perto das pontas), com a normal e o índice do adesivo."""
    pts, normais, quem = [], [], []
    for f in ORDEM_FACES:
        for i, (pos, n) in enumerate(grade_da_face(f)):
            n = np.array(n, float)
            u = np.array([n[1], n[2], n[0]])
            v = np.cross(n, u)
            c = np.array(pos, float) + 0.5 * n
            for a, b in ((0, 0), (-.25, -.25), (.25, -.25), (.25, .25), (-.25, .25)):
                pts.append(c + (a * u + b * v))
                normais.append(n)
                quem.append(ORDEM_FACES.index(f) * 9 + i)
    return np.array(pts), np.array(normais), np.array(quem)


PONTOS, NORMAIS_P, ADESIVO = _pontos_adesivos()
GIROS = [f + s for f in "URFDLB" for s in ("", "'", "2")]


def _rotacoes_busca():
    """Rotações para a busca do zero: várias direções de olhar e giros em volta delas."""
    out = []
    for yaw in range(0, 360, 30):
        for pitch in (-60, -30, 0, 30, 60):
            for roll in range(0, 360, 45):
                a, b, c = np.radians([yaw, pitch, roll])
                ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
                rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
                out.append(rz @ rx @ ry)
    return out


ROTACOES_BUSCA = _rotacoes_busca()


class Rastreador:
    def __init__(self, suavizar: float = 0.5):
        self.R_cam: np.ndarray | None = None  # cubo -> câmera
        self.t: np.ndarray | None = None
        self.R: np.ndarray | None = None  # para desenhar a réplica
        self.face: str | None = None
        self.acertos = 0
        self.qualidade = 0.0
        self.fonte = None
        self.suavizar = suavizar
        self.mao = 0.0
        self.pontos_debug = None
        self.ult_busca = 0.0
        self.historico: list = []
        self.prever = True
        self._ultimo = None
        self.incerto = 0

    # ---------- pose pela grade ----------
    def identifica(self, lida: list[str], cubo: Cubo):
        """Qual face do cubo é essa leitura e como ela está girada (np.rot90 k vezes)."""
        melhor = (0, None, 0)
        for f in ORDEM_FACES:
            grade = np.array(cubo.face(f)).reshape(3, 3)
            for k in range(4):
                n = sum(a == b for a, b in zip(np.rot90(grade, k).reshape(9), lida))
                if n > melhor[0]:
                    melhor = (n, f, k)
        return melhor

    def pela_grade(self, centros, lida, cubo: Cubo, tamanho) -> bool:
        acertos, face, k = self.identifica(lida, cubo)
        self.acertos = acertos
        if acertos < 7:
            return False
        idx = np.rot90(np.arange(9).reshape(3, 3), k).reshape(9)
        grade = grade_da_face(face)
        obj = np.array([np.array(grade[i][0]) + 0.5 * np.array(grade[i][1]) for i in idx], float)
        ok, rvecs, tvecs, erros = cv2.solvePnPGeneric(obj, np.asarray(centros, float), _camera(tamanho), None,
                                                       flags=cv2.SOLVEPNP_IPPE)
        if not ok or not len(rvecs):
            return False
        opcoes = sorted(zip(np.ravel(erros), [cv2.Rodrigues(r)[0] for r in rvecs], tvecs), key=lambda x: x[0])
        R, t = opcoes[0][1], opcoes[0][2].ravel()
        # face plana tem duas inclinações quase iguais na imagem: se empatar, fica com a mais perto da anterior
        if self.R_cam is not None and len(opcoes) > 1 and opcoes[1][0] < opcoes[0][0] * 1.5 + 0.5:
            R, t = min(((o[1], o[2].ravel()) for o in opcoes[:2]), key=lambda x: np.linalg.norm(x[0] - self.R_cam))
        self.R_cam, self.t, self.face, self.fonte = R, t, face, "grade"
        return True

    # ---------- pose pelo modelo ----------
    def _mapa_cores(self, frame, ref: visao.Referencia, mao=None, janela=None):
        """Cor de cada pixel (índice do centro ou -1). Com janela (x0, y0, x1, y1), só calcula ali dentro."""
        peq = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
        H, W = peq.shape[:2]
        if janela is not None:
            x0, y0, x1, y1 = janela
            completo = np.full((H, W), -1, np.int64)
            dentro = self._mapa_cores_bruto(peq[y0:y1, x0:x1], ref,
                                           None if mao is None else mao[y0:y1, x0:x1])
            completo[y0:y1, x0:x1] = dentro
            return completo
        return self._mapa_cores_bruto(peq, ref, mao)

    def _janela(self, frame, R, t):
        """Retângulo (na imagem reduzida) onde o cubo deve aparecer, com folga para o movimento."""
        K = _camera(frame.shape[:2])
        p = (PONTOS @ R.T + t) @ K.T
        uv = p[:, :2] / p[:, 2:3] * ESCALA
        H, W = int(frame.shape[0] * ESCALA), int(frame.shape[1] * ESCALA)
        (x0, y0), (x1, y1) = uv.min(axis=0), uv.max(axis=0)
        folga = 0.6 * max(x1 - x0, y1 - y0)
        return (int(max(0, x0 - folga)), int(max(0, y0 - folga)),
                int(min(W, x1 + folga)), int(min(H, y1 + folga)))

    def _mapa_cores_bruto(self, peq, ref, mao):
        h, w = peq.shape[:2]
        if h == 0 or w == 0:
            return np.full((h, w), -1, np.int64)
        feats = visao._lab(peq.reshape(-1, 3), ref.wb)
        d = np.linalg.norm(feats[:, None] - ref.ref[None], axis=2)
        rotulo = d.argmin(axis=1)
        rotulo[d.min(axis=1) > 90] = -1  # fundo, sombra: não conta
        self.mao = 0.0
        if mao is not None:
            # dedo na frente do cubo não conta nem a favor nem contra
            m = mao.reshape(-1)
            rotulo[m] = -1
            self.mao = float(m.mean())
        return rotulo.reshape(h, w)

    def _nota(self, R, t, mapa, esperado, K, guardar=False):
        normal_cam = NORMAIS_P @ R.T
        p_cam = PONTOS @ R.T + t
        visivel = (normal_cam * -p_cam).sum(axis=1) > 0.05
        if visivel.sum() < 10:
            return -1.0, 0
        p = p_cam[visivel] @ K.T
        uv = p[:, :2] / p[:, 2:3] * ESCALA
        h, w = mapa.shape
        dentro = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if dentro.sum() < 10:
            return -1.0, 0
        uv = uv[dentro].astype(int)
        lido = mapa[uv[:, 1], uv[:, 0]]
        alvo = esperado[visivel][dentro]
        bate = (lido == alvo).sum()
        erra = ((lido != alvo) & (lido >= 0)).sum()
        contam = int((lido >= 0).sum())
        if guardar:
            self.pontos_debug = (uv / ESCALA, np.where(lido == alvo, 1, np.where(lido >= 0, -1, 0)))
        if contam < 10:
            return -1.0, 0
        # nota sobre o que dá pra ver (sem mão e sem fundo), mas sem premiar pose que "some" com o cubo:
        # se quase nada conta, divide pelo menos por 60% dos pontos projetados
        return (bate - 0.5 * erra) / max(contam, 0.6 * len(alvo)), contam

    def buscar(self, frame, cubo: Cubo, ref: visao.Referencia, mao=None, maos_pts=()) -> bool:
        """Acha o cubo do zero, sem face de frente: procura perto da mão e testa várias rotações."""
        if not maos_pts:
            return False  # no modo espelho o cubo está na mão; sem mão, não procura (evita achar cubo na camisa)
        mapa = self._mapa_cores(frame, ref, mao)
        local = self._onde_esta(frame, maos_pts)
        if local is None:
            return False
        cx, cy, lado = local
        K = _camera(frame.shape[:2])
        z = K[0, 0] * 0.8 / lado  # adesivo tem ~0,8 da largura de uma peça
        t = np.array([(cx - K[0, 2]) / K[0, 0] * z, (cy - K[1, 2]) / K[1, 1] * z, z])
        idx = {c: i for i, c in enumerate(ref.nomes)}
        esperado = np.array([idx.get(c, -2) for c in cubo.texto()])[ADESIVO]
        candidatos = []
        for R in ROTACOES_BUSCA:
            nota, _ = self._nota(R, t, mapa, esperado, K)
            candidatos.append((nota, R))
        candidatos.sort(key=lambda x: -x[0])
        melhor = None
        for nota0, R in candidatos[:6]:
            R2, t2, nota = self._refina(R, t.copy(), mapa, esperado, K)
            if melhor is None or nota > melhor[2]:
                melhor = (R2, t2, nota)
        R, t, nota = melhor
        self.qualidade = nota
        # o branco bate com parede; a posição só vale se as cores fortes também baterem
        if nota < 0.5 or self._nota_cores_fortes(R, t, mapa, esperado, K, ref) < 0.5:
            return False
        self._nota(R, t, mapa, esperado, K, guardar=True)
        self.R_cam, self.t, self.fonte = _ortonormal(R), t, "busca"
        return True

    def _onde_esta(self, frame, maos_pts):
        """Centro e tamanho do adesivo do cubo: grupo de quadradinhos parecidos perto da mão."""
        from .detector import _quadrados
        qs = _quadrados(frame)
        if len(qs) < 3:
            return None
        pts = np.vstack(maos_pts)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        folga = max(x1 - x0, y1 - y0)
        perto = [q for q in qs if x0 - folga < q[0] < x1 + folga and y0 - folga < q[1] < y1 + folga]
        if len(perto) < 3:
            return None
        melhor = None
        for cx, cy, lado, _ in perto:
            grupo = [q for q in perto if abs(q[2] - lado) < lado * 0.35 and np.hypot(q[0] - cx, q[1] - cy) < lado * 4.5]
            if melhor is None or len(grupo) > len(melhor):
                melhor = grupo
        if len(melhor) < 3:
            return None
        g = np.array([(q[0], q[1], q[2]) for q in melhor])
        return float(g[:, 0].mean()), float(g[:, 1].mean()), float(np.median(g[:, 2]))

    def _nota_cores_fortes(self, R, t, mapa, esperado, K, ref):
        branco = ref.nomes.index("W") if "W" in ref.nomes else -9
        fortes = np.where(esperado == branco, -2, esperado)
        nota, n = self._nota(R, t, mapa, fortes, K)
        return nota if n >= 15 else -1.0

    def _refina(self, R, t, mapa, esperado, K):
        nota, _ = self._nota(R, t, mapa, esperado, K)
        passos = [(np.radians(8), 0.35), (np.radians(4), 0.18), (np.radians(2), 0.08), (np.radians(1), 0.04)]
        for passo_r, passo_t in passos:
            melhorou = True
            while melhorou:
                melhorou = False
                for eixo in range(6):
                    for sinal in (-1, 1):
                        if eixo < 3:
                            dv = np.zeros(3)
                            dv[eixo] = sinal * passo_r
                            R2, t2 = cv2.Rodrigues(dv)[0] @ R, t
                        else:
                            R2, t2 = R, t.copy()
                            t2[eixo - 3] += sinal * passo_t * (3 if eixo == 5 else 1)
                        n2, _ = self._nota(R2, t2, mapa, esperado, K)
                        if n2 > nota + 1e-6:
                            R, t, nota, melhorou = R2, t2, n2, True
        return R, t, nota

    def pelo_modelo(self, frame, cubo: Cubo, ref: visao.Referencia, mao=None) -> bool:
        if self.R_cam is None:
            return False
        partidas = [(self.R_cam, self.t.copy())]
        prevista = self._prevista() if self.prever else None
        if prevista is not None:
            partidas.append(prevista)
        # janela que cobre onde o cubo estava e onde ele deve estar agora
        jan = [self._janela(frame, R0, t0) for R0, t0 in partidas]
        janela = (min(j[0] for j in jan), min(j[1] for j in jan), max(j[2] for j in jan), max(j[3] for j in jan))
        mapa = self._mapa_cores(frame, ref, mao, janela)
        idx = {c: i for i, c in enumerate(ref.nomes)}
        esperado = np.array([idx.get(c, -2) for c in cubo.texto()])[ADESIVO]
        K = _camera(frame.shape[:2])
        # ajusta a partir da última pose e da prevista; fica com o que encaixar melhor no fim
        R, t, nota = max((self._refina(R0, t0.copy(), mapa, esperado, K) for R0, t0 in partidas),
                         key=lambda x: x[2])
        self._nota(R, t, mapa, esperado, K, guardar=True)
        self.qualidade = nota
        self._ultimo = (mapa, K, {c: i for i, c in enumerate(ref.nomes)})
        if nota < 0.35:
            # nota despencou: pode ser um giro grande de camada, não perda do cubo.
            # guarda a pose por alguns quadros para a detecção de giro decidir.
            self.R_cam, self.t = _ortonormal(R), t
            self.incerto += 1
            return False
        self.incerto = 0
        self.R_cam, self.t, self.fonte = _ortonormal(R), t, "modelo"
        return True

    def segurando(self) -> bool:
        """Tem pose guardada (mesmo que incerta) para testar giros."""
        return self.R_cam is not None and self._ultimo is not None

    def aceita_giro(self):
        """Depois de aplicar um giro, a pose incerta volta a valer."""
        self.incerto = 0

    def melhor_giro(self, cubo: Cubo):
        """Qual dos 18 giros de camada faz a imagem bater melhor que o estado atual (e quanto melhor)."""
        if self._ultimo is None or self.R_cam is None:
            return None, 0.0
        mapa, K, idx = self._ultimo

        def nota(c):
            esperado = np.array([idx.get(x, -2) for x in c.texto()])[ADESIVO]
            return self._nota(self.R_cam, self.t, mapa, esperado, K)[0]

        base = nota(cubo)
        if base > 0.9:
            return None, 0.0  # tudo batendo: ninguém girou camada (e não gasta tempo testando 18 giros)
        brutos = sorted(((nota(cubo.copia().aplica(m)) - base, m) for m in GIROS), reverse=True)
        if brutos[0][0] <= 0.03:
            return None, 0.0
        # comparação justa: a pose foi ajustada para o estado atual, então o candidato também ganha
        # o ajuste dele antes de comparar (senão a pose "entorta" para explicar a imagem sem o giro)
        melhor = (None, 0.0)
        for _, m in brutos[:2]:
            c = cubo.copia().aplica(m)
            esperado = np.array([idx.get(x, -2) for x in c.texto()])[ADESIVO]
            _, _, n = self._refina(self.R_cam, self.t.copy(), mapa, esperado, K)
            if n - base > melhor[1]:
                melhor = (m, n - base)
        return melhor

    def _prevista(self):
        """Pose prevista para agora, seguindo a velocidade dos dois últimos quadros."""
        if len(self.historico) < 2:
            return None
        (_, R0, p0), (_, R1, p1) = self.historico[-2:]
        # a câmera entrega quadros em intervalos regulares: prevê um quadro adiante
        giro = cv2.Rodrigues(R1 @ R0.T)[0].ravel()
        if np.linalg.norm(giro) > np.radians(60):
            return None  # salto grande demais: provavelmente erro, não movimento
        return cv2.Rodrigues(giro)[0] @ R1, p1 + (p1 - p0)

    # ---------- um quadro ----------
    def atualiza(self, frame, centros, lida, cubo: Cubo, ref, espelho=False, mao=None, maos_pts=()) -> bool:
        ok = False
        if centros is not None and lida is not None:
            ok = self.pela_grade(centros, lida, cubo, frame.shape[:2])
        ok = self.pelo_modelo(frame, cubo, ref, mao) or ok
        if ok:
            self.historico = (self.historico + [(time.time(), self.R_cam.copy(), self.t.copy())])[-2:]
        else:
            self.historico = []
        if not ok and 0 < self.incerto <= 6:
            return False  # segura a pose: a detecção de giro ainda pode explicar a queda
        if not ok:
            # perdeu (ou nunca achou): busca do zero, no máximo umas 4 vezes por segundo
            agora = time.time()
            if agora - self.ult_busca > 0.25:
                self.ult_busca = agora
                ok = self.buscar(frame, cubo, ref, mao, maos_pts)
            if not ok:
                self.R_cam = None
        if not ok:
            return False
        novo = CAM_PARA_TELA @ self.R_cam
        if espelho:
            novo = ESPELHA @ novo
        if self.R is None or np.linalg.norm(novo - self.R) > 1.2:
            self.R = novo
        else:
            self.R = _ortonormal((1 - self.suavizar) * self.R + self.suavizar * novo)
        normal_cam = np.array([np.array(grade_da_face(f)[4][1], float) for f in ORDEM_FACES]) @ self.R_cam.T
        self.face = ORDEM_FACES[int(np.argmin(normal_cam[:, 2]))]
        return True
