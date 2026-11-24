"""Onde o cubo está na mão: qual face está virada para a câmera e a rotação 3D dele.

Dois modos que se completam:
- grade: quando uma face aparece de frente, solvePnP nos 9 adesivos dá a pose do zero;
- modelo: entre um quadro e outro, ajusta a pose anterior até as cores projetadas
  de todos os adesivos visíveis baterem com a imagem (funciona com quina e aresta).
"""
from __future__ import annotations

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
    def _mapa_cores(self, frame, ref: visao.Referencia, mao=None):
        peq = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
        h, w = peq.shape[:2]
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
        # nota só sobre o que dá pra ver (sem mão e sem fundo)
        return (bate - 0.5 * erra) / contam, contam

    def pelo_modelo(self, frame, cubo: Cubo, ref: visao.Referencia, mao=None) -> bool:
        if self.R_cam is None:
            return False
        mapa = self._mapa_cores(frame, ref, mao)
        idx = {c: i for i, c in enumerate(ref.nomes)}
        esperado = np.array([idx.get(c, -2) for c in cubo.texto()])[ADESIVO]
        K = _camera(frame.shape[:2])
        R, t = self.R_cam, self.t.copy()
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
        self._nota(R, t, mapa, esperado, K, guardar=True)
        self.qualidade = nota
        if nota < 0.35:
            return False
        self.R_cam, self.t, self.fonte = _ortonormal(R), t, "modelo"
        return True

    # ---------- um quadro ----------
    def atualiza(self, frame, centros, lida, cubo: Cubo, ref, espelho=False, mao=None) -> bool:
        ok = False
        if centros is not None and lida is not None:
            ok = self.pela_grade(centros, lida, cubo, frame.shape[:2])
        ok = self.pelo_modelo(frame, cubo, ref, mao) or ok
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
