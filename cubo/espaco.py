"""Onde o cubo está no espaço, com duas câmeras.

Cada câmera acha uma face de frente e o solvePnP dá a posição do centro do cubo
em relação a ela. Mexendo o cubo, os pares de posições vistas ao mesmo tempo pelas
duas câmeras dizem onde uma câmera está em relação à outra (Kabsch). Depois disso
as duas leituras viram uma só, e quando uma câmera perde o cubo a outra segura.

Unidade: lado de uma peça (num cubo de 57 mm, ~1,9 cm).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CM_POR_PECA = 1.9
ARQUIVO = Path(__file__).resolve().parent.parent / "calibracao" / "estereo.json"

# os 9 adesivos de uma face, em peças, na ordem de leitura (de pé)
GRADE = np.array([(c, l, 0.0) for l in (-1, 0, 1) for c in (-1, 0, 1)], float)


def camera_k(tamanho, zoom: float = 1.0) -> np.ndarray:
    """Mesma lente estimada do pose.py; o zoom por recorte só aumenta a distância focal."""
    h, w = tamanho
    f = 0.75 * zoom * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def pose_da_face(centros, K):
    """(R, t, centro do cubo) na câmera. t é o centro da face; o cubo fica 1,5 peça atrás dela."""
    ok, rvecs, tvecs, erros = cv2.solvePnPGeneric(GRADE, np.asarray(centros, float), K, None,
                                                   flags=cv2.SOLVEPNP_IPPE)
    if not ok or not len(rvecs):
        return None
    i = int(np.argmin(np.ravel(erros)))
    R = cv2.Rodrigues(rvecs[i])[0]
    t = tvecs[i].ravel()
    normal = R[:, 2]
    if normal @ t < 0:  # a normal tem que apontar para longe da câmera, para dentro do cubo
        normal = -normal
    return R, t, t + 1.5 * normal


def kabsch(a, b):
    """R, t, escala que levam os pontos a nos pontos b (b ≈ s R a + t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    A, B = a - ca, b - cb
    u, s, vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1, 1, d])
    R = u @ D @ vt
    escala = (s * np.diag(D)).sum() / max((A ** 2).sum(), 1e-9)
    return R, cb - escala * R @ ca, escala


class Estereo:
    """Posição da câmera B vista da câmera A, aprendida mexendo o cubo."""

    def __init__(self, minimo: int = 25):
        self.pares: list[tuple[np.ndarray, np.ndarray]] = []
        self.minimo = minimo
        self.R = self.t = None
        self.escala = 1.0
        self.erro = None
        self.aceitos = 0

    @property
    def pronto(self) -> bool:
        return self.R is not None

    def adiciona(self, pa, pb, distancia_min: float = 0.8):
        """Guarda um par só se o cubo andou desde o último (pontos espalhados calibram melhor)."""
        pa, pb = np.asarray(pa, float), np.asarray(pb, float)
        if self.pares and min(np.linalg.norm(pa - p) for p, _ in self.pares) < distancia_min:
            return False
        self.pares = (self.pares + [(pa, pb)])[-200:]
        if len(self.pares) >= self.minimo:
            self.calibra()
        return True

    def calibra(self, tolerancia: float = 1.0, voltas: int = 300, semente: int = 0):
        """RANSAC: ajusta com 4 pares sorteados e fica com o ajuste que mais pares aceitam
        (erro abaixo de `tolerancia` peça); leitura falsa de uma câmera não entorta o resultado."""
        a = np.array([p for p, _ in self.pares])
        b = np.array([q for _, q in self.pares])
        rng = np.random.default_rng(semente)
        melhor = None
        for _ in range(voltas):
            amostra = rng.choice(len(a), 4, replace=False)
            R, t, s = kabsch(a[amostra], b[amostra])
            if not 0.3 < s < 3:
                continue
            res = np.linalg.norm((s * a @ R.T + t) - b, axis=1)
            aceitos = res < tolerancia
            if melhor is None or aceitos.sum() > melhor.sum():
                melhor = aceitos
        if melhor is None or melhor.sum() < max(8, 0.5 * len(a)):
            return
        R, t, s = kabsch(a[melhor], b[melhor])
        res = np.linalg.norm((s * a[melhor] @ R.T + t) - b[melhor], axis=1)
        self.R, self.t, self.escala = R, t, s
        self.erro = float(np.median(res))
        self.aceitos = int(melhor.sum())

    def define(self, R, t, erro=0.0):
        """Calibração que veio de outro jeito (cubo parado): escala 1, sem pares."""
        self.R, self.t, self.escala, self.erro = np.asarray(R), np.asarray(t), 1.0, erro
        self.aceitos = 0

    def de_b_para_a(self, pb):
        return self.R.T @ (np.asarray(pb, float) - self.t) / self.escala

    def de_a_para_b(self, pa):
        return self.escala * self.R @ np.asarray(pa, float) + self.t

    def posicao_b(self):
        """Onde a câmera B fica, no espaço da câmera A."""
        return self.de_b_para_a(np.zeros(3))

    def salva(self, arquivo=ARQUIVO, **extra):
        arquivo.write_text(json.dumps({"R": self.R.tolist(), "t": self.t.tolist(), "escala": self.escala,
                                       "erro": self.erro, "pares": len(self.pares), **extra}, indent=1),
                           encoding="utf8")

    def carrega(self, arquivo=ARQUIVO) -> bool:
        if not arquivo.exists():
            return False
        d = json.loads(arquivo.read_text(encoding="utf8"))
        self.R, self.t, self.escala, self.erro = np.array(d["R"]), np.array(d["t"]), d["escala"], d.get("erro")
        return True


def escolhe(pa, pb_em_a, anterior=None, discorda: float = 2.0, salto: float = 8.0):
    """Posição do quadro a partir das duas câmeras, sem deixar uma detecção falsa passar.
    Se elas discordam, vale a que está mais perto da posição anterior; leitura sozinha que
    salta longe da anterior é ignorada (o cubo não pula meio metro entre dois quadros)."""
    lidas = [np.asarray(p, float) for p in (pa, pb_em_a) if p is not None]
    if not lidas:
        return None
    if len(lidas) == 2 and np.linalg.norm(lidas[0] - lidas[1]) < discorda:
        return junta(*lidas)
    if anterior is None:
        return lidas[0] if len(lidas) == 1 else None
    perto = min(lidas, key=lambda p: np.linalg.norm(p - anterior))
    return perto if np.linalg.norm(perto - anterior) < salto else None


def junta(pa, pb_em_a, peso_a: float = 0.5):
    """Uma posição a partir das duas câmeras; se só uma viu, vale ela."""
    if pa is None:
        return pb_em_a
    if pb_em_a is None:
        return pa
    return peso_a * np.asarray(pa) + (1 - peso_a) * np.asarray(pb_em_a)


# ---------- calibração com o cubo parado ----------

def eixos_da_face(centros, K):
    """Centro da face, eixos da grade (coluna, linha) e normal para dentro do cubo, na câmera."""
    pose = pose_da_face(centros, K)
    if pose is None:
        return None
    R, t, c = pose
    return t, R[:, 0], R[:, 1], (c - t) / 1.5


def _hipoteses(ea, eb):
    """(R, t) da câmera A para a B em cada jeito possível de a face da B estar no cubo:
    a mesma face da A (4 giros) ou uma vizinha dela (4 lados x 4 giros)."""
    fA, uA, vA, nA = ea
    fB, uB, vB, nB = eb
    MA = np.c_[uA, vA, nA]
    for X, Y in ((uB, vB), (vB, -uB), (-uB, -vB), (-vB, uB)):
        R = np.c_[X, Y, nB] @ MA.T
        yield R, fB - R @ fA
    cB = fB + 1.5 * nB
    for w in (uB, -uB, vB, -vB):
        e = uB if abs(uB @ w) < 0.5 else vB
        for X in (nB, -nB, e, -e):
            for Y in (nB, -nB, e, -e):
                if abs(X @ Y) > 0.5:
                    continue
                R = np.c_[X, Y, -w] @ MA.T
                if np.linalg.det(R) > 0:
                    yield R, (cB + 1.5 * w) - R @ fA


def _confere(pontos, normal_fora, cores, frame, K, cor_de):
    """Adesivos que a outra câmera deveria ver: (quantos batem, quantos deu para conferir)."""
    h, w = frame.shape[:2]
    bate = conferidos = 0
    for p, c in zip(pontos, cores):
        if p[2] <= 0 or normal_fora @ (-p) < 0.25 * np.linalg.norm(p):
            continue  # de costas ou quase de perfil para essa câmera
        u, v, s = K @ p
        x, y = int(u / s), int(v / s)
        if not (3 <= x < w - 3 and 3 <= y < h - 3):
            continue
        conferidos += 1
        bate += cor_de(frame[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).mean(axis=0).astype(np.uint8)) == c
    return bate, conferidos


def calibra_parado(fr_a, face_a, K_a, fr_b, face_b, K_b, cor_de):
    """Uma foto de cada câmera com o cubo parado basta: testa onde a face da B fica no cubo
    e fica com o jeito em que as cores de cada face caem certas na imagem da outra câmera.
    Devolve (R, t, acertos, conferidos) ou None se nenhum jeito se destacar."""
    ea, eb = eixos_da_face(face_a.centros, K_a), eixos_da_face(face_b.centros, K_b)
    if ea is None or eb is None:
        return None
    cores_a = [cor_de(x) for x in face_a.amostras(fr_a)]
    cores_b = [cor_de(x) for x in face_b.amostras(fr_b)]
    pts_a = np.array([ea[0] + g[0] * ea[1] + g[1] * ea[2] for g in GRADE])
    pts_b = np.array([eb[0] + g[0] * eb[1] + g[1] * eb[2] for g in GRADE])
    notas = []
    for R, t in _hipoteses(ea, eb):
        b1, c1 = _confere(pts_a @ R.T + t, -(R @ ea[3]), cores_a, fr_b, K_b, cor_de)
        b2, c2 = _confere((pts_b - t) @ R, -(R.T @ eb[3]), cores_b, fr_a, K_a, cor_de)
        notas.append((b1 + b2, c1 + c2, R, t))
    notas.sort(key=lambda n: -n[0])
    melhor, segundo = notas[0], notas[1]
    if melhor[0] < 5 or melhor[1] - melhor[0] > 1 or melhor[0] - segundo[0] < 3:
        return None
    return melhor[2], melhor[3], melhor[0], melhor[1]


# ---------- achar o cubo sem ler a grade ----------

# faixas de matiz (OpenCV, 0-180) das cores fortes do cubo
FAIXAS = {"R": ((0, 6), (170, 180)), "O": ((7, 19),), "Y": ((20, 35),), "G": ((36, 85),), "B": ((86, 130),)}


def manchas_do_cubo(frame, escala: float = 0.5, sat_min: int = 150, min_cores: int = 2):
    """Candidatos a cubo: manchas com várias cores bem saturadas juntas.
    Devolve [(cx, cy, lado, n_cores)] em pixels do quadro, a melhor primeiro."""
    peq = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    forte = (s >= sat_min) & (v >= 90)
    mascara = forte.astype(np.uint8) * 255
    # adesivos vizinhos (e a divisão preta entre eles) viram uma mancha só
    k = max(3, int(peq.shape[1] * 0.02)) | 1
    junta = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, rot, stats, cent = cv2.connectedComponentsWithStats(junta)
    area_min = peq.shape[0] * peq.shape[1] * 0.002
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < area_min or max(w, hh) > 3 * min(w, hh):
            continue
        dentro = (rot == i) & forte
        total = int(dentro.sum())
        if total < area_min * 0.5:
            continue
        hs = h[dentro]
        cores = sum(1 for faixas in FAIXAS.values()
                    if sum(((hs >= a) & (hs <= b)).sum() for a, b in faixas) > 0.06 * total)
        if cores < min_cores:
            continue
        ys, xs = np.nonzero(dentro)
        out.append((float(xs.mean() / escala), float(ys.mean() / escala),
                    float(max(w, hh) / escala), cores, float(total)))
    out.sort(key=lambda m: -(m[3] * 1e6 + m[4]))
    return [m[:4] for m in out]


def triangula(uv_a, uv_b, K_a, K_b, R, t):
    """Ponto 3D (no espaço da câmera A) que as duas câmeras veem nesses pixels,
    e o erro de reprojeção médio em pixels (alto = os dois pixels não são o mesmo objeto)."""
    Pa = K_a @ np.c_[np.eye(3), np.zeros(3)]
    Pb = K_b @ np.c_[R, t]
    X = cv2.triangulatePoints(Pa, Pb, np.float64(uv_a).reshape(2, 1), np.float64(uv_b).reshape(2, 1))
    p = (X[:3] / X[3]).ravel()
    erro = 0.0
    for P, uv in ((Pa, uv_a), (Pb, uv_b)):
        q = P @ np.r_[p, 1]
        erro += np.linalg.norm(q[:2] / q[2] - np.asarray(uv)) / 2
    return p, float(erro)


def melhor_par(cands_a, cands_b, K_a, K_b, R, t, erro_max: float = 30.0):
    """Entre os candidatos das duas câmeras, o par que a geometria diz ser o mesmo objeto."""
    melhor = None
    for ca in cands_a[:4]:
        for cb in cands_b[:4]:
            p, erro = triangula(ca[:2], cb[:2], K_a, K_b, R, t)
            if p[2] <= 0 or (R @ p + t)[2] <= 0 or erro > erro_max:
                continue
            if melhor is None or erro < melhor[1]:
                melhor = (p, erro, ca, cb)
    return melhor
