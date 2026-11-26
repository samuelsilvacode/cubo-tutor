import cv2
import numpy as np

from cubo import espaco


def _rot(x, y, z):
    return cv2.Rodrigues(np.radians([x, y, z]).astype(float))[0]


def _projeta(R, t, K):
    p = espaco.GRADE @ R.T + t
    uv = p @ K.T
    return uv[:, :2] / uv[:, 2:3]


def test_pose_da_face_acha_centro_do_cubo():
    K = espaco.camera_k((720, 1280), zoom=1.5)
    R = _rot(15, -20, 5)
    t = np.array([1.0, -2.0, 14.0])
    _, t2, centro = espaco.pose_da_face(_projeta(R, t, K), K)
    assert np.allclose(t2, t, atol=1e-3)
    normal = R[:, 2] if R[:, 2] @ t > 0 else -R[:, 2]
    assert np.allclose(centro, t + 1.5 * normal, atol=1e-3)
    assert centro[2] > t[2]  # o cubo fica atrás da face


def test_kabsch_recupera_transformacao_e_escala():
    rng = np.random.default_rng(1)
    a = rng.uniform(-5, 5, (30, 3)) + [0, 0, 15]
    R, t, s = _rot(10, 25, -5), np.array([3.0, -8.0, 2.0]), 1.1
    R2, t2, s2 = espaco.kabsch(a, s * a @ R.T + t)
    assert np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-6) and abs(s2 - s) < 1e-6


def test_estereo_aprende_com_ruido_e_leitura_errada():
    rng = np.random.default_rng(2)
    R, t = _rot(0, 0, 20), np.array([0.0, 5.0, 1.0])
    e = espaco.Estereo(minimo=25)
    for i in range(40):
        pa = rng.uniform(-6, 6, 3) + [0, 0, 15]
        pb = R @ pa + t + rng.normal(0, 0.05, 3)
        if i % 10 == 0:
            pb = pb + 4  # leitura errada de vez em quando
        e.adiciona(pa, pb, distancia_min=0)
    assert e.pronto
    p = np.array([1.0, 2.0, 14.0])
    assert np.linalg.norm(e.de_b_para_a(R @ p + t) - p) < 0.15
    assert np.linalg.norm(e.posicao_b() - (-R.T @ t)) < 0.3


def test_junta_usa_a_que_viu():
    assert np.allclose(espaco.junta(None, [1, 2, 3]), [1, 2, 3])
    assert np.allclose(espaco.junta([0, 0, 0], [2, 2, 2]), [1, 1, 1])


def test_escolhe_descarta_deteccao_falsa():
    perto, longe = np.array([0, 0, 25.0]), np.array([0, 0, 64.0])
    assert np.allclose(espaco.escolhe(perto, perto + 0.5), perto + 0.25)
    assert np.allclose(espaco.escolhe(longe, perto + 0.3, anterior=perto), perto + 0.3)
    assert espaco.escolhe(longe, None, anterior=perto) is None
    assert espaco.escolhe(longe, perto, anterior=None) is None


def test_triangula_acha_o_ponto_e_recusa_par_errado():
    K = espaco.camera_k((720, 1280), zoom=1.5)
    R, t = _rot(0, -60, 0), np.array([20.0, 0.0, 12.0])
    p = np.array([2.0, -1.0, 24.0])

    def pixel(P):
        q = K @ P
        return q[:2] / q[2]

    achado, erro = espaco.triangula(pixel(p), pixel(R @ p + t), K, K, R, t)
    assert np.linalg.norm(achado - p) < 1e-3 and erro < 0.01
    outro = pixel(R @ (p + [6, 0, 0]) + t) + [0, 80]
    par = espaco.melhor_par([(*pixel(p), 1, 3)], [(*outro, 1, 3), (*pixel(R @ p + t), 1, 3)], K, K, R, t)
    assert np.linalg.norm(par[0] - p) < 1e-3
