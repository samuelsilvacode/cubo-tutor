"""Acha o cubo no espaço com duas câmeras e mostra num mapa visto de cima e de lado.

Uso: python tools/cubo_3d.py [camera_a] [camera_b] [segundos]
Câmera com zoom e mira: 0:1.5:0.5:0.62 (indice:zoom:x:y).
Teclas: C recomeça a calibração entre as câmeras · Q sai.

Calibração: deixe o cubo parado onde as duas câmeras vejam uma face cada.
O programa testa onde a face de uma fica no cubo em relação à da outra e
junta 8 tentativas em que as cores batem e tira a média das que concordam.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from cubo import detector, espaco, obliqua, visao  # noqa: E402
from cubo.camera import Camera  # noqa: E402


def le_camera(txt):
    p = [float(v) for v in txt.split(":")]
    return int(p[0]), (p[1] if len(p) > 1 else 1.0), (tuple(p[2:4]) if len(p) > 3 else (0.5, 0.5))


ca = le_camera(sys.argv[1] if len(sys.argv) > 1 else "0:1.5:0.5:0.62")
cb = le_camera(sys.argv[2] if len(sys.argv) > 2 else "1:1.5")
duracao = float(sys.argv[3]) if len(sys.argv) > 3 else 600
pasta = RAIZ / "logs" / time.strftime("espaco-%Y%m%d-%H%M%S")
pasta.mkdir(parents=True)
dados = open(pasta / "posicoes.jsonl", "w", encoding="utf8")

cams = [Camera(i, zoom=z, mira=m) for i, z, m in (ca, cb)]
zooms = [ca[1], cb[1]]
est = espaco.Estereo()
config = {"cameras": [ca[0], cb[0]], "zooms": zooms}
if est.carrega():
    salvo = json.loads(espaco.ARQUIVO.read_text(encoding="utf8"))
    if salvo.get("cameras") != config["cameras"] or salvo.get("zooms") != zooms:
        est = espaco.Estereo()  # calibração de outra montagem não serve
        print("calibração salva é de outra montagem: calibrando de novo", flush=True)
    else:
        print("calibração carregada", flush=True)

rastro = deque(maxlen=90)
suave = None
ant_a = ant_b = None
ult_tentativa = 0.0
seguidas = []
perdido = 0


def ve_cubo(fr, zoom):
    # face de frente pelo detector normal; de lado (losangos) pelo detector de face inclinada
    face = detector.achar_face(fr) or obliqua.achar_face(fr)
    if face is None:
        return None, None
    pose = espaco.pose_da_face(face.centros, espaco.camera_k(fr.shape[:2], zoom))
    return face, pose


def para_usuario(p):
    """Câmera (x direita da imagem, y baixo, z longe) -> quem está de frente: X sua direita, Y cima, Z distância."""
    return np.array([-p[0], -p[1], p[2]]) * espaco.CM_POR_PECA


def desenha_vista(fr, face, ponto, K, nome, cor, manchas=()):
    vis = fr.copy()
    for x, y, lado, _ in manchas:
        cv2.rectangle(vis, (int(x - lado / 2), int(y - lado / 2)), (int(x + lado / 2), int(y + lado / 2)), cor, 2)
    if face is not None:
        for x, y in face.centros.astype(int):
            cv2.circle(vis, (x, y), 10, cor, 3)
    if ponto is not None and ponto[2] > 0:
        u, v, w = K @ ponto
        cv2.drawMarker(vis, (int(u / w), int(v / w)), (255, 255, 255), cv2.MARKER_CROSS, 60, 4)
    h, w = vis.shape[:2]
    # mira: para calibrar, o cubo tem que passar perto do centro das duas câmeras
    cv2.line(vis, (w // 2 - 50, h // 2), (w // 2 + 50, h // 2), (255, 255, 255), 2)
    cv2.line(vis, (w // 2, h // 2 - 50), (w // 2, h // 2 + 50), (255, 255, 255), 2)
    vis = cv2.resize(cv2.flip(vis, 1), (480, 270))
    cv2.putText(vis, nome, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(vis, nome, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    return vis


def mapa(titulo, rotulos, de_lado, pos_cams, pontos, atual, larg=480, alt=420, escala=4.5):
    """Corte do espaço em cm. De cima: câmera A no topo, distância para baixo.
    De lado: câmera A à esquerda, distância para a direita, altura para cima."""
    img = np.full((alt, larg, 3), 32, np.uint8)

    def px(p):
        if de_lado:
            return int(40 + p[2] * escala), int(alt // 2 - 20 - p[1] * escala)
        return int(larg // 2 + p[0] * escala), int(40 + p[2] * escala)

    for cm in range(10, 80, 10):
        if de_lado:
            x = 40 + int(cm * escala)
            cv2.line(img, (x, 0), (x, alt - 50), (50, 50, 50), 1)
            cv2.putText(img, f"{cm}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
        else:
            y = 40 + int(cm * escala)
            cv2.line(img, (0, y), (larg, y), (50, 50, 50), 1)
            cv2.putText(img, f"{cm} cm", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    for nome, p, cor in pos_cams:
        cv2.circle(img, px(p), 9, cor, -1)
        cv2.putText(img, nome, (px(p)[0] + 12, px(p)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
    pts = [px(p) for p in pontos]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (90, 160, 90), 2)
    if atual is not None:
        x, y = px(atual)
        cv2.rectangle(img, (x - 11, y - 11), (x + 11, y + 11), (80, 230, 255), -1)
    cv2.putText(img, titulo, (10, alt - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    cv2.putText(img, rotulos, (10, alt - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return img


inicio, ult_log = time.time(), 0.0
print(f"cameras {ca[0]} e {cb[0]}, {duracao:.0f} s, em {pasta}", flush=True)
while time.time() - inicio < duracao:
    frames = [c.ler()[0] for c in cams]
    if any(f is None for f in frames):
        cv2.waitKey(5)
        continue
    (fa, pa_pose), (fb, pb_pose) = (ve_cubo(f, z) for f, z in zip(frames, zooms))
    pa = pa_pose[2] if pa_pose else None
    pb = pb_pose[2] if pb_pose else None
    # leitura isolada costuma ser falsa: para calibrar, cada câmera tem que repetir o lugar do quadro anterior
    firmes = all(p is not None and q is not None and np.linalg.norm(p - q) < 1.0
                 for p, q in ((pa, ant_a), (pb, ant_b)))
    ant_a, ant_b = pa, pb
    if firmes and not est.pronto and time.time() - ult_tentativa > 0.5:
        ult_tentativa = time.time()
        Ka_, Kb_ = (espaco.camera_k(f.shape[:2], z) for f, z in zip(frames, zooms))
        r = espaco.calibra_parado(frames[0], fa, Ka_, frames[1], fb, Kb_, visao.cor_rapida)
        if r is not None:
            pos_b = -r[0].T @ r[1]
            seguidas = (seguidas + [(pos_b, r[0], r[1])])[-15:]
            print(f"tentativa: {r[2]}/{r[3]} cores batem, camera B em "
                  f"{(para_usuario(pos_b)).round(1).tolist()} cm ({len(seguidas)}/8)", flush=True)
            if len(seguidas) >= 8:
                # cada foto treme uns cm: fica com as tentativas perto da mediana e tira a média delas
                pos = np.array([q[0] for q in seguidas])
                bons = [q for q, p_ in zip(seguidas, pos) if np.linalg.norm(p_ - np.median(pos, axis=0)) < 2.5]
                if len(bons) >= 6:
                    u_, _, vt_ = np.linalg.svd(sum(q[1] for q in bons))
                    est.define(u_ @ vt_, np.mean([q[2] for q in bons], axis=0))
                    est.salva(**config)
                    print("calibrado e salvo", flush=True)
    pb_em_a = est.de_b_para_a(pb) if (pb is not None and est.pronto) else None
    Ka, Kb = (espaco.camera_k(f.shape[:2], z) for f, z in zip(frames, zooms))
    # sem precisar da grade: a mancha colorida do cubo em cada câmera, cruzada pela geometria das duas
    manchas = [espaco.manchas_do_cubo(f) for f in frames] if est.pronto else [[], []]
    par = espaco.melhor_par(manchas[0], manchas[1], Ka, Kb, est.R, est.t) if est.pronto else None
    fonte = None
    if par is not None:
        agora_a, fonte = par[0], "triangulado"
    else:
        agora_a = espaco.escolhe(pa, pb_em_a, suave)
        if agora_a is not None:
            fonte = "pela face"
        elif suave is not None and est.pronto and (manchas[0] or manchas[1]):
            # só uma câmera vê um pedaço do cubo: direção dela, distância da última posição
            if manchas[0]:
                agora_a = espaco.no_raio(manchas[0][0][:2], Ka, np.linalg.norm(suave))
                fonte = "so A (pedaco)"
            else:
                cb_ = est.posicao_b()
                agora_a = espaco.no_raio(manchas[1][0][:2], Kb, np.linalg.norm(suave - cb_), cb_, est.R.T)
                fonte = "so B (pedaco)"
            if np.linalg.norm(agora_a - suave) > 8:
                agora_a, fonte = None, None
    if agora_a is not None:
        suave = agora_a if suave is None or np.linalg.norm(agora_a - suave) > 4 else 0.6 * agora_a + 0.4 * suave
        rastro.append(para_usuario(suave))
        perdido = 0
    else:
        # leitura recusada como salto: segura a posição por um tempo; sem leitura nenhuma, perde
        perdido += 1
        if (pa is None and pb_em_a is None) or perdido > 15:
            suave = None

    vista_a = desenha_vista(frames[0], fa, suave, Ka, "A (cima)", (80, 200, 255), manchas[0][:1])
    ponto_b = est.de_a_para_b(suave) if (suave is not None and est.pronto) else (pb if not est.pronto else None)
    vista_b = desenha_vista(frames[1], fb, ponto_b, Kb, "B (baixo)", (120, 255, 120), manchas[1][:1])

    cams_mapa = [("A", np.zeros(3), (80, 200, 255))]
    if est.pronto:
        cams_mapa.append(("B", para_usuario(est.posicao_b()), (120, 255, 120)))
    atual = para_usuario(suave) if suave is not None else None
    topo = mapa("visto de cima", "esquerda/direita (X) e distancia (Z)", False, cams_mapa, list(rastro), atual)
    lado = mapa("visto de lado", "distancia (Z) e altura (Y)", True, cams_mapa, list(rastro), atual)

    painel = np.full((150, 960, 3), 20, np.uint8)
    if est.pronto:
        status = "calibrado: B convertida para o espaco da A"
    else:
        status = f"calibrando ({len(seguidas)}/8): deixe o cubo parado, cada camera vendo uma face"
    cv2.putText(painel, status, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    if atual is not None:
        txt = f"X {atual[0]:+6.1f} cm   Y {atual[1]:+6.1f} cm   Z {atual[2]:5.1f} cm   ({fonte or 'segurando'})"
        cv2.putText(painel, txt, (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 230, 255), 2)
    else:
        cv2.putText(painel, "cubo nao encontrado", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
    cv2.putText(painel, "C recalibra  Q sai", (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    tela = np.vstack([np.hstack([vista_a, vista_b]), np.hstack([topo, lado]), painel])
    cv2.imshow("cubo no espaco", tela)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    if k == ord("c"):
        est, seguidas = espaco.Estereo(), []
        print("calibração reiniciada", flush=True)

    t = time.time()
    if t - ult_log >= 1.0:
        ult_log = t
        cv2.imwrite(str(pasta / f"{int(t - inicio):04d}.jpg"), cv2.resize(tela, None, fx=0.6, fy=0.6))
        reg = {"t": round(t - inicio, 1), "a": None if pa is None else para_usuario(pa).round(1).tolist(),
               "b": None if pb is None else para_usuario(pb).round(1).tolist(),
               "b_em_a": None if pb_em_a is None else para_usuario(pb_em_a).round(1).tolist(),
               "calibrado": est.pronto, "fonte": fonte,
               "tri": None if par is None else para_usuario(par[0]).round(1).tolist(),
               "erro_px": None if par is None else round(par[1], 1)}
        dados.write(json.dumps(reg) + "\n")
        dados.flush()
        dif = (None if reg["a"] is None or reg["b_em_a"] is None
               else round(float(np.linalg.norm(np.subtract(reg["a"], reg["b_em_a"]))), 1))
        print(f"t={reg['t']:5.1f} {fonte} tri={reg['tri']} ({reg['erro_px']} px) face A={reg['a']} "
              f"B->A={reg['b_em_a']} diferenca={dif}cm", flush=True)

if est.pronto:
    est.salva(**config)
for c in cams:
    c.release()
cv2.destroyAllWindows()
print("FIM", flush=True)
