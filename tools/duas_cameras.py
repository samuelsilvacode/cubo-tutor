"""Abre duas câmeras lado a lado e mostra o que cada uma lê do cubo.

Uso: python tools/duas_cameras.py [camera_a] [camera_b] [segundos]   (Q sai)
Câmera pode levar zoom e mira: 0:1.5 ou 0:1.5:0.5:0.6 (indice:zoom:x:y).
Para cada câmera: face achada, tamanho do cubo na imagem, reflexo e brilho.
Fotos e dados vão para logs/duas-AAAAMMDD-HHMMSS/.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from cubo import detector, visao  # noqa: E402
from cubo.camera import Camera  # noqa: E402



def le_camera(txt):
    p = [float(v) for v in txt.split(":")]
    return int(p[0]), (p[1] if len(p) > 1 else 1.0), (tuple(p[2:4]) if len(p) > 3 else (0.5, 0.5))


ca = le_camera(sys.argv[1] if len(sys.argv) > 1 else "0")
cb = le_camera(sys.argv[2] if len(sys.argv) > 2 else "1")
ia, ib = ca[0], cb[0]
duracao = float(sys.argv[3]) if len(sys.argv) > 3 else 300
pasta = RAIZ / "logs" / time.strftime("duas-%Y%m%d-%H%M%S")
pasta.mkdir(parents=True)
dados = open(pasta / "leituras.jsonl", "w", encoding="utf8")


def _tem_reflexo(hsv, x, y, lado):
    """Reflexo é uma mancha estourada que cobre só parte do adesivo; adesivo branco inteiro não conta."""
    r = max(2, int(lado * 0.35))
    p = hsv[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
    if not len(p):
        return False
    estourado = ((p[:, 2] > 245) & (p[:, 1] < 40)).mean()
    return 0.05 < estourado < 0.6


def analisa(fr):
    """Leitura de uma câmera + números para ajustar a posição dela."""
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    info = {"brilho": round(float(hsv[..., 2].mean())),
            "estourado": round(float((hsv[..., 2] > 250).mean()) * 100, 1)}
    face = detector.achar_face(fr)
    if face is None:
        return info, None
    amos = face.amostras(fr)
    passo = np.linalg.norm(face.centros[1] - face.centros[0])
    cx, cy = face.centros[4]
    info.update({
        "cores": "".join(visao.cor_rapida(a) for a in amos),
        "tamanho": round(passo * 3 / h * 100),  # % da altura da imagem
        "centro": [round(cx / w * 100), round(cy / h * 100)],
        "reflexo": sum(1 for x, y in face.centros.astype(int) if _tem_reflexo(hsv, x, y, face.lado)),
        "nitidez": round(detector.nitidez(fr, face)),
        "bgr": [a.tolist() for a in amos],
    })
    return info, face


def dicas(info):
    if "cores" not in info:
        return ["sem face"]
    out = []
    if info["tamanho"] < 25:
        out.append("aproxime o cubo")
    elif info["tamanho"] > 60:
        out.append("afaste o cubo")
    if info["reflexo"]:
        out.append(f"reflexo em {info['reflexo']} pecas")
    x, y = info["centro"]
    if not 30 < x < 70 or not 25 < y < 75:
        out.append("cubo fora do centro")
    if info["brilho"] > 170 or info["estourado"] > 8:
        out.append("luz demais no quadro")
    return out or ["ok"]


def desenha(fr, face, info, nome):
    vis = fr.copy()
    if face is not None:
        for x, y in face.centros.astype(int):
            cv2.circle(vis, (x, y), 14, (0, 0, 0), 2)
        for (x, y), c in zip(face.centros.astype(int), info["cores"]):
            cv2.circle(vis, (x, y), 12, visao.BGR[c], -1)
    vis = cv2.flip(vis, 1)
    linhas = [nome] + dicas(info)
    for k, t in enumerate(linhas):
        cv2.putText(vis, t, (12, 34 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(vis, t, (12, 34 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return cv2.resize(vis, (800, 450))


cams = {i: Camera(i, zoom=z, mira=m) for i, z, m in (ca, cb)}
inicio, ultima, n = time.time(), 0.0, 0
print(f"cameras {ia} e {ib}, {duracao:.0f} s, em {pasta}", flush=True)
while time.time() - inicio < duracao:
    paineis, registro = [], {}
    for i, cam in cams.items():
        fr, _ = cam.ler()
        if fr is None:
            paineis.append(np.zeros((450, 800, 3), np.uint8))
            continue
        info, face = analisa(fr)
        registro[i] = info
        paineis.append(desenha(fr, face, info, f"camera {i}"))
    cv2.imshow("duas cameras", np.hstack(paineis))
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
    agora = time.time()
    if agora - ultima >= 1.0:
        ultima, n = agora, n + 1
        if n % 2 == 0:
            cv2.imwrite(str(pasta / f"{n:04d}.jpg"), np.hstack(paineis))
        dados.write(json.dumps({"n": n, "t": round(agora - inicio, 1), "cameras": registro}) + "\n")
        dados.flush()
        resumo = " | ".join(
            f"cam{i}: {r.get('cores', '---------')} tam {r.get('tamanho', '-')}% "
            f"reflexo {r.get('reflexo', '-')} brilho {r['brilho']} -> {', '.join(dicas(r))}"
            for i, r in registro.items())
        print(f"t={agora - inicio:5.1f}s {resumo}", flush=True)
for cam in cams.values():
    cam.release()
cv2.destroyAllWindows()
print("FIM", flush=True)
