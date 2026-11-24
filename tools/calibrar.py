"""Calibração do vermelho e do laranja da webcam.

Pede uma face que só tem vermelho (sem laranja) e depois uma que só tem laranja (sem vermelho).
Guarda as cores lidas em calibracao/cores.json. ESPAÇO pula para a próxima etapa, Q sai.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubo import detector, visao  # noqa: E402

ETAPAS = [
    ("vermelho", "Mostre a face que tem VERMELHO e nao tem laranja"),
    ("laranja", "Mostre a face que tem LARANJA e nao tem vermelho"),
]
PASTA = Path(__file__).resolve().parent.parent / "calibracao"
PASTA.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
resultado, etapa, leituras = {}, 0, []
while etapa < len(ETAPAS):
    ok, fr = cap.read()
    if not ok:
        continue
    if fr.shape[1] > 1280:
        fr = cv2.resize(fr, (1280, int(fr.shape[0] * 1280 / fr.shape[1])), interpolation=cv2.INTER_AREA)
    nome, texto = ETAPAS[etapa]
    vis = fr.copy()
    face = detector.achar_face(fr)
    if face is not None and detector.nitidez(fr, face) > 40:
        amos = np.array(face.amostras(fr), float)
        if leituras and np.abs(amos - leituras[-1]).mean() > 25:
            leituras = []  # mexeu muito: recomeça
        leituras.append(amos)
        for (x, y), a in zip(face.centros.astype(int), amos):
            cv2.circle(vis, (x, y), 18, visao.BGR[visao.cor_rapida(a.astype(np.uint8))], -1)
            cv2.circle(vis, (x, y), 18, (0, 0, 0), 2)
    else:
        leituras = []
    prog = min(1.0, len(leituras) / 15)
    vis = cv2.flip(vis, 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(vis, f"{etapa + 1}/2  {texto}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.rectangle(vis, (0, 60), (int(vis.shape[1] * prog), 72), (0, 220, 0), -1)
    cv2.imshow("calibracao", vis)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    if len(leituras) >= 15:
        media = np.median(np.array(leituras), axis=0)
        resultado[nome] = media.round(1).tolist()
        cv2.imwrite(str(PASTA / f"{nome}.jpg"), fr)
        rapidas = "".join(visao.cor_rapida(a.astype(np.uint8)) for a in media)
        print(f"{nome}: lida como {rapidas} | BGR {[[int(v) for v in a] for a in media]}", flush=True)
        etapa, leituras = etapa + 1, []
        time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()
(PASTA / "cores.json").write_text(json.dumps(resultado, indent=1), encoding="utf8")
print("salvo em", PASTA / "cores.json", flush=True)
