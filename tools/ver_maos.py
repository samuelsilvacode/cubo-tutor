"""Câmera com o esqueleto das mãos e a área que o rastreador vai ignorar. Q sai."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubo.mao import Maos  # noqa: E402

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
maos = Maos()
ultimo, fps = time.time(), 0.0
while True:
    ok, fr = cap.read()
    if not ok:
        continue
    maos.atualiza(fr)
    vis = fr.copy()
    m = maos.mascara(fr.shape, frame=fr)
    vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    maos.desenha(vis)
    vis = cv2.flip(vis, 1)
    agora = time.time()
    fps = 0.9 * fps + 0.1 / max(agora - ultimo, 1e-3)
    ultimo = agora
    cv2.putText(vis, f"maos: {len(maos.maos)}  |  {fps:.0f} fps  |  roxo = area ignorada", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("esqueleto das maos", vis)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
cap.release()
cv2.destroyAllWindows()
