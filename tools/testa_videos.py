"""Mede o detector em vídeos do cubo, opcionalmente piorando a imagem para parecer webcam."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubo.detector import achar_face  # noqa: E402


def piora(fr, rng):
    h, w = fr.shape[:2]
    p = cv2.resize(fr, (w // 3, h // 3))
    p = cv2.GaussianBlur(p, (5, 5), 0)
    p = cv2.resize(p, (w, h))
    p = p.astype(float) * np.array([0.9, 1.0, 1.08]) * 0.85 + 20  # mais lavado e quente
    p += rng.normal(0, 6, p.shape)
    p = cv2.imdecode(cv2.imencode(".jpg", np.clip(p, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 50])[1], 1)
    return p


for caminho in sys.argv[1:]:
    cap = cv2.VideoCapture(caminho)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng = np.random.default_rng(0)
    bom = ruim = tot = 0
    for k in range(0, n, 6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (int(w * 720 / max(h, w)), int(h * 720 / max(h, w))))
        tot += 1
        bom += achar_face(fr) is not None
        ruim += achar_face(piora(fr, rng)) is not None
    print(f"{Path(caminho).name}: original {bom}/{tot} · imitando webcam {ruim}/{tot}")
