"""Grava quadros da webcam com o que o detector achou, para analisar problemas de leitura."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubo import detector, visao  # noqa: E402

pasta = Path(sys.argv[1] if len(sys.argv) > 1 else "diagnostico")
segundos = float(sys.argv[2]) if len(sys.argv) > 2 else 25
pasta.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
inicio, ultimo, n, achou = time.time(), 0, 0, 0
while time.time() - inicio < segundos:
    ok, fr = cap.read()
    if not ok:
        continue
    vis = fr.copy()
    face = detector.achar_face(fr)
    for x, y, w, h, *_ in detector._quadrados(fr):
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 255), -1)
    info = "nada"
    if face is not None:
        achou += 1
        nit = detector.nitidez(fr, face)
        cores = "".join(visao.cor_rapida(a) for a in face.amostras(fr))
        info = f"face {cores} nitidez {nit:.0f}"
        for (x, y) in face.centros.astype(int):
            cv2.circle(vis, (x, y), 18, (0, 255, 0), 3)
    cv2.putText(vis, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("diagnostico", vis)
    cv2.waitKey(1)
    if time.time() - ultimo > 0.5:
        cv2.imwrite(str(pasta / f"{n:03d}.jpg"), vis)
        cv2.imwrite(str(pasta / f"{n:03d}_cru.jpg"), fr)
        ultimo, n = time.time(), n + 1
cap.release()
cv2.destroyAllWindows()
print(f"{n} fotos salvas em {pasta}, face achada em {achou} quadros")
