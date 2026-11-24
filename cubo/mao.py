"""Esqueleto das mãos (MediaPipe): onde a mão está, para não confundir dedo com adesivo."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

MODELO = Path(__file__).resolve().parent.parent / "modelos" / "hand_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
OSSOS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PALMA = [0, 1, 5, 9, 13, 17]
LARGURA = 960  # o esqueleto roda numa cópia menor da imagem
GROSSURA = 0.22


class Maos:
    def __init__(self):
        self.ok = False
        self.maos: list[np.ndarray] = []
        if not MODELO.exists():
            print(f"modelo da mão não encontrado em {MODELO}; baixe de {URL_MODELO}")
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
        except ImportError:
            print("mediapipe não instalado: a mão não será ignorada no rastreamento")
            return
        self._mp = mp
        op = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(MODELO)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            # achar mão nova exige mais certeza (rosto não vira mão); seguir uma mão já achada exige menos
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.3)
        self._det = vision.HandLandmarker.create_from_options(op)
        self._t0 = time.time()
        self._ult_ms = -1
        self.ok = True

    def atualiza(self, frame) -> list[np.ndarray]:
        """Pontos (21 x 2, em pixels do frame) de cada mão vista."""
        if not self.ok:
            return []
        h, w = frame.shape[:2]
        k = LARGURA / w
        peq = cv2.resize(frame, (LARGURA, int(h * k)))
        ms = max(self._ult_ms + 1, int((time.time() - self._t0) * 1000))
        self._ult_ms = ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(peq, cv2.COLOR_BGR2RGB))
        r = self._det.detect_for_video(img, ms)
        self.maos = [np.array([(p.x * w, p.y * h) for p in mao])
                     for mao, lado in zip(r.hand_landmarks, r.handedness) if lado[0].score >= 0.6]
        return self.maos

    def mascara(self, forma, escala=1.0, frame=None) -> np.ndarray:
        """True onde tem mão. Com o frame, recorta o contorno real pela cor da pele."""
        h, w = forma[:2]
        m = np.zeros((h, w), np.uint8)
        for mao in self.maos:
            p = (mao * escala).astype(np.int32)
            palma = max(4, int(np.linalg.norm(p[5] - p[17])))
            if frame is None:
                m |= self._esqueleto(h, w, p, max(3, int(palma * GROSSURA)))
            else:
                m |= self._contorno(frame, escala, p, palma)
        return m.astype(bool)

    def _esqueleto(self, h, w, p, grossura):
        m = np.zeros((h, w), np.uint8)
        for a, b in OSSOS:
            cv2.line(m, tuple(p[a]), tuple(p[b]), 1, grossura)
        for q in p[[4, 8, 12, 16, 20]]:
            cv2.circle(m, tuple(q), grossura // 2, 1, -1)
        cv2.fillConvexPoly(m, cv2.convexHull(p[PALMA]), 1)
        return m

    def _contorno(self, frame, escala, p, palma):
        img = frame if escala == 1.0 else cv2.resize(frame, None, fx=escala, fy=escala)
        h, w = img.shape[:2]
        # onde a mão pode estar: o esqueleto bem engrossado
        zona = self._esqueleto(h, w, p, max(6, int(palma * 0.55)))
        # onde com certeza é mão: o esqueleto fino
        semente = self._esqueleto(h, w, p, max(2, int(palma * 0.08)))
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hist = cv2.calcHist([ycc], [1, 2], semente, [32, 32], [0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        prob = cv2.calcBackProject([ycc], [1, 2], hist, [0, 256, 0, 256], 1)
        prob = cv2.GaussianBlur(prob, (5, 5), 0)
        pele = ((prob > 40) & (zona > 0)).astype(np.uint8)
        pele = cv2.morphologyEx(pele, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        pele |= semente
        # só o que está ligado ao esqueleto
        n, rot = cv2.connectedComponents(pele)
        ligados = np.unique(rot[semente > 0])
        mao = np.isin(rot, ligados[ligados > 0]).astype(np.uint8)
        # margem de segurança em volta do contorno
        borda = max(3, int(palma * 0.07)) | 1
        return cv2.dilate(mao, np.ones((borda, borda), np.uint8))

    def desenha(self, img, escala=1.0):
        for mao in self.maos:
            p = (mao * escala).astype(int)
            for a, b in OSSOS:
                cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 255, 255), 3, cv2.LINE_AA)
            for q in p:
                cv2.circle(img, tuple(q), 5, (255, 0, 255), -1)
