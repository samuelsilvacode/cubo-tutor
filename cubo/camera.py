"""Leitura da webcam em paralelo: o app pega sempre o quadro mais novo, sem esperar a câmera."""
from __future__ import annotations

import threading

import cv2


class Camera:
    def __init__(self, indice: int = 0, largura: int = 1280):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.largura = largura
        self._quadro = None
        self._numero = 0
        self._entregue = -1
        self._trava = threading.Lock()
        self._rodando = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._rodando:
            ok, fr = self.cap.read()
            if not ok:
                continue
            if fr.shape[1] > self.largura:
                fr = cv2.resize(fr, (self.largura, int(fr.shape[0] * self.largura / fr.shape[1])),
                                interpolation=cv2.INTER_AREA)
            with self._trava:
                self._quadro = fr
                self._numero += 1

    def ler(self):
        """(quadro mais recente, se é novo desde a última leitura)."""
        with self._trava:
            if self._quadro is None:
                return None, False
            novo = self._numero != self._entregue
            self._entregue = self._numero
            return self._quadro.copy(), novo

    def release(self):
        self._rodando = False
        self._thread.join(timeout=1)
        self.cap.release()
