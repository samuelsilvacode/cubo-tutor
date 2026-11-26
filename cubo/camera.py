"""Leitura da webcam em paralelo: o app pega sempre o quadro mais novo, sem esperar a câmera."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2


class Camera:
    def __init__(self, indice: int = 0, largura: int = 1280, zoom: float = 1.0, mira=(0.5, 0.5)):
        """zoom > 1 recorta em volta de `mira` (fração da imagem); captura em 1080p para não perder detalhe."""
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 if zoom > 1 else 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 if zoom > 1 else 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._aplica_ajustes()
        self.largura = largura
        self.zoom = zoom
        self.mira = mira
        self._quadro = None
        self._numero = 0
        self._entregue = -1
        self._trava = threading.Lock()
        self._rodando = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _aplica_ajustes(self):
        """Brilho/contraste/saturação escolhidos por tools/otimizar_camera.py, se houver."""
        arq = Path(__file__).resolve().parent.parent / "calibracao" / "camera.json"
        if not arq.exists():
            return
        props = {"brilho": cv2.CAP_PROP_BRIGHTNESS, "contraste": cv2.CAP_PROP_CONTRAST,
                 "saturacao": cv2.CAP_PROP_SATURATION}
        for nome, valor in json.loads(arq.read_text(encoding="utf8")).items():
            if nome in props:
                self.cap.set(props[nome], valor)

    def _loop(self):
        while self._rodando:
            ok, fr = self.cap.read()
            if not ok:
                continue
            if self.zoom > 1:
                fr = recorta(fr, self.zoom, self.mira)
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


def recorta(fr, zoom: float, mira=(0.5, 0.5)):
    """Pedaço do quadro 1/zoom do tamanho, centrado em `mira` sem sair da imagem."""
    h, w = fr.shape[:2]
    rw, rh = int(w / zoom), int(h / zoom)
    x = min(max(int(mira[0] * w - rw / 2), 0), w - rw)
    y = min(max(int(mira[1] * h - rh / 2), 0), h - rh)
    return fr[y:y + rh, x:x + rw]
