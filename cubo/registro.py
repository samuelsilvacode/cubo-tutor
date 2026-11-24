"""Log da sessão: o que a câmera viu e o que o programa entendeu."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent / "logs"


class Registro:
    def __init__(self):
        self.pasta = RAIZ / time.strftime("sessao-%Y%m%d-%H%M%S")
        (self.pasta / "telas").mkdir(parents=True, exist_ok=True)
        (self.pasta / "cru").mkdir(exist_ok=True)
        self.eventos = open(self.pasta / "eventos.jsonl", "a", encoding="utf8")
        self.resumo = open(self.pasta / "resumo.log", "a", encoding="utf8")
        self.inicio = time.time()
        self.ult_quadro = self.ult_foto = self.ult_resumo = 0.0
        self.janela: list[dict] = []
        self.n_foto = 0

    def evento(self, tipo: str, **dados):
        dados.update(tipo=tipo, t=round(time.time() - self.inicio, 2))
        self.eventos.write(json.dumps(dados, ensure_ascii=False, default=str) + "\n")
        self.eventos.flush()

    def quadro(self, info: dict, tela, cru):
        agora = time.time()
        self.janela.append(info)
        if agora - self.ult_quadro > 0.2:
            self.evento("quadro", **info)
            self.ult_quadro = agora
        if agora - self.ult_foto > 1:
            self.n_foto += 1
            cv2.imwrite(str(self.pasta / "telas" / f"{self.n_foto:04d}.jpg"), cv2.resize(tela, None, fx=0.6, fy=0.6))
            if cru is not None and self.n_foto % 2 == 0:
                cv2.imwrite(str(self.pasta / "cru" / f"{self.n_foto:04d}.jpg"), cru)
            self.ult_foto = agora
        if agora - self.ult_resumo > 10:
            self._resumo()
            self.ult_resumo = agora

    def _resumo(self):
        j, self.janela = self.janela, []
        if not j:
            return
        n = len(j)
        fontes = Counter(q.get("fonte") for q in j)
        nq = [q["quadrados"] for q in j if q.get("quadrados") is not None]
        partes = [
            f"t={time.time() - self.inicio:.0f}s",
            f"modo={j[-1].get('modo')}",
            f"quadros={n}",
            f"face={100 * (n - fontes[None]) / n:.0f}% (detector {100 * fontes['detector'] / n:.0f}%, "
            f"grade {100 * fontes['grade'] / n:.0f}%)",
        ]
        if nq:
            partes.append(f"pecas_vistas_media={sum(nq) / len(nq):.1f}")
        if j[-1].get("grupos") is not None:
            partes.append("grupos=" + " ".join(f"{g['votos']}:{g['cores']}" for g in j[-1]["grupos"]))
        conf = [q["confere"] for q in j if q.get("confere") is not None]
        if conf:
            partes.append(f"confere={100 * sum(conf) / len(conf):.0f}% giro={j[-1].get('giro')}")
        linha = " | ".join(partes)
        self.resumo.write(linha + "\n")
        self.resumo.flush()

    def fecha(self):
        self._resumo()
        self.resumo.write("FIM\n")
        self.resumo.flush()
        self.eventos.close()
        self.resumo.close()
