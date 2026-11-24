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
        self.ult_espelho = 0.0
        self.n_espelho = 0

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
        if j[-1].get("lidas") is not None:
            partes.append(f"lidas={j[-1]['lidas'] or '-'} pedida={j[-1].get('pedida')}")
        if j[-1].get("aviso"):
            partes.append(f"aviso={j[-1]['aviso']}")
        if j[-1].get("grupos") is not None:
            partes.append("grupos=" + " ".join(f"{g['votos']}:{g['cores']}" for g in j[-1]["grupos"]))
        esp = [q for q in j if q.get("modo") == "espelho"]
        if esp:
            com = [q for q in esp if q.get("face_pose")]
            qual = [q["qualidade"] for q in com if q.get("qualidade") is not None]
            partes.append(f"espelho: pose {100 * len(com) / len(esp):.0f}%"
                          f" | grade {sum(q.get('fonte_pose') == 'grade' for q in com)}"
                          f" modelo {sum(q.get('fonte_pose') == 'modelo' for q in com)}"
                          f" | cores batem {100 * sum(qual) / len(qual) if qual else 0:.0f}%"
                          f" | mão vista {100 * sum(q.get('maos', 0) > 0 for q in esp) / len(esp):.0f}% dos quadros"
                          f" | foto {self.n_espelho:04d}")
        conf = [q["confere"] for q in j if q.get("confere") is not None]
        if conf:
            partes.append(f"confere={100 * sum(conf) / len(conf):.0f}% giro={j[-1].get('giro')}")
        linha = " | ".join(partes)
        self.resumo.write(linha + "\n")
        self.resumo.flush()

    def foto_espelho(self, frame, rastreador, maos, achou, replica):
        """Foto de diagnóstico: pontos do modelo (verde bate, vermelho não, cinza ignorado) e esqueleto da mão."""
        agora = time.time()
        if agora - self.ult_espelho < 0.5:
            return
        self.ult_espelho = agora
        self.n_espelho += 1
        vis = frame.copy()
        maos.desenha(vis)
        if rastreador.pontos_debug is not None:
            uv, estado = rastreador.pontos_debug
            cores = {1: (60, 220, 60), -1: (40, 40, 230), 0: (160, 160, 160)}
            for (x, y), e in zip(uv.astype(int), estado):
                cv2.circle(vis, (int(x), int(y)), 5, cores[int(e)], -1)
        texto = (f"{'POSE OK' if achou else 'PERDEU'} via {rastreador.fonte} | "
                 f"cores batem {100 * max(0, rastreador.qualidade):.0f}% | mao {100 * rastreador.mao:.0f}% da imagem")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(vis, texto, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (60, 220, 60) if achou else (40, 40, 230), 2)
        h = vis.shape[0]
        rep = cv2.resize(replica, (h // 2, h // 2))
        vis[h - rep.shape[0]:, vis.shape[1] - rep.shape[1]:] = rep
        pasta = self.pasta / "espelho"
        pasta.mkdir(exist_ok=True)
        cv2.imwrite(str(pasta / f"{self.n_espelho:04d}.jpg"), cv2.resize(vis, None, fx=0.6, fy=0.6))

    def fecha(self):
        self._resumo()
        self.resumo.write("FIM\n")
        self.resumo.flush()
        self.eventos.close()
        self.resumo.close()
