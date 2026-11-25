"""Testa combinações de brilho/contraste/saturação da webcam com o cubo na frente
e escolhe a que mais separa vermelho de laranja sem estourar a cor.

Segure parada, de frente, uma face que tenha vermelho E laranja. Q sai.
Grava a melhor combinação em calibracao/camera.json (o app aplica ao abrir a câmera).
"""
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from cubo import detector, visao  # noqa: E402

PROPS = {"brilho": cv2.CAP_PROP_BRIGHTNESS, "contraste": cv2.CAP_PROP_CONTRAST, "saturacao": cv2.CAP_PROP_SATURATION}
OPCOES = {"brilho": [128, 105], "contraste": [126, 140], "saturacao": [143, 115, 90]}


def separacao(amostras):
    """Quão separados ficam os grupos vermelho e laranja (pela proporção de verde) e quanto estourou."""
    quentes = [a for a in amostras if visao._parece_vermelho_ou_laranja(a)]
    if len(quentes) < 2:
        return None
    p = np.sort([visao._proporcao_laranja(a) for a in quentes])
    # maior buraco entre valores vizinhos = fronteira entre vermelho e laranja
    buracos = np.diff(p)
    k = int(np.argmax(buracos))
    baixo, alto = p[:k + 1], p[k + 1:]
    espalhado = (baixo.std() if len(baixo) > 1 else 0) + (alto.std() if len(alto) > 1 else 0) + 0.01
    estouro = np.mean([max(a) >= 250 for a in quentes])
    return buracos[k] / espalhado, estouro, len(baixo), len(alto)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
originais = {n: cap.get(p) for n, p in PROPS.items()}
resultados = []
combos = list(itertools.product(*OPCOES.values()))
print(f"testando {len(combos)} combinações; segure uma face com vermelho e laranja", flush=True)
i = 0
while i < len(combos):
    combo = dict(zip(OPCOES, combos[i]))
    for n, v in combo.items():
        cap.set(PROPS[n], v)
    inicio, leituras, desde = time.time(), [], None
    # só conta tempo com o cubo à vista: pausa se ele sair
    while len(leituras) < 10:
        ok, fr = cap.read()
        if not ok:
            continue
        peq = cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA)
        face = detector.achar_face(peq)
        vis = peq.copy()
        txt = f"{i + 1}/{len(combos)}  brilho {combo['brilho']} contraste {combo['contraste']} saturacao {combo['saturacao']}"
        if face is not None:
            desde = desde or time.time()
            for (x, y), a in zip(face.centros.astype(int), face.amostras(peq)):
                cv2.circle(vis, (x, y), 16, visao.BGR[visao.cor_rapida(a)], -1)
                cv2.circle(vis, (x, y), 16, (0, 0, 0), 2)
            if time.time() - desde > 0.8:  # espera a câmera se ajustar ao novo valor
                leituras.append(np.array(face.amostras(peq), float))
            txt += f"  lendo {len(leituras)}/10"
        else:
            desde = None
            txt += "  MOSTRE A FACE COM VERMELHO E LARANJA"
        vis = cv2.flip(vis, 1)
        cv2.putText(vis, txt, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("otimizar camera", vis)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            i = len(combos)
            break
    if leituras:
        s = separacao(np.median(np.array(leituras), axis=0))
        if s:
            resultados.append((combo, s))
            print(f"{combo} -> separação {s[0]:.1f} | estouradas {100 * s[1]:.0f}% | "
                  f"{s[2]} vermelhas, {s[3]} laranjas", flush=True)
        else:
            print(f"{combo} -> não achei vermelho/laranja na face", flush=True)
    else:
        print(f"{combo} -> não vi a face", flush=True)
    i += 1
for n, v in originais.items():
    cap.set(PROPS[n], v)
cap.release()
cv2.destroyAllWindows()
if resultados:
    # cor estourada não é confiável: a separação dela pode estar separando os grupos errados
    # (no teste da K420 a combinação de fábrica, 100% estourada, tinha a maior "separação" e errou)
    confiaveis = [r for r in resultados if r[1][1] <= 0.5] or resultados
    melhor = max(confiaveis, key=lambda r: r[1][0] - 3 * r[1][1])
    (RAIZ / "calibracao").mkdir(exist_ok=True)
    (RAIZ / "calibracao" / "camera.json").write_text(json.dumps(melhor[0], indent=1), encoding="utf8")
    print(f"MELHOR: {melhor[0]} (separação {melhor[1][0]:.1f}, estouradas {100 * melhor[1][1]:.0f}%)", flush=True)
