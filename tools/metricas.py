"""Relatório de assertividade a partir dos logs das sessões."""
import json
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def sessao(pasta: Path) -> dict | None:
    arq = pasta / "eventos.jsonl"
    if not arq.exists():
        return None
    ev = [json.loads(l) for l in arq.open(encoding="utf8") if l.strip()]
    quadros = [e for e in ev if e["tipo"] == "quadro" and e.get("modo") == "escanear"]
    aula = [e for e in ev if e["tipo"] == "quadro" and e.get("modo") == "aula" and e.get("confere") is not None]
    montagens = [e for e in ev if e["tipo"] == "montagem"]
    # tempo até montar: do início (ou do último recomeço) até a montagem
    tempos, inicio = [], 0.0
    for e in ev:
        if e["tipo"] == "reinicio":
            inicio = e["t"]
        elif e["tipo"] == "montagem":
            tempos.append(e["t"] - inicio)
    corr = [sum(a != b for a, b in zip(m["cruas"], m["certas"])) for m in montagens]
    avancos = [e for e in ev if e["tipo"] == "avanco"]
    return {
        "nome": pasta.name,
        "quadros": len(quadros),
        "face_achada": sum(q.get("fonte") is not None for q in quadros) / len(quadros) if quadros else None,
        "montagens": len(montagens),
        "tempo_montar": sum(tempos) / len(tempos) if tempos else None,
        "corrigidos": sum(corr) / len(corr) if corr else None,
        "montagem_limpa": sum(c <= 2 for c in corr) / len(corr) if corr else None,
        "aula_confere": sum(q["confere"] for q in aula) / len(aula) if aula else None,
        "giros_auto": sum(a["para"] - a["de"] for a in avancos),
    }


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def num(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


if __name__ == "__main__":
    pastas = sorted(LOGS.glob("sessao-*"))
    if len(sys.argv) > 1:
        pastas = pastas[-int(sys.argv[1]):]
    linhas = [s for s in map(sessao, pastas) if s]
    print(f"{'sessão':24} {'face achada':>11} {'montagens':>9} {'tempo p/ montar':>15} "
          f"{'corrigidos':>10} {'montagem limpa':>14} {'aula confere':>12} {'giros auto':>10}")
    for s in linhas:
        print(f"{s['nome']:24} {pct(s['face_achada']):>11} {s['montagens']:>9} "
              f"{num(s['tempo_montar'], '{:.0f} s'):>15} {num(s['corrigidos']):>10} "
              f"{pct(s['montagem_limpa']):>14} {pct(s['aula_confere']):>12} {s['giros_auto']:>10}")
