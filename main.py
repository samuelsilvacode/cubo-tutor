import argparse
import random

from cubo.app import App
from cubo.modelo import Cubo, embaralhar


def main():
    ap = argparse.ArgumentParser(description="Lê o cubo mágico pela webcam e ensina a montar.")
    ap.add_argument("--camera", type=int, default=0, help="índice da webcam (padrão 0)")
    ap.add_argument("--demo", action="store_true", help="sem câmera, com um cubo embaralhado aleatório")
    ap.add_argument("--estado", help="54 letras na ordem U R F D L B (W Y G B R O)")
    args = ap.parse_args()

    if args.demo:
        mix = embaralhar(25, random.Random())
        print("embaralhamento:", mix)
        App(camera=None, estado=Cubo.resolvido().aplica(mix).texto()).rodar()
    else:
        App(camera=args.camera, estado=args.estado).rodar()


if __name__ == "__main__":
    main()
