"""Uso: python -m metricas VP VN FP FN

Sem argumentos, usa a matriz de exemplo do README (VP=40, VN=45, FP=5, FN=10).
"""

import sys

from .matriz import MatrizConfusao


def main(argv: list[str]) -> int:
    if argv and len(argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        matriz = MatrizConfusao(*map(int, argv)) if argv else MatrizConfusao(vp=40, vn=45, fp=5, fn=10)
    except ValueError as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2
    print(matriz.relatorio())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
