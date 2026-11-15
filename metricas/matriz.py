from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrizConfusao:
    """Matriz de confusão de um classificador binário.

    vp: positivos classificados como positivos (verdadeiros positivos)
    vn: negativos classificados como negativos (verdadeiros negativos)
    fp: negativos classificados como positivos (falsos positivos)
    fn: positivos classificados como negativos (falsos negativos)

    Quando uma métrica tem denominador zero (por exemplo, precisão de um modelo que
    nunca previu positivo), o valor é 0.0, a mesma convenção do scikit-learn.
    """

    vp: int
    vn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        for nome in ("vp", "vn", "fp", "fn"):
            valor = getattr(self, nome)
            if not isinstance(valor, int) or isinstance(valor, bool) or valor < 0:
                raise ValueError(f"{nome.upper()} precisa ser um inteiro não negativo, recebi {valor!r}.")
        if self.n == 0:
            raise ValueError("A matriz de confusão está vazia.")

    @classmethod
    def de_rotulos(cls, verdadeiros: Sequence[int], previstos: Sequence[int], positivo: int = 1) -> "MatrizConfusao":
        """Monta a matriz a partir dos rótulos reais e dos previstos pelo modelo."""
        if len(verdadeiros) != len(previstos):
            raise ValueError("As listas de rótulos precisam ter o mesmo tamanho.")
        pares = list(zip(verdadeiros, previstos, strict=True))
        return cls(
            vp=sum(v == positivo and p == positivo for v, p in pares),
            vn=sum(v != positivo and p != positivo for v, p in pares),
            fp=sum(v != positivo and p == positivo for v, p in pares),
            fn=sum(v == positivo and p != positivo for v, p in pares),
        )

    @staticmethod
    def _razao(numerador: int, denominador: int) -> float:
        return numerador / denominador if denominador else 0.0

    @property
    def n(self) -> int:
        return self.vp + self.vn + self.fp + self.fn

    @property
    def acuracia(self) -> float:
        """(VP + VN) / N: quanto o modelo acerta no total."""
        return (self.vp + self.vn) / self.n

    @property
    def sensibilidade(self) -> float:
        """VP / (VP + FN), ou recall: dos positivos reais, quantos o modelo encontrou."""
        return self._razao(self.vp, self.vp + self.fn)

    @property
    def especificidade(self) -> float:
        """VN / (VN + FP): dos negativos reais, quantos o modelo deixou de fora."""
        return self._razao(self.vn, self.vn + self.fp)

    @property
    def precisao(self) -> float:
        """VP / (VP + FP): dos que o modelo chamou de positivo, quantos eram mesmo."""
        return self._razao(self.vp, self.vp + self.fp)

    @property
    def f_score(self) -> float:
        """2·P·S / (P + S): média harmônica entre precisão e sensibilidade."""
        p, s = self.precisao, self.sensibilidade
        return self._razao(2 * p * s, p + s) if p + s else 0.0

    def metricas(self) -> dict[str, float]:
        return {
            "acurácia": self.acuracia,
            "sensibilidade": self.sensibilidade,
            "especificidade": self.especificidade,
            "precisão": self.precisao,
            "F-score": self.f_score,
        }

    def relatorio(self) -> str:
        linhas = [
            "                 previsto +   previsto -",
            f"real +   {self.vp:>14} {self.fn:>12}",
            f"real -   {self.fp:>14} {self.vn:>12}",
            "",
        ]
        linhas += [f"{nome:>15}: {valor:.4f}  ({valor:.1%})" for nome, valor in self.metricas().items()]
        return "\n".join(linhas)
