import pytest

from metricas import MatrizConfusao
from metricas.__main__ import main

EXEMPLO = MatrizConfusao(vp=40, vn=45, fp=5, fn=10)


def test_metricas_da_matriz_de_exemplo():
    assert EXEMPLO.n == 100
    assert EXEMPLO.acuracia == pytest.approx(0.85)
    assert EXEMPLO.sensibilidade == pytest.approx(40 / 50)
    assert EXEMPLO.especificidade == pytest.approx(45 / 50)
    assert EXEMPLO.precisao == pytest.approx(40 / 45)
    assert EXEMPLO.f_score == pytest.approx(2 * (40 / 45) * 0.8 / ((40 / 45) + 0.8))


def test_f_score_e_media_harmonica():
    m = MatrizConfusao(vp=1, vn=0, fp=0, fn=9)  # precisão 1, sensibilidade 0,1
    assert m.f_score == pytest.approx(2 * 1 * 0.1 / 1.1)
    assert m.f_score < (m.precisao + m.sensibilidade) / 2


def test_denominador_zero_vira_zero():
    nunca_positivo = MatrizConfusao(vp=0, vn=10, fp=0, fn=5)
    assert nunca_positivo.precisao == 0.0
    assert nunca_positivo.f_score == 0.0
    assert nunca_positivo.especificidade == 1.0


@pytest.mark.parametrize("valores", [(-1, 0, 0, 0), (1.5, 0, 0, 0), (True, 1, 1, 1), (0, 0, 0, 0)])
def test_matriz_invalida(valores):
    with pytest.raises(ValueError):
        MatrizConfusao(*valores)


def test_matriz_a_partir_de_rotulos():
    reais = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    previstos = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    assert MatrizConfusao.de_rotulos(reais, previstos) == MatrizConfusao(vp=3, vn=4, fp=1, fn=2)


def test_rotulos_de_tamanhos_diferentes():
    with pytest.raises(ValueError):
        MatrizConfusao.de_rotulos([1, 0], [1])


def test_confere_com_o_scikit_learn():
    sklearn = pytest.importorskip("sklearn.metrics")
    import random

    gerador = random.Random(7)
    reais = [gerador.randint(0, 1) for _ in range(500)]
    previstos = [r if gerador.random() < 0.8 else 1 - r for r in reais]
    m = MatrizConfusao.de_rotulos(reais, previstos)
    assert m.acuracia == pytest.approx(sklearn.accuracy_score(reais, previstos))
    assert m.precisao == pytest.approx(sklearn.precision_score(reais, previstos))
    assert m.sensibilidade == pytest.approx(sklearn.recall_score(reais, previstos))
    assert m.especificidade == pytest.approx(sklearn.recall_score(reais, previstos, pos_label=0))
    assert m.f_score == pytest.approx(sklearn.f1_score(reais, previstos))


def test_linha_de_comando(capsys):
    assert main(["40", "45", "5", "10"]) == 0
    saida = capsys.readouterr().out
    assert "acurácia: 0.8500" in saida
    assert main(["1", "2"]) == 2
    assert main(["-1", "0", "0", "0"]) == 2
