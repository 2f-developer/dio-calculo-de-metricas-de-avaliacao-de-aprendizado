# Métricas de avaliação de classificadores

Desafio "Cálculo de Métricas de Avaliação de Aprendizado", da formação Machine Learning Specialist da DIO. A partir de uma matriz de confusão escolhida livremente, calcular as cinco métricas da Tabela 1 do enunciado:

| Métrica | Fórmula | Pergunta que responde |
| --- | --- | --- |
| Acurácia | (VP + VN) / N | quanto o modelo acerta no total? |
| Sensibilidade (recall) | VP / (VP + FN) | dos positivos reais, quantos ele encontrou? |
| Especificidade | VN / (VN + FP) | dos negativos reais, quantos ele deixou de fora? |
| Precisão | VP / (VP + FP) | quando ele diz "positivo", quantas vezes está certo? |
| F-score | 2·P·S / (P + S) | equilíbrio entre precisão e sensibilidade |

## A matriz escolhida

Imagine um teste que avaliou 100 pessoas, 50 com a condição e 50 sem:

|  | previsto positivo | previsto negativo |
| --- | --- | --- |
| **realmente positivo** | VP = 40 | FN = 10 |
| **realmente negativo** | FP = 5 | VN = 45 |

```
$ python -m metricas 40 45 5 10
       acurácia: 0.8500  (85.0%)
  sensibilidade: 0.8000  (80.0%)
 especificidade: 0.9000  (90.0%)
       precisão: 0.8889  (88.9%)
        F-score: 0.8421  (84.2%)
```

A acurácia de 85% parece boa, mas a sensibilidade mostra que 1 em cada 5 positivos passou despercebido. É por isso que se olha mais de uma métrica: num exame médico, esses 10 falsos negativos podem pesar muito mais do que os 5 falsos positivos.

## Como usar

Não tem dependências além do Python 3.10+.

```bash
python -m metricas VP VN FP FN     # sem argumentos, usa a matriz acima
```

Como biblioteca:

```python
from metricas import MatrizConfusao

m = MatrizConfusao(vp=40, vn=45, fp=5, fn=10)
m.f_score  # 0.8421...
m.metricas()  # dicionário com as cinco

# ou a partir das saídas de um modelo
MatrizConfusao.de_rotulos(reais=[1, 0, 1, 1], previstos=[1, 0, 0, 1])
```

A matriz recusa valores negativos, não inteiros ou uma matriz toda zerada. Quando uma métrica teria divisão por zero (precisão de um modelo que nunca prevê positivo, por exemplo), o resultado é 0, a mesma convenção do scikit-learn.

## Testes

```bash
pip install -e '.[dev]'
pytest
```

Além dos casos calculados à mão e das bordas (divisão por zero, entradas inválidas), um teste gera 500 previsões aleatórias e confere as cinco métricas com as do **scikit-learn**, para garantir que as fórmulas batem com a implementação de referência. O GitHub Actions roda lint e testes a cada push.
