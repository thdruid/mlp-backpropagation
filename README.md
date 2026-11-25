# MLP com backpropagation

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thdruid/mlp-backpropagation/blob/main/mlp_backpropagation_banknote.ipynb)

Rede neural MLP com forward, entropia cruzada binária (BCE), backpropagation e SGD em mini-lotes em NumPy, aplicada ao dataset Banknote Authentication (UCI). O scikit-learn é usado só para dividir os dados, padronizar e calcular métricas; ele não treina a rede.

Trabalho da disciplina de Redes Neurais Artificiais (CIn-UFPE).

## O que o notebook faz

- Implementa a rede (4 entradas, camadas ocultas ReLU/tanh, saída sigmoid) com as fórmulas de forward e backward explicitadas.
- Verifica os gradientes analíticos contra diferenças finitas: erro relativo da ordem de 1e-11 para ReLU e tanh.
- Mostra um passo de treino numa rede pequena, com deltas e gradientes impressos, para ligar a matemática ao código.
- Compara 10 configurações x 5 sementes: taxa de aprendizagem, largura, profundidade, ativação, padronização, inicialização e tamanho do lote.
- Escolhe a configuração só pela validação e abre o teste uma única vez, no final.

## Principais resultados (semente 42 e médias de 5 sementes)

- Taxa 0,001 aprende mais devagar em 400 épocas: acurácia média de validação 96,4%, contra 99,3% da baseline (taxa 0,01) e 100% da taxa 0,1.
- Largura 2 e largura 32 igualam a acurácia média da baseline, mas com BCE diferente: a acurácia satura, então a BCE é mais informativa.
- Teste: baseline com 98,52% (4 erros) e configuração selecionada (taxa 0,1) com 99,63% (1 erro), ambas com recall de 100% na classe 1.

## Protocolo

- Duplicatas exatas removidas antes da divisão (1.372 para 1.348 amostras).
- Divisão estratificada 60/20/20, semente 42.
- Padronização ajustada somente no treino.
- Sementes 1, 7, 21, 42 e 99 para medir a variação de inicialização.

## Como rodar

No Colab, use o botão acima. Localmente:

```bash
git clone https://github.com/thdruid/mlp-backpropagation.git
cd mlp-backpropagation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

O notebook baixa o dataset da UCI automaticamente (`MODO_DADOS = "uci"`). A execução completa leva cerca de 1 minuto em CPU.

## Limitações

- As 5 sementes variam só a inicialização e a ordem dos lotes, não a partição dos dados.
- A mesma partição foi usada na seleção e no teste, então o teste não é um benchmark externo.
- Os atributos já vêm extraídos das imagens: o experimento é educacional e não valida um sistema real de autenticação de cédulas.

## Dados e referências

Lohweg, V. (2012). Banknote Authentication. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57 (CC BY 4.0).
