# Previsão de Consumo de Energia Residencial

Projeto de Aprendizado de Máquina para prever o consumo de energia elétrica residencial a partir do dataset **Individual Household Electric Power Consumption** (UCI Machine Learning Repository). O objetivo é comparar **Regressão Linear** e **Random Forest Regressor** em um protocolo experimental reprodutível, com geração de métricas, gráficos e tabelas para análise acadêmica.

## Objetivo

Prever a variável `Global_active_power` (potência ativa global consumida pela residência, em kW) utilizando variáveis elétricas do medidor e atributos temporais derivados da data e hora da medição.

## Estrutura do projeto

```
projeto-individual/
├── data/                           # Dataset (não versionado)
├── docs/                           # Especificação e documentação
├── plots/                          # Gráficos gerados
├── results/                        # CSVs, metadados e predições
├── src/
│   ├── data_loader.py              # Carregamento do dataset
│   ├── preprocessing.py            # Limpeza e conversão temporal
│   ├── feature_engineering.py      # Monta as matrizes X (features) e y (alvo).
│   ├── linear_regression_model.py  # Treina modelo de regressão linear
│   ├── random_forest_model.py      # Treina modelo de random forest
│   ├── evaluation.py               # Métricas e estatísticas
│   ├── visualization.py            # Gráficos
│   ├── experiment_runner.py        # Orquestração do experimento
│   └── validation.py               # Validação dos artefatos gerados
├── main.py                         # Ponto de entrada
└── requirements.txt
```

## Como funciona

O pipeline executa as seguintes etapas:

1. **Carregamento** — lê o arquivo `.txt` com separador `;`, trata `?` como valor ausente e converte colunas numéricas.
2. **Pré-processamento** — remove linhas com dados faltantes, une `Date` e `Time` em `Datetime` e ordena cronologicamente.
3. **Engenharia de atributos** — cria `hour`, `day`, `month` e `weekday` a partir de `Datetime`.
4. **Gráficos exploratórios** — histograma da variável alvo, heatmap de correlação e série temporal.
5. **Experimento (30 execuções)** — para cada seed de 1 a 30:
  - divide os dados em 80% treino / 20% teste;
  - treina Regressão Linear (com `StandardScaler`) e Random Forest;
  - calcula MAE, RMSE, R² e tempos de execução.
6. **Agregação e persistência** — salva métricas individuais e estatísticas (média, desvio padrão, mediana) em CSV e JSON.
7. **Gráficos de resultados** — scatter Real vs Predito (ambos os modelos) e boxplots das métricas.
8. **Validação** — verifica se todos os arquivos esperados foram gerados corretamente.

## Requisitos

- Python 3.10 ou superior (recomendado)
- Dependências listadas em `requirements.txt`

## Instalação

```bash
pip install -r requirements.txt
```

## Dataset

Coloque o arquivo `household_power_consumption.txt` na pasta `data/`. O dataset pode ser obtido em [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

## Execução

Na raiz do projeto:

```bash
python main.py
```

O experimento completo pode levar bastante tempo devido ao volume de dados (~2 milhões de amostras) e às 30 execuções do Random Forest.

## Saídas geradas

### Gráficos (`plots/`)


| Arquivo                      | Descrição                                 |
| ---------------------------- | ----------------------------------------- |
| `target_distribution.png`    | Distribuição de `Global_active_power`     |
| `correlation_heatmap.png`    | Matriz de correlação                      |
| `consumption_timeseries.png` | Consumo ao longo do tempo (média horária) |
| `real_vs_predicted.png`      | Valores reais vs preditos (seed=1)        |
| `metrics_boxplot.png`        | Boxplots de MAE, RMSE e R²                |


### Resultados (`results/`)


| Arquivo                       | Descrição                                        |
| ----------------------------- | ------------------------------------------------ |
| `all_runs.csv`                | Métricas das 60 execuções (30 seeds × 2 modelos) |
| `linear_regression_runs.csv`  | Execuções da Regressão Linear                    |
| `random_forest_runs.csv`      | Execuções do Random Forest                       |
| `summary.csv`                 | Média, desvio padrão e mediana por modelo        |
| `experiment_metadata.json`    | Metadados do experimento                         |
| `plot_predictions_seed_1.npz` | Predições usadas no gráfico Real vs Predito      |


## Documentação adicional

- `[docs/spec.md](docs/spec.md)` — especificação completa do projeto
- `[docs/implementacao.txt](docs/implementacao.txt)` — detalhes da implementação e metodologia experimental (base para relatório)

