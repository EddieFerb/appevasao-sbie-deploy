# AppEvasao - Benchmark SBIE 2026

Repositorio minimalista de deploy do AppEvasao para o Streamlit Community Cloud.

URL pretendida: https://appevasao.streamlit.app/

Main file path:

```text
src/evasao/dashboard/app_evasao.py
```

Comando local:

```bash
streamlit run src/evasao/dashboard/app_evasao.py
```

## Objetivo

Este app apresenta uma evidencia publica, pequena e auditavel do benchmark temporal reportado no artigo SBIE 2026 sobre evasao no Ensino Superior brasileiro com dados oficiais agregados do INEP/MEC.

Este repositorio de deploy nao inclui o pipeline cientifico interno, scripts de processamento, scripts de modelagem, analises auxiliares, modelos salvos, dados brutos ou bases processadas completas. Ele contem apenas o app Streamlit minimalista e os artefatos consolidados necessarios para exposicao das metricas.

## Fonte das metricas

As metricas sao lidas diretamente de:

```text
modelos/resultados_modelos/comparacao_baselines_temporal.csv
```

Linha principal do artigo: `Random Forest`.

Metricas regressivas fixas:

```text
R2   = 0,3337
MAE  = 0,1778
RMSE = 0,2331
MSE  = 0,0543
```

Metricas classificatorias auxiliares, quando exibidas, sao apenas apoio interpretativo. A comparacao principal do artigo e regressiva.

## Nota anti-leakage

A formulacao temporal usa variaveis observadas no ano-base `t` para prever `taxa_evasao_alvo` em `t+1`. O modelo principal nao usa `taxa_evasao`, `taxa_conclusao` nem colunas derivadas diretamente do alvo como features. A avaliacao usa particionamento temporal, nao aleatorio.

## Escopo publico

Os scripts completos, processamento, modelagem, analises e demais componentes cientificos internos nao integram este repositorio minimalista de deploy.
