# AppEvasao - Evidencia SBIE 2026

Repositorio minimalista de deploy do AppEvasao para o Streamlit Community Cloud.

URL publica: https://appevasaosbie.streamlit.app/

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

A interface foi organizada como narrativa cientifica:

- Historia
- Mapa da Revisao
- Lacuna Metodologica
- Benchmark Temporal
- Laboratorio de Threshold
- Diagnosticos
- Artefatos

## Fonte das metricas

As metricas principais sao fixas e lidas diretamente de:

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

Metricas classificatorias auxiliares e simulacoes de threshold usam somente artefatos versionados de holdout. O app nao treina modelo, nao recalcula o benchmark regressivo e nao altera resultados.

## Nota anti-leakage

A formulacao temporal usa variaveis observadas no ano-base `t` para prever `taxa_evasao_alvo` em `t+1`. O modelo principal nao usa `taxa_evasao`, `taxa_conclusao` nem colunas derivadas diretamente do alvo como features. A avaliacao usa particionamento temporal, nao aleatorio.

## Escopo publico

Este repositorio de deploy nao inclui:

- pipeline cientifico interno;
- scripts de processamento;
- scripts de modelagem;
- analises auxiliares;
- modelos salvos;
- dados brutos;
- bases processadas completas.

Ele contem apenas o app Streamlit e os artefatos consolidados necessarios para exposicao publica das evidencias.

## Artefatos incluidos

```text
modelos/resultados_modelos/comparacao_baselines_temporal.csv
modelos/resultados_modelos/comparacao_baselines_temporal.md
modelos/resultados_modelos/metricas_modelos.txt
modelos/resultados_modelos/feature_importance_random_forest.csv
modelos/resultados_modelos/feature_importance_random_forest.md
modelos/resultados_modelos/diagnostico_completude_temporal.csv
modelos/resultados_modelos/diagnostico_completude_temporal.md
modelos/resultados_modelos/diagnostico_balanceamento_classes.csv
modelos/resultados_modelos/diagnostico_balanceamento_classes.md
modelos/resultados_modelos/predicoes_holdout_random_forest_sbie.csv
```
