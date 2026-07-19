# AppEvasão — Evidência SBIE 2026

Repositório minimalista de deploy do AppEvasão para o Streamlit Community Cloud.

Aplicação pública: https://appevasaosbie.streamlit.app/

Repositório: https://github.com/EddieFerb/appevasao-sbie-deploy

Main file path:

```text
src/evasao/dashboard/app_evasao.py
```

Comando local:

```bash
streamlit run src/evasao/dashboard/app_evasao.py
```

## Objetivo

Este app apresenta uma evidência pública, pequena e auditável do benchmark temporal reportado no artigo SBIE 2026 sobre evasão no Ensino Superior brasileiro.

A interface foi organizada como narrativa científica:

- História
- Revisão em Números
- Lacunas
- Benchmark Temporal
- Evidências Visuais
- Diagnósticos e Limites
- Artefatos

## Fonte dos dados

Fonte oficial: Microdados do Censo da Educação Superior — INEP/MEC. Os microdados públicos foram tratados e agregados para construção do benchmark temporal apresentado neste artigo.

Link oficial:

```text
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior
```

## Fonte das métricas

As métricas principais são fixas e lidas diretamente de:

```text
modelos/resultados_modelos/comparacao_baselines_temporal.csv
```

Linha principal do artigo: `Random Forest`.

Métricas regressivas fixas:

```text
R²   = 0,3337
MAE  = 0,1778
RMSE = 0,2331
MSE  = 0,0543
```

Métricas classificatórias auxiliares são exibidas apenas quando já estão no artefato comparativo. O app não treina modelo, não ajusta threshold em runtime, não recalcula o benchmark e não altera resultados.

## Nota anti-leakage

A formulação temporal usa variáveis observadas no ano-base `t` para prever `taxa_evasao_alvo` em `t+1`. O modelo principal não usa `taxa_evasao`, `taxa_conclusao` nem colunas derivadas diretamente do alvo como features. A avaliação usa particionamento temporal, não aleatório.

## Escopo público

Este é um repositório minimalista de deploy. O pipeline científico completo não é publicado nesta versão por integrar artefato institucional em desenvolvimento. As métricas, variáveis, critérios de filtragem e divisão temporal estão descritos no artigo.

Este repositório de deploy não inclui:

- pipeline científico interno;
- scripts de processamento;
- scripts de modelagem;
- análises auxiliares;
- modelos salvos;
- dados brutos;
- bases processadas completas.

Ele contém apenas o app Streamlit e os artefatos consolidados necessários para exposição pública das evidências.

## Artefatos incluídos

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
```
