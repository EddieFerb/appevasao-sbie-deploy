from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CSV = ROOT / "modelos" / "resultados_modelos" / "comparacao_baselines_temporal.csv"
BENCHMARK_MD = ROOT / "modelos" / "resultados_modelos" / "comparacao_baselines_temporal.md"
METRICAS_TXT = ROOT / "modelos" / "resultados_modelos" / "metricas_modelos.txt"

EXPECTED = {
    "r2": 0.3337047249133319,
    "mae": 0.1778456230862687,
    "rmse": 0.2330517498461937,
    "mse": 0.05431311810637285,
}

st.set_page_config(
    page_title="AppEvasao - Benchmark SBIE 2026",
    page_icon="📊",
    layout="wide",
)


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).lower(): str(col) for col in df.columns}


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Artefato nao encontrado: {BENCHMARK_CSV}")
    return pd.read_csv(BENCHMARK_CSV)


def random_forest_row(df: pd.DataFrame) -> pd.Series:
    mask = df.astype(str).apply(
        lambda row: row.str.contains("Random Forest", case=False, na=False).any(),
        axis=1,
    )
    if not mask.any():
        raise ValueError("Linha Random Forest nao encontrada no CSV do benchmark.")
    return df[mask].iloc[0]


def fmt_decimal(value: object) -> str:
    return f"{float(value):.4f}".replace(".", ",")


def fmt_int(value: object) -> str:
    return f"{int(float(value)):,}".replace(",", ".")


def validate_expected(row: pd.Series, cols: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for key, expected in EXPECTED.items():
        actual = float(row[cols[key]])
        if abs(actual - expected) > 1e-4:
            warnings.append(f"{key}: esperado {expected}, encontrado {actual}")
    return warnings


st.sidebar.title("AppEvasao")
st.sidebar.markdown("**Benchmark SBIE 2026**")
st.sidebar.markdown("URL publica pretendida:")
st.sidebar.code("https://appevasao.streamlit.app/")
st.sidebar.markdown("Main file path:")
st.sidebar.code("src/evasao/dashboard/app_evasao.py")
st.sidebar.markdown("Repositorio:")
st.sidebar.code("EddieFerb/appevasao-sbie-deploy")

st.title("AppEvasao - Benchmark SBIE 2026")
st.subheader("Evidencia publica do benchmark temporal com dados oficiais INEP/MEC")

st.markdown(
    """
Esta aplicacao acompanha o artigo como artefato publico de evidencia. As metricas
desta pagina sao fixas e foram lidas dos artefatos versionados do benchmark
reportado no artigo. O objetivo nao e afirmar superioridade universal do modelo,
mas tornar auditavel a formulacao temporal, os baselines e as metricas.
"""
)

try:
    df = load_benchmark()
    cols = column_lookup(df)
    rf = random_forest_row(df)
    missing = [name for name in ["r2", "mae", "rmse", "mse"] if name not in cols]
    if missing:
        st.error(f"Colunas obrigatorias ausentes no CSV: {', '.join(missing)}")
        st.stop()
except Exception as exc:
    st.error(str(exc))
    st.stop()

warnings = validate_expected(rf, cols)
if warnings:
    st.warning("Divergencia em relacao aos valores auditados: " + "; ".join(warnings))

st.markdown("## Metricas fixas - Random Forest")
metric_cols = st.columns(4)
metric_cols[0].metric("R²", fmt_decimal(rf[cols["r2"]]))
metric_cols[1].metric("MAE", fmt_decimal(rf[cols["mae"]]))
metric_cols[2].metric("RMSE", fmt_decimal(rf[cols["rmse"]]))
metric_cols[3].metric("MSE", fmt_decimal(rf[cols["mse"]]))

aux_keys = ["accuracy", "precision", "recall", "f1"]
if all(key in cols for key in aux_keys):
    st.caption(
        "Metricas classificatorias auxiliares; a comparacao principal do artigo e regressiva."
    )
    aux_cols = st.columns(4)
    aux_cols[0].metric("Accuracy", fmt_decimal(rf[cols["accuracy"]]))
    aux_cols[1].metric("Precision", fmt_decimal(rf[cols["precision"]]))
    aux_cols[2].metric("Recall", fmt_decimal(rf[cols["recall"]]))
    aux_cols[3].metric("F1", fmt_decimal(rf[cols["f1"]]))

st.markdown("## Contexto metodologico")
context_rows = [
    ("Benchmark", "Temporal com dados oficiais INEP/MEC"),
    ("Formulação", "Predicao da taxa de evasao em t+1 a partir de variaveis observadas em t"),
    ("Treino", "ano_base <= 2018"),
    ("Teste", "ano_base > 2018"),
    ("Registros de treino", fmt_int(rf[cols["linhas_treino"]]) if "linhas_treino" in cols else "199.429"),
    ("Registros de teste", fmt_int(rf[cols["linhas_teste"]]) if "linhas_teste" in cols else "111.328"),
    ("Anos-base de treino", "2009-2018"),
    ("Anos-base de teste", "2019, 2021, 2022 e 2023"),
    ("Anos-alvo", "2020, 2022, 2023 e 2024"),
]
st.dataframe(
    pd.DataFrame(context_rows, columns=["Item", "Descricao"]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("## Variaveis e alvo")
var_cols = st.columns(2)
with var_cols[0]:
    st.markdown(
        """
**Variaveis do modelo**
- `ingressantes_t`
- `concluintes_t`
- `matriculados_t`
- `vagas_totais_t`
- `inscritos_totais_t`
"""
    )
with var_cols[1]:
    st.markdown(
        """
**Alvo**
- `taxa_evasao_alvo` em `t+1`

**Nota anti-leakage**
- O modelo principal nao usa `taxa_evasao`, `taxa_conclusao` nem colunas derivadas diretamente do alvo como features.
- A avaliacao usa particionamento temporal, nao aleatorio.
"""
    )

st.markdown("## Tabela completa de baselines")
preferred = [
    "modelo",
    "ano_limite_treino",
    "linhas_treino",
    "linhas_teste",
    "threshold_binario",
    "r2",
    "mae",
    "rmse",
    "mse",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
available = [cols[name] for name in preferred if name in cols]
st.dataframe(df[available], use_container_width=True, hide_index=True)

with st.expander("Artefatos versionados usados nesta pagina"):
    st.write(f"CSV: `{BENCHMARK_CSV.relative_to(ROOT)}`")
    if BENCHMARK_MD.exists():
        st.write(f"Markdown: `{BENCHMARK_MD.relative_to(ROOT)}`")
    if METRICAS_TXT.exists():
        st.write(f"Metricas TXT: `{METRICAS_TXT.relative_to(ROOT)}`")

st.info(
    "Repositorio minimalista de deploy: sem pipeline interno, sem treinamento em runtime "
    "e sem recálculo do benchmark."
)
