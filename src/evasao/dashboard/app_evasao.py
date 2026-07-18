from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CSV = ROOT / "modelos" / "resultados_modelos" / "comparacao_baselines_temporal.csv"
BENCHMARK_MD = ROOT / "modelos" / "resultados_modelos" / "comparacao_baselines_temporal.md"
METRICAS_TXT = ROOT / "modelos" / "resultados_modelos" / "metricas_modelos.txt"
PUBLIC_URL = "https://appevasaosbie.streamlit.app/"

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

st.markdown(
    """
<style>
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
.hero {
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0 0 .35rem 0; font-size: 2.05rem; letter-spacing: 0; }
.hero p { margin: 0; color: #405166; font-size: 1.02rem; }
.metric-card {
    border: 1px solid #d8e1ea;
    border-radius: 8px;
    background: #ffffff;
    padding: 1rem 1.05rem;
    min-height: 124px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .05);
}
.metric-card .label { color: #52677a; font-size: .88rem; font-weight: 700; margin-bottom: .35rem; }
.metric-card .value { color: #13283f; font-size: 2.15rem; font-weight: 800; line-height: 1.05; white-space: nowrap; }
.metric-card .hint { color: #60758a; font-size: .82rem; margin-top: .45rem; }
.flow-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .65rem; margin: .4rem 0 1rem 0; }
.flow-card { border: 1px solid #d8e1ea; border-radius: 8px; padding: .85rem; background: #fbfdff; min-height: 92px; }
.flow-card strong { display: block; color: #13283f; font-size: .92rem; margin-bottom: .25rem; }
.flow-card span { color: #52677a; font-size: .8rem; }
.note-box { border-left: 4px solid #1f77b4; background: #f5f9fd; padding: .85rem 1rem; border-radius: 6px; color: #263b50; }
@media (max-width: 900px) { .flow-row { grid-template-columns: 1fr; } .metric-card .value { font-size: 1.75rem; } }
</style>
""",
    unsafe_allow_html=True,
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


def metric_card(label: str, value: object, hint: str) -> str:
    return """
<div class=\"metric-card\">
  <div class=\"label\">{label}</div>
  <div class=\"value\">{value}</div>
  <div class=\"hint\">{hint}</div>
</div>
""".format(label=label, value=fmt_decimal(value), hint=hint)


def download_artifact(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            width="stretch",
        )
    else:
        st.caption(f"Arquivo indisponivel: {path.relative_to(ROOT)}")


def highlighted_baseline_table(df: pd.DataFrame, cols: dict[str, str]) -> object:
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
    table = df[available].copy()
    numeric_cols = [col for col in table.columns if col.lower() not in {"modelo"}]
    for col in numeric_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    def highlight_rf(row: pd.Series) -> list[str]:
        is_rf = row.astype(str).str.contains("Random Forest", case=False, na=False).any()
        return ["background-color: #eaf4ff; font-weight: 700" if is_rf else "" for _ in row]

    return table.style.apply(highlight_rf, axis=1).format(precision=4, thousands=".", decimal=",")


st.sidebar.title("AppEvasao")
st.sidebar.markdown("**Benchmark SBIE 2026**")
st.sidebar.markdown("URL publica:")
st.sidebar.code(PUBLIC_URL)
st.sidebar.caption("Subdominio alternativo pretendido originalmente: https://appevasao.streamlit.app/")
st.sidebar.markdown("Main file path:")
st.sidebar.code("src/evasao/dashboard/app_evasao.py")
st.sidebar.markdown("Repositorio:")
st.sidebar.code("EddieFerb/appevasao-sbie-deploy")

st.markdown(
    """
<div class="hero">
  <h1>AppEvasao - Benchmark SBIE 2026</h1>
  <p>Evidencia publica do benchmark temporal com dados oficiais INEP/MEC.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="note-box">
Esta aplicacao e um artefato publico de evidencia do artigo. As metricas sao
fixas e lidas dos artefatos versionados. O objetivo e tornar auditavel a
formulacao temporal, os baselines e os resultados reportados, nao substituir o
artigo nem afirmar superioridade universal do modelo.
</div>
""",
    unsafe_allow_html=True,
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
card_cols = st.columns(4)
with card_cols[0]:
    st.markdown(metric_card("R²", rf[cols["r2"]], "maior e melhor"), unsafe_allow_html=True)
with card_cols[1]:
    st.markdown(metric_card("MAE", rf[cols["mae"]], "menor e melhor"), unsafe_allow_html=True)
with card_cols[2]:
    st.markdown(metric_card("RMSE", rf[cols["rmse"]], "menor e melhor"), unsafe_allow_html=True)
with card_cols[3]:
    st.markdown(metric_card("MSE", rf[cols["mse"]], "menor e melhor"), unsafe_allow_html=True)

st.markdown("## Fluxo metodologico do benchmark")
st.markdown(
    """
<div class="flow-row">
  <div class="flow-card"><strong>Dados oficiais INEP/MEC</strong><span>Microdados agregados do Censo da Educacao Superior.</span></div>
  <div class="flow-card"><strong>Variaveis em t</strong><span>Ingressantes, concluintes, matriculados, vagas e inscritos.</span></div>
  <div class="flow-card"><strong>Validacao temporal</strong><span>Treino: ano_base <= 2018. Teste: ano_base > 2018.</span></div>
  <div class="flow-card"><strong>Random Forest</strong><span>Modelo regressivo comparado com baselines temporais.</span></div>
  <div class="flow-card"><strong>Metricas fixas</strong><span>Valores reportados no artigo e lidos do CSV versionado.</span></div>
</div>
""",
    unsafe_allow_html=True,
)

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
    ("Alvo", "taxa_evasao_alvo em t+1"),
]
st.dataframe(pd.DataFrame(context_rows, columns=["Item", "Descricao"]), width="stretch", hide_index=True)

st.markdown("## Visualizacoes do benchmark")
plot_df = df.copy()
for name in ["r2", "mae", "rmse", "mse", "accuracy", "precision", "recall", "f1"]:
    if name in cols:
        plot_df[cols[name]] = pd.to_numeric(plot_df[cols[name]], errors="coerce")

fig_r2 = px.bar(
    plot_df,
    x=cols["modelo"],
    y=cols["r2"],
    text=plot_df[cols["r2"]].map(lambda value: f"{value:.4f}"),
    title="R² por modelo",
    labels={cols["modelo"]: "Modelo", cols["r2"]: "R²"},
)
fig_r2.add_annotation(
    x="Random Forest",
    y=float(rf[cols["r2"]]),
    text="Random Forest: melhor R²",
    showarrow=True,
    arrowhead=2,
    yshift=28,
)
fig_r2.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_r2, width="stretch")

error_cols = [cols[name] for name in ["mae", "rmse", "mse"] if name in cols]
errors_long = plot_df[[cols["modelo"], *error_cols]].melt(
    id_vars=cols["modelo"], var_name="Metrica", value_name="Valor"
)
fig_errors = px.bar(
    errors_long,
    x=cols["modelo"],
    y="Valor",
    color="Metrica",
    barmode="group",
    title="Erros regressivos por modelo",
    labels={cols["modelo"]: "Modelo", "Valor": "Erro"},
)
fig_errors.add_annotation(
    x="Random Forest",
    y=float(rf[cols["rmse"]]),
    text="Menores erros no benchmark",
    showarrow=True,
    arrowhead=2,
    yshift=36,
)
fig_errors.update_layout(margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_errors, width="stretch")

aux_keys = ["accuracy", "precision", "recall", "f1"]
if all(key in cols for key in aux_keys):
    aux_df = pd.DataFrame(
        {
            "Metrica": ["Accuracy", "Precision", "Recall", "F1"],
            "Valor": [float(rf[cols[key]]) for key in aux_keys],
        }
    )
    fig_aux = px.bar(
        aux_df,
        x="Metrica",
        y="Valor",
        text=aux_df["Valor"].map(lambda value: f"{value:.4f}"),
        title="Metricas classificatorias auxiliares - Random Forest",
        labels={"Valor": "Valor"},
    )
    fig_aux.update_yaxes(range=[0, 1])
    fig_aux.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_aux, width="stretch")
    st.caption("Metricas auxiliares; a comparacao principal do artigo e regressiva.")

st.markdown("## Tabela completa de baselines")
st.dataframe(highlighted_baseline_table(df, cols), width="stretch", hide_index=True)

st.markdown("## Variaveis, alvo e anti-leakage")
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

st.markdown("## Artefatos versionados")
download_cols = st.columns(3)
with download_cols[0]:
    download_artifact(BENCHMARK_CSV, "Download CSV", "text/csv")
with download_cols[1]:
    download_artifact(BENCHMARK_MD, "Download Markdown", "text/markdown")
with download_cols[2]:
    download_artifact(METRICAS_TXT, "Download TXT", "text/plain")

with st.expander("Caminhos dos artefatos usados nesta pagina"):
    st.write(f"CSV: `{BENCHMARK_CSV.relative_to(ROOT)}`")
    if BENCHMARK_MD.exists():
        st.write(f"Markdown: `{BENCHMARK_MD.relative_to(ROOT)}`")
    if METRICAS_TXT.exists():
        st.write(f"Metricas TXT: `{METRICAS_TXT.relative_to(ROOT)}`")

st.info(
    "Repositorio minimalista de deploy: sem pipeline interno, sem treinamento em runtime "
    "e sem recalculo do benchmark."
)
