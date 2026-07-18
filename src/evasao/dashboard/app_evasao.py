from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "modelos" / "resultados_modelos"

BENCHMARK_CSV = RESULTS_DIR / "comparacao_baselines_temporal.csv"
BENCHMARK_MD = RESULTS_DIR / "comparacao_baselines_temporal.md"
METRICAS_TXT = RESULTS_DIR / "metricas_modelos.txt"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_random_forest.csv"
FEATURE_IMPORTANCE_MD = RESULTS_DIR / "feature_importance_random_forest.md"
COMPLETUDE_CSV = RESULTS_DIR / "diagnostico_completude_temporal.csv"
COMPLETUDE_MD = RESULTS_DIR / "diagnostico_completude_temporal.md"
BALANCEAMENTO_CSV = RESULTS_DIR / "diagnostico_balanceamento_classes.csv"
BALANCEAMENTO_MD = RESULTS_DIR / "diagnostico_balanceamento_classes.md"
PREDICOES_CSV = RESULTS_DIR / "predicoes_holdout_random_forest_sbie.csv"

PUBLIC_URL = "https://appevasaosbie.streamlit.app/"

EXPECTED = {
    "r2": 0.3337047249133319,
    "mae": 0.1778456230862687,
    "rmse": 0.2330517498461937,
    "mse": 0.05431311810637285,
}

REVIEW_COUNTS = {
    "studies": 88,
    "without_inep": 78,
    "with_inep": 10,
    "random_forest": 49,
    "artifacts": 14,
    "inep_rf": 7,
    "temporal_validation": 7,
    "without_artifacts": 74,
    "imported": 3827,
    "triage": 2350,
    "fts": 119,
}

st.set_page_config(
    page_title="AppEvasao - Evidencia SBIE 2026",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
<style>
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
.hero {
    border: 1px solid #d6dde6;
    border-radius: 8px;
    padding: 1.25rem 1.45rem;
    background: #f8fafc;
    margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; letter-spacing: 0; color: #122033; }
.hero p { margin: 0; color: #425466; font-size: 1.02rem; }
.metric-card {
    border: 1px solid #d6dde6;
    border-radius: 8px;
    background: #ffffff;
    padding: .95rem 1rem;
    min-height: 122px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .05);
}
.metric-card .label { color: #4e6478; font-size: .86rem; font-weight: 700; margin-bottom: .32rem; }
.metric-card .value { color: #14243a; font-size: 2.05rem; font-weight: 800; line-height: 1.05; white-space: nowrap; }
.metric-card .hint { color: #5f7185; font-size: .8rem; margin-top: .42rem; }
.story-box {
    border-left: 4px solid #1f77b4;
    background: #f5f9fd;
    padding: .85rem 1rem;
    border-radius: 6px;
    color: #263b50;
    margin-bottom: 1rem;
}
.step-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .65rem; margin: .4rem 0 1rem 0; }
.step-card { border: 1px solid #d8e1ea; border-radius: 8px; padding: .8rem; background: #fbfdff; min-height: 98px; }
.step-card strong { display: block; color: #13283f; font-size: .9rem; margin-bottom: .25rem; }
.step-card span { color: #52677a; font-size: .79rem; }
.small-muted { color: #60758a; font-size: .9rem; }
@media (max-width: 900px) {
  .step-grid { grid-template-columns: 1fr; }
  .metric-card .value { font-size: 1.65rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).lower(): str(col) for col in df.columns}


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    df = read_csv(BENCHMARK_CSV)
    if df is None:
        raise FileNotFoundError(f"Artefato nao encontrado: {BENCHMARK_CSV}")
    return df


def random_forest_row(df: pd.DataFrame) -> pd.Series:
    mask = df.astype(str).apply(
        lambda row: row.str.contains("Random Forest", case=False, na=False).any(),
        axis=1,
    )
    if not mask.any():
        raise ValueError("Linha Random Forest nao encontrada no CSV do benchmark.")
    return df[mask].iloc[0]


def fmt_decimal(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_int(value: object) -> str:
    return f"{int(float(value)):,}".replace(",", ".")


def metric_card(label: str, value: object, hint: str, digits: int = 4) -> str:
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{fmt_decimal(value, digits)}</div>
  <div class="hint">{hint}</div>
</div>
"""


def count_card(label: str, value: object, hint: str) -> str:
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{fmt_int(value)}</div>
  <div class="hint">{hint}</div>
</div>
"""


def validate_expected(row: pd.Series, cols: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for key, expected in EXPECTED.items():
        actual = float(row[cols[key]])
        if abs(actual - expected) > 1e-4:
            warnings.append(f"{key}: esperado {expected}, encontrado {actual}")
    return warnings


def baseline_table(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
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
    for col in table.columns:
        if col.lower() != "modelo":
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def highlighted_baseline_table(table: pd.DataFrame) -> object:
    def highlight_rf(row: pd.Series) -> list[str]:
        is_rf = row.astype(str).str.contains("Random Forest", case=False, na=False).any()
        return ["background-color: #eaf4ff; font-weight: 700" if is_rf else "" for _ in row]

    return table.style.apply(highlight_rf, axis=1).format(precision=4, thousands=".", decimal=",")


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
        st.caption(f"Indisponivel: {path.relative_to(ROOT)}")


def method_flow_markup() -> str:
    return """
<div class="step-grid">
  <div class="step-card"><strong>INEP/MEC</strong><span>Dados oficiais agregados do Censo da Educacao Superior.</span></div>
  <div class="step-card"><strong>Features em t</strong><span>Ingressantes, concluintes, matriculados, vagas e inscritos.</span></div>
  <div class="step-card"><strong>Alvo em t+1</strong><span>taxa_evasao_alvo deslocada temporalmente.</span></div>
  <div class="step-card"><strong>Holdout temporal</strong><span>Treino ate 2018. Teste apos 2018.</span></div>
  <div class="step-card"><strong>Benchmark fixo</strong><span>Random Forest comparada a baselines versionados.</span></div>
</div>
"""


def threshold_metrics(pred: pd.DataFrame, threshold: float) -> dict[str, float]:
    real = (pred["y_real"].astype(float) >= threshold).astype(int)
    pred_cls = (pred["y_pred"].astype(float).clip(0, 1) >= threshold).astype(int)
    tp = int(((real == 1) & (pred_cls == 1)).sum())
    tn = int(((real == 0) & (pred_cls == 0)).sum())
    fp = int(((real == 0) & (pred_cls == 1)).sum())
    fn = int(((real == 1) & (pred_cls == 0)).sum())
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positivos_reais": int(real.sum()),
        "positivos_preditos": int(pred_cls.sum()),
    }


def render_confusion_matrix(values: dict[str, float], title: str) -> None:
    z = [[values["tn"], values["fp"]], [values["fn"], values["tp"]]]
    text = [[fmt_int(values["tn"]), fmt_int(values["fp"])], [fmt_int(values["fn"]), fmt_int(values["tp"])]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predito: baixa", "Predito: alta"],
            y=["Real: baixa", "Real: alta"],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=60, b=20), height=410)
    st.plotly_chart(fig, width="stretch")


st.sidebar.title("AppEvasao")
st.sidebar.markdown("**Evidencia SBIE 2026**")
st.sidebar.markdown("URL publica:")
st.sidebar.code(PUBLIC_URL)
st.sidebar.markdown("Repositorio:")
st.sidebar.code("EddieFerb/appevasao-sbie-deploy")
st.sidebar.markdown("Main file path:")
st.sidebar.code("src/evasao/dashboard/app_evasao.py")

st.markdown(
    """
<div class="hero">
  <h1>AppEvasao - Evidencia SBIE 2026</h1>
  <p>Uma narrativa publica, auditavel e anti-leakage do benchmark temporal de evasao no Ensino Superior brasileiro.</p>
</div>
""",
    unsafe_allow_html=True,
)

try:
    df = load_benchmark()
    cols = column_lookup(df)
    rf = random_forest_row(df)
    missing = [name for name in ["modelo", "r2", "mae", "rmse", "mse"] if name not in cols]
    if missing:
        st.error(f"Colunas obrigatorias ausentes no CSV: {', '.join(missing)}")
        st.stop()
except Exception as exc:
    st.error(str(exc))
    st.stop()

warnings = validate_expected(rf, cols)
if warnings:
    st.warning("Divergencia em relacao aos valores auditados: " + "; ".join(warnings))

table = baseline_table(df, cols)
feature_importance = read_csv(FEATURE_IMPORTANCE_CSV)
completude = read_csv(COMPLETUDE_CSV)
balanceamento = read_csv(BALANCEAMENTO_CSV)
predicoes = read_csv(PREDICOES_CSV)

tabs = st.tabs(
    [
        "História",
        "Mapa da Revisão",
        "Lacuna Metodológica",
        "Benchmark Temporal",
        "Laboratório de Threshold",
        "Diagnósticos",
        "Artefatos",
    ]
)

with tabs[0]:
    st.markdown(
        """
<div class="story-box">
A evasao no Ensino Superior brasileiro nao e apenas uma taxa administrativa:
ela concentra perdas de trajetoria estudantil, capacidade institucional e
politica publica. O AppEvasao apresenta a resposta experimental do artigo:
um benchmark temporal pequeno, rastreavel e comparavel, construido sobre dados
oficiais e sem treinamento em tempo de execucao.
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(count_card("Estudos incluidos", REVIEW_COUNTS["studies"], "base da revisao de escopo"), unsafe_allow_html=True)
    with c2:
        st.markdown(count_card("Estudos com INEP explicito", REVIEW_COUNTS["with_inep"], "uso declarado de dados oficiais"), unsafe_allow_html=True)
    with c3:
        st.markdown(count_card("Validacao temporal", REVIEW_COUNTS["temporal_validation"], "aprox. 8,0% da amostra"), unsafe_allow_html=True)

    st.markdown("### A tese operacional")
    st.markdown(method_flow_markup(), unsafe_allow_html=True)
    st.write(
        "O app fixa a pergunta empirica: usando variaveis observadas no ano-base `t`, "
        "quao bem um modelo consegue estimar `taxa_evasao_alvo` no ano `t+1`?"
    )

with tabs[1]:
    st.markdown("### Mapa da revisão de escopo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(count_card("Registros importados", REVIEW_COUNTS["imported"], "entrada inicial"), unsafe_allow_html=True)
    with c2:
        st.markdown(count_card("Triagem", REVIEW_COUNTS["triage"], "registros avaliados"), unsafe_allow_html=True)
    with c3:
        st.markdown(count_card("Texto completo", REVIEW_COUNTS["fts"], "FTS avaliados"), unsafe_allow_html=True)
    with c4:
        st.markdown(count_card("Incluidos", REVIEW_COUNTS["studies"], "corpus final"), unsafe_allow_html=True)

    review_df = pd.DataFrame(
        [
            ("Sem INEP/MEC explicito", REVIEW_COUNTS["without_inep"]),
            ("Com INEP/MEC explicito", REVIEW_COUNTS["with_inep"]),
            ("Usam Random Forest", REVIEW_COUNTS["random_forest"]),
            ("Trazem artefato", REVIEW_COUNTS["artifacts"]),
            ("INEP/MEC + Random Forest", REVIEW_COUNTS["inep_rf"]),
            ("Validacao temporal", REVIEW_COUNTS["temporal_validation"]),
        ],
        columns=["Categoria", "Estudos"],
    )
    fig = px.bar(review_df, x="Categoria", y="Estudos", text="Estudos", title="Sinais metodologicos encontrados na literatura")
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch")

    artifact_df = pd.DataFrame(
        [
            ("Com artefato", REVIEW_COUNTS["artifacts"]),
            ("Sem artefato", REVIEW_COUNTS["without_artifacts"]),
        ],
        columns=["Grupo", "Estudos"],
    )
    st.plotly_chart(
        px.pie(artifact_df, names="Grupo", values="Estudos", title="Disponibilidade de artefatos na amostra"),
        width="stretch",
    )

with tabs[2]:
    st.markdown("### Lacuna metodológica")
    st.markdown(
        """
<div class="story-box">
A revisao indica um campo produtivo, mas fragmentado: muitos trabalhos usam
classificacao e modelos populares, poucos deixam artefatos auditaveis e poucos
validam o desempenho respeitando a ordem temporal. A contribuicao do app e
exibir um benchmark reproduzivel como evidencia publica, nao uma promessa de
predicao perfeita.
</div>
""",
        unsafe_allow_html=True,
    )
    gap_df = pd.DataFrame(
        [
            ("Random Forest na literatura", REVIEW_COUNTS["random_forest"]),
            ("Artefatos disponiveis", REVIEW_COUNTS["artifacts"]),
            ("Validacao temporal", REVIEW_COUNTS["temporal_validation"]),
            ("Metricas MAE", 4),
            ("Metricas MSE", 3),
            ("Metricas RMSE", 1),
            ("Metricas R2/Pseudo-R2", 1),
        ],
        columns=["Evidencia", "Quantidade"],
    )
    fig = px.bar(gap_df, y="Evidencia", x="Quantidade", orientation="h", text="Quantidade", title="O que aparece pouco no corpus")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.markdown("### Benchmark temporal fixo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("R²", rf[cols["r2"]], "maior e melhor"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("MAE", rf[cols["mae"]], "menor e melhor"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("RMSE", rf[cols["rmse"]], "menor e melhor"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("MSE", rf[cols["mse"]], "menor e melhor"), unsafe_allow_html=True)

    plot_df = df.copy()
    for name in ["r2", "mae", "rmse", "mse", "accuracy", "precision", "recall", "f1"]:
        if name in cols:
            plot_df[cols[name]] = pd.to_numeric(plot_df[cols[name]], errors="coerce")

    fig_r2 = px.bar(
        plot_df,
        x=cols["modelo"],
        y=cols["r2"],
        text=plot_df[cols["r2"]].map(lambda value: f"{value:.4f}"),
        title="R² por modelo no holdout temporal",
        labels={cols["modelo"]: "Modelo", cols["r2"]: "R²"},
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
    fig_errors.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_errors, width="stretch")
    st.dataframe(highlighted_baseline_table(table), width="stretch", hide_index=True)

with tabs[4]:
    st.markdown("### Laboratório de Threshold")
    st.caption("Exploracao sobre o holdout real versionado. Nao recalcula regressao nem altera metricas do artigo.")
    if predicoes is None:
        st.info("Arquivo de predicoes do holdout nao encontrado nesta versao.")
    else:
        threshold = st.slider("Threshold binario", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
        values = threshold_metrics(predicoes, threshold)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Accuracy", values["accuracy"], "classificacao auxiliar"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Precision", values["precision"], "classe alta evasao"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Recall", values["recall"], "classe alta evasao"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("F1", values["f1"], "equilibrio precision/recall"), unsafe_allow_html=True)
        render_confusion_matrix(values, f"Matriz de confusao no threshold {fmt_decimal(threshold, 2)}")
        scatter_sample = predicoes.sample(min(len(predicoes), 5000), random_state=42)
        fig_scatter = px.scatter(
            scatter_sample,
            x="y_real",
            y="y_pred",
            opacity=0.35,
            title="Amostra do holdout: valor real vs predicao",
            labels={"y_real": "Taxa real", "y_pred": "Taxa predita"},
        )
        fig_scatter.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="#d62728", dash="dash"))
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_scatter, width="stretch")

with tabs[5]:
    st.markdown("### Diagnósticos")
    if feature_importance is not None and {"feature", "importance"}.issubset(feature_importance.columns):
        fi = feature_importance.sort_values("importance", ascending=True)
        st.plotly_chart(
            px.bar(
                fi,
                x="importance",
                y="feature",
                orientation="h",
                title="Importancia relativa dos atributos - Random Forest",
                labels={"importance": "Importancia", "feature": "Feature"},
            ),
            width="stretch",
        )
    if completude is not None:
        comp = completude.copy()
        comp["percentual_perda"] = pd.to_numeric(comp["percentual_perda"], errors="coerce")
        st.plotly_chart(
            px.line(
                comp,
                x="ano_base",
                y="percentual_perda",
                markers=True,
                title="Perda apos dropna por ano-base",
                labels={"ano_base": "Ano-base", "percentual_perda": "Percentual de perda"},
            ),
            width="stretch",
        )
        st.dataframe(comp, width="stretch", hide_index=True)
    if balanceamento is not None:
        bal = balanceamento.copy()
        st.plotly_chart(
            px.bar(
                bal,
                x="conjunto",
                y=["prop_baixa_evasao", "prop_alta_evasao"],
                barmode="group",
                title="Balanceamento das classes por conjunto",
                labels={"value": "Proporcao", "conjunto": "Conjunto", "variable": "Classe"},
            ),
            width="stretch",
        )
        st.dataframe(bal, width="stretch", hide_index=True)

with tabs[6]:
    st.markdown("### Artefatos versionados")
    st.write("Repositorio: `EddieFerb/appevasao-sbie-deploy`")
    st.write(f"URL publica: `{PUBLIC_URL}`")
    st.markdown(
        "Este repo publico contem apenas o app e artefatos consolidados. "
        "Nao contem pipeline de modelagem, processamento, analises internas, dados brutos ou modelos salvos."
    )

    artifact_specs = [
        (BENCHMARK_CSV, "Benchmark CSV", "text/csv"),
        (BENCHMARK_MD, "Benchmark MD", "text/markdown"),
        (METRICAS_TXT, "Metricas TXT", "text/plain"),
        (FEATURE_IMPORTANCE_CSV, "Importancia CSV", "text/csv"),
        (FEATURE_IMPORTANCE_MD, "Importancia MD", "text/markdown"),
        (COMPLETUDE_CSV, "Completude CSV", "text/csv"),
        (COMPLETUDE_MD, "Completude MD", "text/markdown"),
        (BALANCEAMENTO_CSV, "Balanceamento CSV", "text/csv"),
        (BALANCEAMENTO_MD, "Balanceamento MD", "text/markdown"),
        (PREDICOES_CSV, "Predicoes holdout CSV", "text/csv"),
    ]
    for row in range(0, len(artifact_specs), 3):
        cols_download = st.columns(3)
        for container, spec in zip(cols_download, artifact_specs[row : row + 3]):
            with container:
                download_artifact(*spec)

    with st.expander("Caminhos usados pelo app"):
        for path, _, _ in artifact_specs:
            if path.exists():
                st.write(f"`{path.relative_to(ROOT)}`")
