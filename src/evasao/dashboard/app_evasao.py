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

PUBLIC_URL = "https://appevasaosbie.streamlit.app/"
REPO_URL = "https://github.com/EddieFerb/appevasao-sbie-deploy"
INEP_URL = "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior"

EXPECTED = {
    "r2": 0.3337047249133319,
    "mae": 0.1778456230862687,
    "rmse": 0.2330517498461937,
    "mse": 0.05431311810637285,
}

REVIEW_COUNTS = {
    "studies": 88,
    "universe_a": 78,
    "universe_b": 10,
    "random_forest": 49,
    "artifacts": 14,
    "inep_rf": 7,
    "temporal_validation": 7,
    "without_artifacts": 74,
    "imported": 3827,
    "triage": 2350,
    "full_text": 119,
}

st.set_page_config(
    page_title="AppEvasão — Evidência SBIE 2026",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --bg0: #07111f;
  --bg1: #0d1c30;
  --glass: rgba(255, 255, 255, .105);
  --glass-strong: rgba(255, 255, 255, .16);
  --line: rgba(255, 255, 255, .22);
  --text: #f8fbff;
  --muted: rgba(235, 244, 255, .76);
  --soft: rgba(235, 244, 255, .58);
  --cyan: #7dd3fc;
  --blue: #60a5fa;
  --green: #8ee6c9;
  --amber: #f6d365;
}
.stApp {
  color: var(--text);
  background:
    radial-gradient(circle at 18% 12%, rgba(96, 165, 250, .30), transparent 30rem),
    radial-gradient(circle at 80% 5%, rgba(142, 230, 201, .18), transparent 26rem),
    radial-gradient(circle at 54% 78%, rgba(246, 211, 101, .11), transparent 28rem),
    linear-gradient(135deg, var(--bg0), var(--bg1) 48%, #101828);
}
.main .block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: rgba(6, 14, 27, .76); border-right: 1px solid rgba(255, 255, 255, .10); }
h1, h2, h3, h4, p, li, label, span, div { letter-spacing: 0; }
.glass-hero {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 28px;
  padding: 2rem;
  background: linear-gradient(135deg, rgba(255,255,255,.19), rgba(255,255,255,.07));
  box-shadow: 0 24px 70px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.28);
  backdrop-filter: blur(20px);
  margin-bottom: 1.2rem;
}
.glass-hero:after {
  content: "";
  position: absolute;
  inset: auto -12% -35% 28%;
  height: 10rem;
  background: linear-gradient(90deg, rgba(125,211,252,.30), rgba(142,230,201,.22), rgba(246,211,101,.18));
  filter: blur(26px);
}
.hero-layout {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, .75fr);
  gap: 1.35rem;
  align-items: center;
}
.glass-hero .eyebrow { color: var(--green); font-size: .82rem; font-weight: 800; text-transform: uppercase; }
.glass-hero h1 { max-width: 860px; margin: .35rem 0 .65rem; font-size: clamp(2rem, 3.8vw, 4.2rem); line-height: 1; color: var(--text); }
.glass-hero p { max-width: 880px; margin: 0; color: var(--muted); font-size: 1.08rem; line-height: 1.55; }
.hero-evidence {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .55rem .85rem;
  padding: .25rem 0 .2rem;
  background: transparent;
  border: 0;
  box-shadow: none;
}
.chip-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.05rem; }
.story-chip {
  position: relative;
  display: block;
  padding: .18rem 0 .18rem 1rem;
  color: rgba(255,255,255,.86);
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  font-size: .92rem;
  font-weight: 650;
  line-height: 1.32;
}
.story-chip:before {
  content: "";
  position: absolute;
  left: 0;
  top: .72rem;
  width: .42rem;
  height: .42rem;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--cyan), var(--green));
  box-shadow: 0 0 14px rgba(125,211,252,.42);
}
.section-title { margin: .6rem 0 .45rem; color: var(--text); font-size: 1.35rem; font-weight: 800; }
.glass-panel, .source-card {
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1.05rem 1.15rem;
  background: rgba(255,255,255,.095);
  box-shadow: 0 18px 50px rgba(0,0,0,.22);
  backdrop-filter: blur(16px);
  margin-bottom: 1rem;
}
.glass-panel p, .source-card p { color: var(--muted); margin: 0; line-height: 1.55; }
.evidence-note {
  border-left: 4px solid var(--green);
  border-radius: 18px;
  padding: .9rem 1rem;
  background: rgba(142,230,201,.10);
  color: rgba(248,251,255,.90);
  margin: .8rem 0 1rem;
}
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .7rem 0 1.1rem; }
.kpi-card, .metric-card, .governance-card, .roadmap-card {
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 22px;
  padding: 1rem;
  background: linear-gradient(150deg, rgba(255,255,255,.15), rgba(255,255,255,.06));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.20), 0 16px 38px rgba(0,0,0,.18);
  backdrop-filter: blur(14px);
  min-height: 126px;
}
.kpi-label, .metric-label { color: var(--soft); font-size: .82rem; font-weight: 750; text-transform: uppercase; }
.kpi-value, .metric-value { color: var(--text); font-size: 2.25rem; font-weight: 850; line-height: 1.08; margin-top: .2rem; white-space: nowrap; }
.kpi-hint, .metric-hint { color: var(--muted); font-size: .84rem; margin-top: .42rem; line-height: 1.35; }
.metric-value { color: var(--cyan); }
.value-flow { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .72rem; margin: .8rem 0 1.2rem; }
.value-step {
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 20px;
  padding: .9rem;
  background: rgba(255,255,255,.085);
  min-height: 160px;
}
.value-step .num { color: var(--green); font-size: .78rem; font-weight: 850; }
.value-step strong { display: block; margin: .35rem 0 .3rem; color: var(--text); font-size: .95rem; }
.value-step span { color: var(--muted); font-size: .83rem; line-height: 1.38; }
.stTabs [data-baseweb="tab-list"] {
  gap: .75rem;
  flex-wrap: wrap;
  border-bottom: 1px solid rgba(255,255,255,.14);
  padding-bottom: .55rem;
}
.stTabs [data-baseweb="tab"] {
  min-height: 58px;
  padding: .85rem 1.25rem;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 18px 18px 6px 6px;
  background: rgba(255,255,255,.07);
  color: rgba(255,255,255,.88);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 10px 28px rgba(0,0,0,.16);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  transition: transform .16s ease, background .16s ease, border-color .16s ease, box-shadow .16s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  transform: translateY(-1px);
  background: rgba(255,255,255,.11);
  border-color: rgba(255,255,255,.30);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(120,190,210,.26), rgba(255,255,255,.10));
  border-color: rgba(255,255,255,.38);
  color: #fff;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.25), 0 14px 36px rgba(0,0,0,.22);
}
.stTabs [data-baseweb="tab"] p {
  font-size: 24px !important;
  font-weight: 750 !important;
  letter-spacing: 0;
  color: rgba(255,255,255,.94) !important;
}
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,.15); border-radius: 18px; overflow: hidden; }
a { color: var(--cyan); }
@media (max-width: 1050px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .value-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hero-layout { grid-template-columns: 1fr; } }
@media (max-width: 680px) { .kpi-grid, .value-flow, .hero-evidence { grid-template-columns: 1fr; } .glass-hero { padding: 1.35rem; border-radius: 22px; } .kpi-value, .metric-value { font-size: 1.85rem; } .stTabs [data-baseweb="tab"] { min-height: 52px; padding: .72rem 1rem; } .stTabs [data-baseweb="tab"] p { font-size: 20px !important; } }
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
        raise FileNotFoundError(f"Artefato não encontrado: {BENCHMARK_CSV}")
    return df


def random_forest_row(df: pd.DataFrame) -> pd.Series:
    mask = df.astype(str).apply(
        lambda row: row.str.contains("Random Forest", case=False, na=False).any(),
        axis=1,
    )
    if not mask.any():
        raise ValueError("Linha Random Forest não encontrada no CSV do benchmark.")
    return df[mask].iloc[0]


def fmt_decimal(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_int(value: object) -> str:
    return f"{int(float(value)):,}".replace(",", ".")


def kpi_card(label: str, value: object, hint: str, numeric: bool = True) -> str:
    formatted = fmt_int(value) if numeric else str(value)
    return f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{formatted}</div>
  <div class="kpi-hint">{hint}</div>
</div>
"""


def metric_card(label: str, value: object, hint: str, digits: int = 4) -> str:
    return f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{fmt_decimal(value, digits)}</div>
  <div class="metric-hint">{hint}</div>
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
        return ["background-color: #e7f5ff; font-weight: 700" if is_rf else "" for _ in row]

    return table.style.apply(highlight_rf, axis=1).format(precision=4, thousands=".", decimal=",")


def download_artifact(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(label=label, data=path.read_bytes(), file_name=path.name, mime=mime, width="stretch")
    else:
        st.caption(f"Indisponível: {path.relative_to(ROOT)}")


def value_flow_markup() -> str:
    steps = [
        ("01", "Fonte oficial", "Microdados do Censo da Educação Superior — INEP/MEC."),
        ("02", "Agregação e preparação", "Dados tratados e agregados para modelagem temporal."),
        ("03", "Formulação temporal", "Variáveis observadas em t → taxa_evasao_alvo em t+1."),
        ("04", "Validação", "Treino: ano_base <= 2018. Teste: ano_base > 2018."),
        ("05", "Comparação", "Média histórica, Persistência, Regressão Linear e Random Forest."),
        ("06", "Evidência", "Métricas fixas lidas dos artefatos versionados."),
    ]
    cards = "\n".join(
        f'<div class="value-step"><div class="num">{num}</div><strong>{title}</strong><span>{body}</span></div>'
        for num, title, body in steps
    )
    return f'<div class="value-flow">{cards}</div>'


def source_card_markup() -> str:
    return f"""
<div class="source-card">
  <div class="section-title">Fonte dos dados</div>
  <p><strong>Fonte oficial:</strong> Microdados do Censo da Educação Superior — INEP/MEC.
  Os microdados públicos foram tratados e agregados para construção do benchmark temporal apresentado neste artigo.</p>
  <p style="margin-top:.65rem;"><a href="{INEP_URL}" target="_blank">Página oficial dos microdados do Censo da Educação Superior</a></p>
</div>
"""


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
            colorscale="Teal",
            showscale=False,
        )
    )
    fig.update_layout(title=title, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20), height=410)
    st.plotly_chart(fig, width="stretch")


def apply_plot_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#edf6ff",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


st.sidebar.title("AppEvasão")
st.sidebar.markdown("**Evidência SBIE 2026**")
st.sidebar.markdown("Aplicação:")
st.sidebar.link_button("Abrir app público", PUBLIC_URL)
st.sidebar.markdown("Repositório:")
st.sidebar.link_button("GitHub", REPO_URL)
st.sidebar.markdown("Fonte: **INEP/MEC — Censo da Educação Superior**")
st.sidebar.link_button("Link oficial dos microdados", INEP_URL)
st.sidebar.caption("Main file path: `src/evasao/dashboard/app_evasao.py`")

try:
    df = load_benchmark()
    cols = column_lookup(df)
    rf = random_forest_row(df)
    missing = [name for name in ["modelo", "r2", "mae", "rmse", "mse"] if name not in cols]
    if missing:
        st.error(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")
        st.stop()
except Exception as exc:
    st.error(str(exc))
    st.stop()

warnings = validate_expected(rf, cols)
if warnings:
    st.warning("Divergência em relação aos valores auditados: " + "; ".join(warnings))

table = baseline_table(df, cols)
feature_importance = read_csv(FEATURE_IMPORTANCE_CSV)
completude = read_csv(COMPLETUDE_CSV)
balanceamento = read_csv(BALANCEAMENTO_CSV)

st.markdown(
    """
<div class="glass-hero">
  <div class="hero-layout">
    <div>
      <div class="eyebrow">AppEvasão — Evidência SBIE 2026</div>
      <h1>DA EVASÃO NO ENSINO SUPERIOR AO BENCHMARK TEMPORAL</h1>
      <p>Uma revisão estruturada identificou lacunas. O benchmark temporal transforma essas lacunas em evidência pública: dados oficiais, separação cronológica, baselines explícitos e métricas versionadas.</p>
    </div>
    <div class="hero-evidence">
      <span class="story-chip">Dados oficiais do INEP/MEC</span>
      <span class="story-chip">88 estudos mapeados</span>
      <span class="story-chip">Subgrupo RF — estudos com Random Forest</span>
      <span class="story-chip">Validação temporal</span>
      <span class="story-chip">Prevenção de vazamento de dados</span>
      <span class="story-chip">Artefato computacional de apoio à decisão</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "História",
        "Revisão em Números",
        "Lacunas",
        "Benchmark Temporal",
        "Evidências Visuais",
        "Diagnósticos e Limites",
        "Artefatos",
    ]
)

with tabs[0]:
    st.markdown(
        """
<div class="glass-panel">
  <div class="section-title">Tese da página</div>
  <p>O objetivo não é prometer uma solução universal para evasão, mas tornar visível, verificável e discutível uma formulação temporal baseada em dados oficiais, baselines explícitos e métricas versionadas.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(source_card_markup(), unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass-panel">
  <div class="section-title">Como ler os termos</div>
  <p>Neste painel, <strong>Universo A</strong> indica estudos sem uso explícito de dados INEP/MEC; <strong>Universo B</strong> indica estudos com uso explícito de dados INEP/MEC; e <strong>Subgrupo RF</strong> indica estudos que utilizaram Random Forest. Assim, INEP/MEC e Random Forest são dimensões analíticas distintas.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(value_flow_markup(), unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Revisão em Números</div>', unsafe_allow_html=True)
    for row in [
        [
            ("Registros importados", REVIEW_COUNTS["imported"], "entrada inicial da revisão"),
            ("Estudos incluídos", REVIEW_COUNTS["studies"], "corpus final"),
            ("Universo A — sem INEP/MEC explícito", REVIEW_COUNTS["universe_a"], "estudos sem uso explícito de dados INEP/MEC"),
            ("Universo B — com INEP/MEC explícito", REVIEW_COUNTS["universe_b"], "estudos com uso explícito de dados INEP/MEC"),
        ],
        [
            ("Subgrupo RF — estudos com Random Forest", REVIEW_COUNTS["random_forest"], "estudos que utilizaram Random Forest"),
            ("Artefatos", REVIEW_COUNTS["artifacts"], "materiais computacionais disponíveis"),
            ("Subgrupo INEP/MEC + RF — estudos com INEP/MEC explícito e Random Forest", REVIEW_COUNTS["inep_rf"], "interseção metodológica"),
            ("Validação temporal", REVIEW_COUNTS["temporal_validation"], "aproximadamente 8,0% dos estudos"),
        ],
    ]:
        cols_cards = st.columns(4)
        for container, (label, value, hint) in zip(cols_cards, row):
            with container:
                st.markdown(kpi_card(label, value, hint), unsafe_allow_html=True)

    funnel = pd.DataFrame(
        [
            ("Importados", REVIEW_COUNTS["imported"]),
            ("Triagem", REVIEW_COUNTS["triage"]),
            ("Full text", REVIEW_COUNTS["full_text"]),
            ("Incluídos", REVIEW_COUNTS["studies"]),
        ],
        columns=["Etapa", "Registros"],
    )
    fig = px.funnel(funnel, x="Registros", y="Etapa", title="Funil da revisão estruturada")
    st.plotly_chart(apply_plot_style(fig), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        universe = pd.DataFrame(
            [
                ("Universo A — sem INEP/MEC explícito", 78),
                ("Universo B — com INEP/MEC explícito", 10),
            ],
            columns=["Universo", "Estudos"],
        )
        st.plotly_chart(apply_plot_style(px.bar(universe, x="Universo", y="Estudos", text="Estudos", title="Universo A e Universo B por uso explícito de INEP/MEC")), width="stretch")
    with c2:
        signals = pd.DataFrame(
            [
                ("Subgrupo RF — estudos com Random Forest", 49),
                ("Artefato computacional de apoio à decisão", 14),
                ("Subgrupo INEP/MEC + RF — estudos com INEP/MEC explícito e Random Forest", 7),
            ],
            columns=["Sinal", "Estudos"],
        )
        st.plotly_chart(apply_plot_style(px.bar(signals, x="Sinal", y="Estudos", text="Estudos", title="Sinais de aproximação ao benchmark")), width="stretch")

with tabs[2]:
    st.markdown(
        """
<div class="glass-panel">
  <div class="section-title">Lacunas metodológicas</div>
  <p>A revisão mostrou que a literatura brasileira é rica em experimentos, mas ainda apresenta baixa padronização comparativa: poucos estudos usam INEP/MEC explicitamente, poucos usam validação temporal e poucos convertem modelos em artefatos computacionais.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cols_gap = st.columns(4)
    gap_cards = [
        ("Validação temporal", 7, "estudos"),
        ("Artefatos computacionais", 14, "estudos"),
        ("INEP/MEC explícito", 10, "estudos"),
        ("Subgrupo INEP/MEC + RF — estudos com INEP/MEC explícito e Random Forest", 7, "estudos"),
    ]
    for container, (label, value, hint) in zip(cols_gap, gap_cards):
        with container:
            st.markdown(kpi_card(label, value, hint), unsafe_allow_html=True)

    validations = pd.DataFrame(
        [("Cross-validation", 31), ("Holdout", 21), ("Temporal", 7), ("Não clara", 29)],
        columns=["Validação", "Estudos"],
    )
    metrics = pd.DataFrame(
        [("Accuracy", 40), ("Precision", 22), ("Recall", 24), ("F1", 27), ("MAE", 4), ("MSE", 3), ("RMSE", 1), ("R²", 1)],
        columns=["Métrica", "Estudos"],
    )
    targets = pd.DataFrame(
        [("Risco de evasão", 32), ("Permanência/retenção", 18), ("Evasão binária", 29), ("Taxa de evasão", 9)],
        columns=["Tipo de alvo", "Estudos"],
    )
    st.plotly_chart(apply_plot_style(px.bar(validations, x="Validação", y="Estudos", text="Estudos", title="Estratégias de validação reportadas")), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(apply_plot_style(px.bar(metrics, x="Métrica", y="Estudos", text="Estudos", title="Métricas reportadas")), width="stretch")
    with c2:
        st.plotly_chart(apply_plot_style(px.bar(targets, x="Tipo de alvo", y="Estudos", text="Estudos", title="Tipos de alvo")), width="stretch")

with tabs[3]:
    st.markdown('<div class="section-title">Benchmark Temporal</div>', unsafe_allow_html=True)
    st.markdown(value_flow_markup(), unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_specs = [("R²", "r2", "maior é melhor"), ("MAE", "mae", "menor é melhor"), ("RMSE", "rmse", "menor é melhor"), ("MSE", "mse", "menor é melhor")]
    for container, (label, key, hint) in zip(metric_cols, metric_specs):
        with container:
            st.markdown(metric_card(label, rf[cols[key]], hint), unsafe_allow_html=True)
    st.markdown('<div class="evidence-note">As métricas desta seção são fixas e foram lidas diretamente de <code>comparacao_baselines_temporal.csv</code>.</div>', unsafe_allow_html=True)
    st.dataframe(highlighted_baseline_table(table), width="stretch", hide_index=True)

with tabs[4]:
    st.markdown('<div class="section-title">Evidências Visuais</div>', unsafe_allow_html=True)
    plot_df = df.copy()
    for name in ["r2", "mae", "rmse", "mse", "accuracy", "precision", "recall", "f1"]:
        if name in cols:
            plot_df[cols[name]] = pd.to_numeric(plot_df[cols[name]], errors="coerce")

    fig_r2 = px.bar(plot_df, x=cols["modelo"], y=cols["r2"], text=plot_df[cols["r2"]].map(lambda value: f"{value:.4f}"), title="R² por modelo no holdout temporal")
    st.plotly_chart(apply_plot_style(fig_r2), width="stretch")

    error_cols = [cols[name] for name in ["mae", "rmse", "mse"] if name in cols]
    errors_long = plot_df[[cols["modelo"], *error_cols]].melt(id_vars=cols["modelo"], var_name="Métrica", value_name="Valor")
    st.plotly_chart(apply_plot_style(px.bar(errors_long, x=cols["modelo"], y="Valor", color="Métrica", barmode="group", title="MAE, RMSE e MSE por modelo")), width="stretch")

    aux_keys = ["accuracy", "precision", "recall", "f1"]
    if all(key in cols for key in aux_keys):
        aux_df = pd.DataFrame({"Métrica": ["Accuracy", "Precision", "Recall", "F1"], "Valor": [float(rf[cols[key]]) for key in aux_keys]})
        fig_aux = px.bar(aux_df, x="Métrica", y="Valor", text=aux_df["Valor"].map(lambda value: f"{value:.4f}"), title="Métricas classificatórias auxiliares da Random Forest")
        fig_aux.update_yaxes(range=[0, 1])
        st.plotly_chart(apply_plot_style(fig_aux), width="stretch")

    cm_cols = ["cm_tn", "cm_fp", "cm_fn", "cm_tp"]
    if all(name in cols for name in cm_cols):
        fixed = {"tn": int(rf[cols["cm_tn"]]), "fp": int(rf[cols["cm_fp"]]), "fn": int(rf[cols["cm_fn"]]), "tp": int(rf[cols["cm_tp"]])}
        render_confusion_matrix(fixed, "Matriz de confusão fixa - Random Forest, threshold 0,50")

    st.info(
        "Esta seção usa apenas métricas fixas e artefatos versionados. "
        "Não há treino, ajuste de threshold ou recálculo do benchmark em runtime."
    )

with tabs[5]:
    st.markdown('<div class="section-title">Diagnósticos e Limites</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass-panel">
  <p><strong>Moderado não é fraco:</strong> no contexto temporal e de prevenção de vazamento de dados, o resultado é uma referência comparável e auditável, não uma promessa de acerto absoluto.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
- Ausência de `ano_base=2020` após critério de completude.
- Dados agregados, não individuais.
- Dependência do fechamento anual para uso de `concluintes_t`.
- Leitura binária alta/baixa evasão apenas como apoio interpretativo.
- Desempenho regressivo moderado, coerente com a dificuldade temporal do problema.
"""
    )
    if feature_importance is not None and {"feature", "importance"}.issubset(feature_importance.columns):
        fi = feature_importance.sort_values("importance", ascending=True)
        st.plotly_chart(apply_plot_style(px.bar(fi, x="importance", y="feature", orientation="h", title="Importância relativa dos atributos - Random Forest")), width="stretch")
    if completude is not None:
        comp = completude.copy()
        comp["percentual_perda"] = pd.to_numeric(comp["percentual_perda"], errors="coerce")
        st.plotly_chart(apply_plot_style(px.line(comp, x="ano_base", y="percentual_perda", markers=True, title="Perda após dropna por ano-base")), width="stretch")
        st.dataframe(comp, width="stretch", hide_index=True)
    if balanceamento is not None:
        bal = balanceamento.copy()
        st.plotly_chart(apply_plot_style(px.bar(bal, x="conjunto", y=["prop_baixa_evasao", "prop_alta_evasao"], barmode="group", title="Balanceamento das classes por conjunto")), width="stretch")
        st.dataframe(bal, width="stretch", hide_index=True)

with tabs[6]:
    st.markdown('<div class="section-title">Artefatos</div>', unsafe_allow_html=True)
    st.markdown(source_card_markup(), unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="glass-panel">
  <p><strong>Repositório:</strong> <a href="{REPO_URL}" target="_blank">{REPO_URL}</a></p>
  <p><strong>Aplicação:</strong> <a href="{PUBLIC_URL}" target="_blank">{PUBLIC_URL}</a></p>
  <p style="margin-top:.65rem;">Este é um repositório minimalista de deploy. O pipeline científico completo não é publicado nesta versão por integrar artefato institucional em desenvolvimento. As métricas, variáveis, critérios de filtragem e divisão temporal estão descritos no artigo.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    artifact_specs = [
        (BENCHMARK_CSV, "Benchmark CSV", "text/csv"),
        (BENCHMARK_MD, "Benchmark MD", "text/markdown"),
        (METRICAS_TXT, "Métricas TXT", "text/plain"),
        (FEATURE_IMPORTANCE_CSV, "Importância CSV", "text/csv"),
        (FEATURE_IMPORTANCE_MD, "Importância MD", "text/markdown"),
        (COMPLETUDE_CSV, "Completude CSV", "text/csv"),
        (COMPLETUDE_MD, "Completude MD", "text/markdown"),
        (BALANCEAMENTO_CSV, "Balanceamento CSV", "text/csv"),
        (BALANCEAMENTO_MD, "Balanceamento MD", "text/markdown"),
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
