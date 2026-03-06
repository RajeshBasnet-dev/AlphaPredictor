from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import DataLoaderError, fetch_stock_data
from modules.model import ModelError, train_and_predict
from modules.visualization import build_price_chart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaPredictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme state ───────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def _toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ── Theme tokens ──────────────────────────────────────────────────────────────
T = {
    "dark": {
        "bg":            "#0f1117",
        "banner_bg":     "#0d1b2e",
        "card_bg":       "#161b27",
        "border":        "rgba(255,255,255,0.06)",
        "text":          "#e2e8f0",
        "text_muted":    "#475569",
        "label":         "#475569",
        "divider":       "rgba(255,255,255,0.05)",
        "input_bg":      "#161b27",
        "input_border":  "rgba(59,130,246,0.2)",
        "chart_bg":      "#1e2130",
        "grid":          "rgba(255,255,255,0.05)",
        "legend_color":  "#94a3b8",
        "pbar_track":    "rgba(255,255,255,0.06)",
        "sec_head":      "#3b82f6",
        "table_head_bg": "#1a2236",
        "table_row_bg":  "#161b27",
        "table_hover":   "#1a2236",
        "hero_before_opacity": "0.15",
        "toggle_label":  "☀  Light mode",
        "plotly_tmpl":   "plotly_dark",
    },
    "light": {
        "bg":            "#f5f7fa",
        "banner_bg":     "#eef2ff",
        "card_bg":       "#ffffff",
        "border":        "rgba(0,0,0,0.07)",
        "text":          "#0f172a",
        "text_muted":    "#64748b",
        "label":         "#64748b",
        "divider":       "rgba(0,0,0,0.06)",
        "input_bg":      "#ffffff",
        "input_border":  "rgba(59,130,246,0.35)",
        "chart_bg":      "#ffffff",
        "grid":          "rgba(0,0,0,0.06)",
        "legend_color":  "#475569",
        "pbar_track":    "rgba(0,0,0,0.08)",
        "sec_head":      "#2563eb",
        "table_head_bg": "#eef2ff",
        "table_row_bg":  "#ffffff",
        "table_hover":   "#f1f5f9",
        "hero_before_opacity": "0.08",
        "toggle_label":  "◑  Dark mode",
        "plotly_tmpl":   "plotly_white",
    },
}[st.session_state.theme]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {T["bg"]}; color: {T["text"]}; }}
#MainMenu, header, footer {{ visibility: hidden; }}
.block-container {{ padding: 0 3rem 3rem; max-width: 1440px; }}

/* ── Hero banner ── */
.hero-banner {{
    background: {T["banner_bg"]};
    border-bottom: 1px solid rgba(59,130,246,0.15);
    padding: 2.25rem 0 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}}
.hero-banner::before {{
    content: '';
    position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(59,130,246,{T["hero_before_opacity"]}) 0%, transparent 70%);
    pointer-events: none;
}}
@keyframes glow-pulse {{
    0%, 100% {{ text-shadow: 0 0 20px rgba(96,165,250,0.4), 0 0 40px rgba(167,139,250,0.15); }}
    50%       {{ text-shadow: 0 0 35px rgba(96,165,250,0.7), 0 0 70px rgba(167,139,250,0.35); }}
}}
.hero-title {{
    font-size: 2.9rem; font-weight: 900; letter-spacing: -2px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #10b981 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: glow-pulse 3s ease-in-out infinite;
    margin-bottom: 0.5rem;
}}
.hero-sub {{
    font-size: 0.82rem; color: {T["text_muted"]}; font-weight: 500;
    letter-spacing: 0.07em; text-transform: uppercase;
}}

/* ── KPI cards ── */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
.kpi-card {{
    background: {T["card_bg"]};
    border: 1px solid {T["border"]};
    border-radius: 14px; padding: 1.2rem 1.4rem;
    border-left: 4px solid var(--accent);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,{"0.45" if is_dark else "0.1"});
}}
.kpi-card.blue  {{ --accent: #3b82f6; }}
.kpi-card.green {{ --accent: #22c55e; }}
.kpi-card.red   {{ --accent: #ef4444; }}
.kpi-icon   {{ font-size: 1rem; margin-bottom: 0.35rem; }}
.kpi-label  {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {T["label"]}; margin-bottom: 0.5rem; }}
.kpi-value  {{ font-size: 1.7rem; font-weight: 800; color: {T["text"]}; line-height: 1.1; }}
.kpi-sub    {{ font-size: 0.77rem; font-weight: 500; margin-top: 0.4rem; }}
.txt-up     {{ color: #16a34a; }}
.txt-down   {{ color: #dc2626; }}
.txt-muted  {{ color: {T["text_muted"]}; }}

/* ── Progress bar ── */
.pbar-track {{ background: {T["pbar_track"]}; border-radius: 999px; height: 6px; margin-top: 0.6rem; overflow: hidden; }}
.pbar-fill  {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #3b82f6, #22c55e); }}

/* ── Signal badges ── */
.sig-bull {{ display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.35); color: #16a34a; border-radius: 999px; padding: 0.2rem 0.85rem; font-size: 0.86rem; font-weight: 700; }}
.sig-bear {{ display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35); color: #dc2626; border-radius: 999px; padding: 0.2rem 0.85rem; font-size: 0.86rem; font-weight: 700; }}

/* ── Section headers ── */
.sec-head {{ display: flex; align-items: center; gap: 0.75rem; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.14em; color: {T["sec_head"]}; margin: 2.25rem 0 1rem; }}
.sec-head::after {{ content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(59,130,246,0.3), transparent); }}

/* ── Prediction card ── */
.pred-card {{ background: {T["card_bg"]}; border: 1px solid {T["border"]}; border-radius: 16px; padding: 1.75rem 2rem; margin-bottom: 1.5rem; }}
.pred-title {{ font-size: 0.95rem; font-weight: 700; color: {T["text"]}; margin-bottom: 1.1rem; padding-bottom: 0.75rem; border-bottom: 1px solid {T["divider"]}; }}
.pred-row {{ display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; font-size: 0.88rem; border-bottom: 1px solid {T["divider"]}; }}
.pred-row:last-child {{ border-bottom: none; }}
.pred-key {{ color: {T["text_muted"]}; display: flex; align-items: center; gap: 0.5rem; }}
.pred-val {{ color: {T["text"]}; font-weight: 700; font-variant-numeric: tabular-nums; }}
.pred-ens {{ font-size: 1.6rem; font-weight: 800; color: #3b82f6; font-variant-numeric: tabular-nums; }}

/* ── Model badges ── */
.badge {{ display: inline-block; padding: 0.12rem 0.55rem; border-radius: 5px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
.badge-lr  {{ background: rgba(59,130,246,0.12);  color: #3b82f6;  border: 1px solid rgba(59,130,246,0.3);  }}
.badge-rf  {{ background: rgba(249,115,22,0.12);  color: #ea6800;  border: 1px solid rgba(249,115,22,0.3);  }}
.badge-xgb {{ background: rgba(34,197,94,0.12);   color: #16a34a;  border: 1px solid rgba(34,197,94,0.3);   }}

/* ── Metrics table ── */
.m-table {{ width:100%; border-collapse:separate; border-spacing:0; border-radius:14px; overflow:hidden; border:1px solid {T["border"]}; }}
.m-table th {{ background:{T["table_head_bg"]}; color:{T["text_muted"]}; font-size:0.67rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; padding:0.85rem 1.25rem; text-align:left; }}
.m-table td {{ background:{T["table_row_bg"]}; padding:0.85rem 1.25rem; font-size:0.88rem; color:{T["text"]}; font-variant-numeric:tabular-nums; border-top:1px solid {T["divider"]}; }}
.m-table tr:hover td {{ background:{T["table_hover"]}; }}
.cell-best  {{ color:#16a34a !important; font-weight:700; }}
.cell-worst {{ color:#dc2626 !important; font-weight:700; }}
.best-badge {{ display:inline-block; margin-left:0.5rem; background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.35); color:#16a34a; border-radius:4px; font-size:0.61rem; font-weight:700; padding:0.1rem 0.4rem; letter-spacing:0.05em; text-transform:uppercase; vertical-align:middle; }}

/* ── Streamlit widget overrides ── */
div[data-testid="stTextInput"] input {{
    background: {T["input_bg"]} !important; border: 1px solid {T["input_border"]} !important;
    border-radius: 10px !important; color: {T["text"]} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: rgba(59,130,246,0.55) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important; outline: none !important;
}}
div[data-testid="stTextInput"] label {{ color: {T["text_muted"]} !important; font-size: 0.8rem !important; font-weight: 500 !important; }}
div[data-testid="stButton"] button {{
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important; border-radius: 10px !important; color: #fff !important;
    font-weight: 700 !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important; letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.15s !important;
}}
div[data-testid="stButton"] button:hover {{ opacity: 0.88 !important; transform: translateY(-1px) !important; }}
div[data-testid="stExpander"] {{ background: {T["card_bg"]} !important; border: 1px solid {T["border"]} !important; border-radius: 12px !important; }}
div[data-testid="stAlert"] {{ border-radius: 12px !important; }}

.empty {{ text-align:center; padding:5rem 2rem; }}
.empty-icon {{ font-size:2.5rem; margin-bottom:1rem; opacity:0.3; }}
.empty-text {{ font-size:0.95rem; color:{T["text_muted"]}; }}
.empty-text strong {{ color:#3b82f6; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: metrics table HTML ────────────────────────────────────────────────
def _metrics_table(metrics_df, best_model: str) -> str:
    num_cols = ["MAE", "RMSE", "MAPE (%)"]
    rows = [{c: float(r[c]) for c in num_cols} for _, r in metrics_df.iterrows()]
    col_min = {c: min(r[c] for r in rows) for c in num_cols}
    col_max = {c: max(r[c] for r in rows) for c in num_cols}

    headers = "<th>Model</th>" + "".join(f"<th>{c}</th>" for c in num_cols)
    body = ""
    for _, row in metrics_df.iterrows():
        name = row["Model"]
        tag = '<span class="best-badge">Best</span>' if name == best_model else ""
        tds = f"<td><strong>{name}</strong>{tag}</td>"
        for c in num_cols:
            v = float(row[c])
            if v == col_min[c]:
                cls = ' class="cell-best"'
            elif v == col_max[c]:
                cls = ' class="cell-worst"'
            else:
                cls = ""
            tds += f"<td{cls}>{v}</td>"
        body += f"<tr>{tds}</tr>"
    return f'<table class="m-table"><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>'


# ── Hero + theme toggle ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">◈ AlphaPredictor</div>
  <div class="hero-sub">Powered by XGBoost &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Linear Regression</div>
</div>
""", unsafe_allow_html=True)

# Theme toggle sits right below the banner in a right-aligned column
_, toggle_col = st.columns([10, 1])
with toggle_col:
    st.button(T["toggle_label"], on_click=_toggle_theme, key="theme_toggle")

# ── Input row ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Examples: AAPL, TSLA, MSFT, NVDA, GOOGL")
with col2:
    st.write("")
    st.write("")
    predict_clicked = st.button("Predict →", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Fetching data and training models..."):
        try:
            stock_data = fetch_stock_data(ticker=ticker, period_years=5)
            result = train_and_predict(stock_data)
        except DataLoaderError as exc:
            st.error(f"Data error: {exc}")
            st.stop()
        except ModelError as exc:
            st.error(f"Model error: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    latest  = result.latest_close
    pred    = result.ensemble_prediction
    delta   = result.percent_change
    conf    = result.confidence
    is_bull = delta >= 0

    arrow        = "▲" if is_bull else "▼"
    delta_cls    = "txt-up" if is_bull else "txt-down"
    sig_cls      = "sig-bull" if is_bull else "sig-bear"
    signal_label = f"{arrow} Bullish" if is_bull else f"{arrow} Bearish"
    card_sig_cls = "green" if is_bull else "red"
    sig_color    = "#16a34a" if is_bull else "#dc2626"

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <div class="kpi-icon">💰</div>
        <div class="kpi-label">Latest Close</div>
        <div class="kpi-value">${latest:,.2f}</div>
        <div class="kpi-sub txt-muted">Most recent closing price</div>
      </div>
      <div class="kpi-card {card_sig_cls}">
        <div class="kpi-icon">{"📈" if is_bull else "📉"}</div>
        <div class="kpi-label">Predicted Next Close</div>
        <div class="kpi-value">${pred:,.2f}</div>
        <div class="kpi-sub {delta_cls}">{arrow}&nbsp;{delta:+.2f}% expected change</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-label">Model Confidence</div>
        <div class="kpi-value">{conf:.1f}<span style="font-size:1rem;color:{T["text_muted"]}">%</span></div>
        <div class="pbar-track"><div class="pbar-fill" style="width:{conf:.1f}%"></div></div>
      </div>
      <div class="kpi-card {card_sig_cls}">
        <div class="kpi-icon">{"📈" if is_bull else "📉"}</div>
        <div class="kpi-label">Trend Signal</div>
        <div class="kpi-value" style="margin-top:0.4rem">
          <span class="{sig_cls}">{signal_label}</span>
        </div>
        <div class="kpi-sub txt-muted">{ticker.upper()} next-day forecast</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction summary ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Prediction Breakdown</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pred-card">
      <div class="pred-title">Next-Day Price Forecast — {ticker.upper()}</div>
      <div class="pred-row">
        <span class="pred-key"><span class="badge badge-lr">LR</span> Linear Regression</span>
        <span class="pred-val">${result.linear_prediction:,.2f}</span>
      </div>
      <div class="pred-row">
        <span class="pred-key"><span class="badge badge-rf">RF</span> Random Forest</span>
        <span class="pred-val">${result.random_forest_prediction:,.2f}</span>
      </div>
      <div class="pred-row">
        <span class="pred-key"><span class="badge badge-xgb">XGB</span> XGBoost</span>
        <span class="pred-val">${result.xgboost_prediction:,.2f}</span>
      </div>
      <div style="border-top:1px solid rgba(59,130,246,0.15);margin-top:0.75rem;padding-top:1rem;">
        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{T["text_muted"]};margin-bottom:0.4rem;">Ensemble Average</div>
        <div class="pred-ens">${pred:,.2f}</div>
        <div style="margin-top:0.35rem;font-size:0.84rem;color:{T["text_muted"]};">95% CI: ${result.prediction_interval_low:,.2f} – ${result.prediction_interval_high:,.2f}</div>
        <div style="margin-top:0.5rem;display:flex;align-items:center;gap:0.75rem;">
          <span class="{sig_cls}">{signal_label}</span>
          <span style="font-size:0.88rem;color:{sig_color}">{delta:+.2f}% expected</span>
        </div>
        <div style="margin-top:0.75rem;">
          <div style="font-size:0.71rem;color:{T["text_muted"]};margin-bottom:0.35rem;">Confidence: {conf:.1f}%</div>
          <div class="pbar-track" style="max-width:320px;"><div class="pbar-fill" style="width:{conf:.1f}%"></div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Model Evaluation — Held-Out Test Set</div>', unsafe_allow_html=True)
    metrics_df = result.metrics_table.copy()
    metrics_df["MAE"]  = metrics_df["MAE"].map(lambda v: round(float(v), 4))
    metrics_df["RMSE"] = metrics_df["RMSE"].map(lambda v: round(float(v), 4))
    metrics_df["MAPE"] = metrics_df["MAPE"].map(lambda v: round(float(v) * 100, 2))
    metrics_df = metrics_df.rename(columns={"MAPE": "MAPE (%)"})
    best_model = metrics_df.loc[metrics_df["MAPE (%)"].idxmin(), "Model"]
    st.markdown(_metrics_table(metrics_df, best_model), unsafe_allow_html=True)

    # ── Actual vs Predicted chart ──────────────────────────────────────────────
    st.markdown('<div class="sec-head">Actual vs Predicted — Test Period</div>', unsafe_allow_html=True)
    tp = result.test_predictions
    cmp = go.Figure()
    cmp.add_trace(go.Scatter(x=tp.index, y=tp["Actual"],
        mode="lines", name="Actual", line=dict(color="#0f172a" if not is_dark else "#f8fafc", width=2.5)))
    cmp.add_trace(go.Scatter(x=tp.index, y=tp["Linear Regression"],
        mode="lines", name="Linear Regression", line=dict(color="#3b82f6", width=1.8, dash="dot")))
    cmp.add_trace(go.Scatter(x=tp.index, y=tp["Random Forest"],
        mode="lines", name="Random Forest", line=dict(color="#f97316", width=1.8, dash="dot")))
    cmp.add_trace(go.Scatter(x=tp.index, y=tp["XGBoost"],
        mode="lines", name="XGBoost", line=dict(color="#22c55e", width=1.8, dash="dot")))
    split_x = str(tp.index[0])
    cmp.add_shape(
        type="line",
        x0=split_x, x1=split_x, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(139,92,246,0.6)", width=1.5, dash="dash"),
    )
    cmp.add_annotation(
        x=split_x, yref="paper", y=1.02,
        text="Test split", showarrow=False,
        font=dict(color="#8b5cf6", size=11),
        xanchor="left",
    )
    cmp.update_layout(
        template=T["plotly_tmpl"],
        paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_bg"],
        xaxis=dict(showgrid=True, gridcolor=T["grid"], title="Date", color=T["text_muted"]),
        yaxis=dict(showgrid=True, gridcolor=T["grid"], title="Price (USD)", color=T["text_muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color=T["legend_color"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(cmp, use_container_width=True)

    # ── Historical price chart ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Historical Price Trend</div>', unsafe_allow_html=True)
    hist = build_price_chart(stock_data, ticker)
    hist.update_layout(
        template=T["plotly_tmpl"],
        paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_bg"],
        xaxis=dict(showgrid=True, gridcolor=T["grid"], color=T["text_muted"]),
        yaxis=dict(showgrid=True, gridcolor=T["grid"], color=T["text_muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color=T["legend_color"]), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(hist, use_container_width=True)

    # ── Raw predictions expander ───────────────────────────────────────────────
    with st.expander("Raw Model Predictions", expanded=False):
        st.write({
            "Linear Regression": round(result.linear_prediction, 2),
            "Random Forest":     round(result.random_forest_prediction, 2),
            "XGBoost":           round(result.xgboost_prediction, 2),
            "Ensemble":          round(result.ensemble_prediction, 2),
        })

else:
    st.markdown(f"""
    <div class="empty">
      <div class="empty-icon">◈</div>
      <div class="empty-text">
        Enter a stock ticker above and click <strong>Predict →</strong> to generate a next-day forecast.
      </div>
    </div>
    """, unsafe_allow_html=True)