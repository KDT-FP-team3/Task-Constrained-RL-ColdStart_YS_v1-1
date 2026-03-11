import streamlit as st
import importlib
import os
import sys
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name, get_fee_info
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation_with_log
from common.evaluator import calculate_ctpt_and_color, calculate_mdd

# 루트 경로 설정
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

st.set_page_config(page_title="Chainers Master Fund", layout="wide", initial_sidebar_state="collapsed")

# ── 반응형 레이아웃 & 다크/라이트 모드 공통 CSS ──
st.markdown("""
<style>
/* 상단 여백 최소화 */
.block-container { padding-top: 0 !important; padding-bottom: 1rem !important; }
/* metric 카드 compact */
[data-testid="stMetric"] { padding: 4px 8px !important; }
/* expander 내부 여백 축소 */
[data-testid="stExpander"] > div:last-child { padding: 0.5rem 0.75rem !important; }
/* plotly 차트 마진 제거 */
[data-testid="stPlotlyChart"] { margin-bottom: 0 !important; }
/* 다크/라이트 모드 div 테두리 */
.st-summary-card { border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; padding: 12px 14px; }
/* ── sticky-header-marker 숨김 ── */
.element-container:has(.sticky-header-marker) {
    display: none !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
}
/* ── 헤더 컨테이너 sticky 고정 ── */
[data-testid="stVerticalBlock"]:has(> .element-container > [data-testid="stMarkdownContainer"] .sticky-header-marker) {
    position: sticky !important;
    top: var(--header-height, 2.875rem) !important;
    z-index: 999 !important;
    background-color: var(--background-color, #0e1117) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    padding-bottom: 2px !important;
}
/* ── 헤더 제목 소형화 ── */
.sticky-main-title {
    font-size: 1.4rem !important; font-weight: 700 !important;
    margin: 0.15rem 0 !important; line-height: 1.3 !important;
}
.sticky-sub-title {
    font-size: 1.1rem !important; font-weight: 700 !important;
    margin: 0.05rem 0 !important;
}
.sticky-divider { margin: 0.15rem 0 !important; border-color: rgba(128,128,128,0.3) !important; }
/* Run Evaluation / Simulation 버튼 공통: 텍스트 맞춤 너비, 균일 높이 */
[data-testid="stButton"] button[kind="primary"] {
    min-height: 2.4rem !important;
    height: 2.4rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    white-space: nowrap !important;
}
/* btn-pair-marker 컨테이너 숨김 */
.element-container:has(.btn-pair-marker) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* 두 버튼이 담긴 stVerticalBlock을 가로 배치로 전환 — > 직접 자식으로 범위 제한 */
[data-testid="stVerticalBlock"]:has(> .element-container > [data-testid="stMarkdownContainer"] .btn-pair-marker) {
    flex-direction: row !important;
    align-items: flex-end !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
}
/* Simulation 버튼 보라색 — 버튼 쌍 중 마지막 primary 버튼 */
[data-testid="stVerticalBlock"]:has(> .element-container > [data-testid="stMarkdownContainer"] .btn-pair-marker) > .element-container:last-child button[kind="primary"] {
    background-color: #7B2FBE !important;
    border-color: #7B2FBE !important;
}
[data-testid="stVerticalBlock"]:has(> .element-container > [data-testid="stMarkdownContainer"] .btn-pair-marker) > .element-container:last-child button[kind="primary"]:hover {
    background-color: #6322A3 !important;
    border-color: #6322A3 !important;
}
</style>
""", unsafe_allow_html=True)

# --- UI 세션 상태 ---
if 'prev_summary' not in st.session_state:
    st.session_state.prev_summary = {}
if 'prev_final_contributions' not in st.session_state:
    st.session_state.prev_final_contributions = []
if 'prev_episodes_run' not in st.session_state:
    st.session_state.prev_episodes_run = 0
if 'stock_trial_history' not in st.session_state:
    st.session_state.stock_trial_history = {}  # key: f"{m_name}_{stock_name}" → list of trial dicts
if 'fallback_params' not in st.session_state:
    st.session_state.fallback_params = {}       # 마지막 "All 적용" 시 스냅샷
if 'stock_use_fallback' not in st.session_state:
    st.session_state.stock_use_fallback = None  # "ALL" 이면 전체 fallback 활성
if 'stocks_reverted' not in st.session_state:
    st.session_state.stocks_reverted = set()    # Run Evaluation 클릭으로 되돌아온 종목
if 'sim_result' not in st.session_state:
    st.session_state.sim_result = {}            # key: hist_key → best params dict

# ==========================================
# 1. 실시간 시스템 상태 계기판
# ==========================================
def update_gauge(episodes_run, placeholder):
    max_load = 6000
    load_pct = min((episodes_run / max_load) * 100, 100)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=load_pct,
        number={'suffix': "%", 'valueformat': ".1f", 'font': {'weight': 'bold', 'size': 40}},
        title={'text': "Real-time Load", 'font': {'weight': 'bold', 'size': 20}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2196f3"},
            'steps': [
                {'range': [0, 50], 'color': "#333"},
                {'range': [50, 80], 'color': "#ff9800"},
                {'range': [80, 100], 'color': "#ff4b4b"}
            ]
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    # [핵심 수정] placeholder.plotly_chart() 대신 with 문으로 컨텍스트 교체:
    # st.empty() 슬롯을 with 구문으로 사용해야 기존 요소를 올바르게 교체합니다.
    with placeholder:
        st.plotly_chart(fig_gauge, use_container_width=True)

st.sidebar.markdown("### System Status")
gauge_placeholder = st.sidebar.empty()
# Gauge is rendered once at end of script to avoid DuplicateElementId

st.sidebar.markdown("---")

# ── All 적용 버튼 (파라미터 창 위에 배치) ──
_fb_active = st.session_state.stock_use_fallback == "ALL"
_fb_label  = "✅ Fallback 적용 중" if _fb_active else "🔁 All 적용"
apply_all_clicked = st.sidebar.button(
    _fb_label, key="sidebar_apply_all", type="primary",
    help="아래 설정값을 모든 멤버·종목의 시뮬레이션에 일괄 적용합니다\n"
         "(각 종목 Parameters 위젯 값은 변경되지 않습니다)"
)

with st.sidebar.expander("Fallback Parameters", expanded=False):
    st.markdown("<small><b>System Parameters</b></small>", unsafe_allow_html=True)
    global_episodes   = st.slider("Trading Days",       10,    500,  100,        key="fb_epi")
    global_frame      = st.slider("Frame Speed (sec)",  0.01,  2.0,  0.05,
                                  step=0.01, format="%.2f",    key="fb_frame")
    global_seed       = st.number_input("Base Seed",    value=2026,  step=1,     key="fb_seed")
    global_auto_runs  = st.number_input("Auto Run Count", min_value=1, value=1,
                                        step=1,                key="fb_auto")
    global_active_agents = st.multiselect(
        "Active Agents",
        options=["Vanilla RL", "STATIC RL"],
        default=["Vanilla RL", "STATIC RL"],
        key="fb_active_agents",
        help="체크 해제된 에이전트는 연산 없이 0% 수평선으로 표시"
    )
    st.markdown(
        "<small><b>RL Hyperparameters &nbsp;"
        "<span style='color:#4a90d9;'>STATIC RL</span>: α / γ / ε(S) &nbsp;|&nbsp; "
        "<span style='color:#e05050;'>Vanilla RL</span>: ε(V)</b></small>",
        unsafe_allow_html=True
    )
    global_lr         = st.slider("Learning Rate (α)",  0.001, 0.1,  0.01,
                                  step=0.001, format="%.3f",   key="fb_lr")
    global_gamma      = st.slider("Discount Factor (γ)",0.1,   0.99, 0.98,       key="fb_gamma")
    global_epsilon    = st.slider("STATIC ε",           0.01,  0.5,  0.10,       key="fb_eps",
                                  help="STATIC RL 탐험율")
    global_v_epsilon  = st.slider("Vanilla ε",          0.01,  0.5,  0.10,       key="fb_v_eps",
                                  help="Vanilla RL 탐험율 (STATIC과 독립적으로 조정)")

# 버튼 클릭 시 현재 슬라이더 값 스냅샷 저장 (슬라이더가 위에서 이미 렌더됨)
if apply_all_clicked:
    st.session_state.fallback_params = {
        "episodes":      global_episodes,
        "frame_speed":   global_frame,
        "seed":          int(global_seed),
        "auto_runs":     int(global_auto_runs),
        "active_agents": global_active_agents,
        "lr":            global_lr,
        "gamma":         global_gamma,
        "epsilon":       global_epsilon,
        "v_epsilon":     global_v_epsilon,
    }
    st.session_state.stock_use_fallback = "ALL"
    st.session_state.stocks_reverted    = set()

with st.container():
    st.markdown('<span class="sticky-header-marker"></span>', unsafe_allow_html=True)
    st.markdown('<h1 class="sticky-main-title">Chainers Master Fund: Performance Monitoring Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="sticky-divider">', unsafe_allow_html=True)
    st.markdown('<h2 class="sticky-sub-title">Master Fund Portfolio Report</h2>', unsafe_allow_html=True)
    master_progress_placeholder = st.empty()
    summary_placeholder = st.empty()
    st.markdown('<hr class="sticky-divider">', unsafe_allow_html=True)

# ==========================================
# 2. 통합 대시보드 (Bold & Red Font & Alpha 비교)
# ==========================================
def draw_top_dashboard(final_contribs, container, is_updating=False):
    df_contrib = pd.DataFrame(final_contribs)
    if df_contrib.empty:
        return {}

    df_contrib = df_contrib.sort_values(by="Member").reset_index(drop=True)
    distinct_colors = px.colors.qualitative.Plotly
    df_contrib['Unique_Color'] = [distinct_colors[i % len(distinct_colors)] for i in range(len(df_contrib))]

    total_fund_capital = df_contrib['Final_Capital'].sum()

    # (1) 도넛 그래프
    fig_donut = go.Figure(go.Pie(
        labels=df_contrib['Member'], values=df_contrib['Final_Capital'], hole=0.6,
        marker=dict(colors=df_contrib['Unique_Color']), textinfo="percent",
        texttemplate="<b>%{percent}</b><br><b>%{value:.2f}$</b>",
        textfont=dict(weight='bold'), sort=False
    ))
    title_text = "<b>Master Fund Contribution (STATIC)</b>"
    if is_updating:
        title_text += " <span style='color:#ff9800;'>(Updating...)</span>"

    fig_donut.update_layout(
        title=title_text, height=350, margin=dict(l=0, r=0, t=40, b=0),
        annotations=[dict(
            text=f"Total Capital<br><b>{total_fund_capital:.2f} $</b>",
            x=0.5, y=0.5, font_size=18, showarrow=False
        )],
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=-0.4, traceorder="normal")
    )

    # (2) 수익 바 차트 (Vanilla vs STATIC 비교)
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Bar(
        x=df_contrib['Member'], y=df_contrib['Vanilla_Profit'],
        name="Vanilla RL", marker_color="#ff4b4b", opacity=0.7,
        text=df_contrib['Vanilla_Profit'].apply(lambda x: f"<b>{x:.2f}$</b>"), textposition='outside'
    ))
    fig_profit.add_trace(go.Bar(
        x=df_contrib['Member'], y=df_contrib['Profit_Dollar'],
        name="STATIC RL", marker_color="#2196f3",
        text=df_contrib['Profit_Dollar'].apply(lambda x: f"<b>{x:.2f}$</b>"), textposition='outside'
    ))
    fig_profit.update_layout(
        title="<b>Profit Comparison ($): Vanilla vs STATIC</b>", barmode='group',
        height=350, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # (3) 성과 테이블
    table_data = []
    current_summary = {}

    for _, row in df_contrib.iterrows():
        m_name, c_ret, v_ret = row['Member'], row['Avg_Return'], row['Vanilla_Return']
        current_summary[m_name] = {'return': c_ret}
        delta = c_ret - v_ret
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        table_data.append({
            "Member": m_name, "Persona": row['CTPT_Code'], "Capital ($)": f"{row['Final_Capital']:.2f}$",
            "STATIC (%)": f"{c_ret:.2f}", "Vanilla (%)": f"{v_ret:.2f}",
            "Alpha (Gap)": f"{delta_str}", "STATIC MDD": f"{row['Avg_MDD']:.2f}%"
        })

    def color_negative_red(val):
        if isinstance(val, str) and val.strip().startswith('-'):
            return 'color: #FF4B4B; font-weight: bold;'
        return ''

    styled_table = pd.DataFrame(table_data).style.map(color_negative_red)

    # [핵심 수정] is_updating 값에 따라 고유 key suffix를 부여해 DuplicateElementId 방지
    key_suffix = "upd" if is_updating else "fin"

    with container:
        col1, col2, col3 = st.columns([1, 1.2, 1.3])
        with col1:
            st.plotly_chart(fig_donut, use_container_width=True, key=f"top_donut_{key_suffix}")
        with col2:
            st.plotly_chart(fig_profit, use_container_width=True, key=f"top_profit_{key_suffix}")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Portfolio Alpha Strategy Report")
            st.dataframe(styled_table, use_container_width=True, hide_index=True)

    return current_summary

# ==========================================
# 3. 시뮬레이션 및 차트 생성
# ==========================================
def _make_cumulative_fig(stock_name, df, v_trace, s_trace, real_ret_trace):
    """구버전 'S&P 500 Performance' fig_main 스타일: Cumulative Return Comparison"""
    height = 400
    title_size = 18
    axis_size = 14
    legend_size = 12

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=v_trace, mode='lines', name='<b>Vanilla RL</b>',
        line=dict(color='#e05050', width=2, dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=s_trace, mode='lines', name='<b>STATIC RL</b>',
        line=dict(color='#4a90d9', width=2.5, dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=real_ret_trace, mode='lines', name='<b>Market</b>',
        line=dict(color='#2ecc71', width=1.8, dash='dot')
    ))
    fig.update_layout(
        title=dict(text=f"<b>Cumulative Return Comparison ({stock_name})</b>", font=dict(size=title_size)),
        xaxis=dict(title=dict(text="<b>Trading Days</b>", font=dict(size=axis_size)), showgrid=True),
        yaxis=dict(title=dict(text="<b>Total Cumulative Return (%)</b>", font=dict(size=axis_size)), showgrid=True),
        legend=dict(font=dict(size=legend_size), x=0.01, y=0.99,
                    bgcolor='rgba(128,128,128,0.15)', bordercolor='rgba(128,128,128,0.3)', borderwidth=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=height, margin=dict(t=50, b=50, l=60, r=30)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig



def _make_trend_fig(df_h):
    """구버전 fig_trend: Trial-by-Trial Return Progression & Stability"""
    v_mean = df_h['Vanilla Final (%)'].mean()
    v_max  = df_h['Vanilla Final (%)'].max()
    v_min  = df_h['Vanilla Final (%)'].min()
    s_mean = df_h['STATIC Final (%)'].mean()
    s_max  = df_h['STATIC Final (%)'].max()
    s_min  = df_h['STATIC Final (%)'].min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Vanilla Final (%)'],
        mode='lines+markers', name='<b>Vanilla Return</b>',
        line=dict(color='#e05050', width=2),
        marker=dict(size=7, symbol='circle', color='#e05050')))
    fig.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['STATIC Final (%)'],
        mode='lines+markers', name='<b>STATIC Return (Ours)</b>',
        line=dict(color='#4a90d9', width=2),
        marker=dict(size=7, symbol='square', color='#4a90d9')))

    for y, dash, color, label, pos in [
        (v_mean, "solid",  "#e05050", "Vanilla Mean", "top right"),
        (v_max,  "dot",    "#e05050", "Vanilla Max",  "top right"),
        (v_min,  "dot",    "#e05050", "Vanilla Min",  "bottom right"),
        (s_mean, "solid",  "#4a90d9", "STATIC Mean",  "top left"),
        (s_max,  "dot",    "#4a90d9", "STATIC Max",   "top left"),
        (s_min,  "dot",    "#4a90d9", "STATIC Min",   "bottom left"),
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position=pos)

    fig.update_layout(
        title=dict(text="<b>Trial-by-Trial Return Progression & Stability</b>",
                   font=dict(size=20, family="Arial Black")),
        xaxis=dict(title="<b>Trial Number</b>", tickmode='linear', dtick=1),
        yaxis=dict(title="<b>Final Return (%)</b>"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=320, margin=dict(t=45, b=25, l=40, r=80)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def _make_trial_box_fig(df_h):
    """구버전 fig_box: Return Distribution across Trials"""
    v_mean     = df_h['Vanilla Final (%)'].mean()
    s_mean     = df_h['STATIC Final (%)'].mean()
    med_v      = df_h['Vanilla Final (%)'].median()
    med_s      = df_h['STATIC Final (%)'].median()
    avg_market = df_h['Market Final (%)'].mean()

    fig = go.Figure()
    fig.add_trace(go.Box(y=df_h['Vanilla Final (%)'], x0=1.0,
        name='<b>Vanilla RL</b>', line=dict(color='#e05050', width=3),
        fillcolor='rgba(224,80,80,0.05)', boxmean=True, width=0.5))
    fig.add_trace(go.Box(y=df_h['STATIC Final (%)'], x0=2.25,
        name='<b>STATIC RL (Ours)</b>', line=dict(color='#4a90d9', width=3),
        fillcolor='rgba(74,144,217,0.05)', boxmean=True, width=0.5))

    fig.add_annotation(x=0.75, y=v_mean, text=f"<b>Mean: {v_mean:.2f}%</b>",
        showarrow=False, xshift=-4, yshift=8, xanchor='right',
        font=dict(color='#e05050', size=13, family="Arial Black"))
    fig.add_annotation(x=0.75, y=med_v, text=f"<b>Median: {med_v:.2f}%</b>",
        showarrow=False, xshift=-4, yshift=-8, xanchor='right',
        font=dict(color='#e05050', size=13, family="Arial Black"))
    fig.add_annotation(x=2.5, y=med_s, text=f"<b>Median: {med_s:.2f}%</b>",
        showarrow=False, xshift=4, yshift=8, xanchor='left',
        font=dict(color='#4a90d9', size=13, family="Arial Black"))
    fig.add_annotation(x=2.5, y=s_mean, text=f"<b>Mean: {s_mean:.2f}%</b>",
        showarrow=False, xshift=4, yshift=-8, xanchor='left',
        font=dict(color='#4a90d9', size=13, family="Arial Black"))

    fig.add_hline(y=avg_market, line_width=2.5, line_dash="dot", line_color="green")
    fig.add_annotation(x=1.625, xref="x", y=avg_market,
        text=f"<b>Market (Buy&Hold)<br>{avg_market:.2f}%</b>",
        showarrow=False, yshift=18, xanchor='center', align='center',
        font=dict(color="green", size=13, family="Arial Black"), bgcolor="rgba(0,0,0,0)")

    fig.update_layout(
        title=dict(text="<b>Return Distribution across Trials</b>",
                   font=dict(size=22, family="Arial Black")),
        yaxis=dict(title=dict(text="<b>Final Return (%)</b>", font=dict(size=18, family="Arial Black"))),
        xaxis=dict(
            title=dict(text="<b>Performance Metrics</b>", font=dict(size=18, family="Arial Black")),
            tickmode='array', tickvals=[1.0, 2.25],
            ticktext=['<b>Vanilla RL</b>', '<b>STATIC RL (Ours)</b>'], range=[0.3, 2.9]
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=510, margin=dict(t=55, b=50, l=50, r=50)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def get_rl_data(ticker, lr, gamma, epsilon, episodes, seed, v_epsilon=None, fee_rate=0.0):
    """시뮬레이션을 1회만 실행하여 원시 데이터 + 일별 행동 로그를 반환.
    v_epsilon: Vanilla RL 전용 탐험율. None이면 epsilon과 동일하게 사용.
    fee_rate: 왕복 거래 수수료율 (CASH→BUY 진입 시 1회 부과)."""
    df_full = fetch_stock_data(ticker, period="2y")
    if df_full.empty or len(df_full) < 10:
        return None, None, None, None, 0.0, [], []
    df = df_full.tail(episodes).copy()
    real_ret_trace = (df['Close'] / df['Close'].iloc[0] - 1) * 100
    _v_eps = v_epsilon if v_epsilon is not None else epsilon
    v_trace, v_log = run_rl_simulation_with_log(df, lr, gamma, _v_eps, episodes=episodes, use_static=False, seed=seed, fee_rate=fee_rate)
    s_trace, s_log = run_rl_simulation_with_log(df, lr, gamma, epsilon,  episodes=episodes, use_static=True,  seed=seed, fee_rate=fee_rate)
    s_mdd = calculate_mdd(s_trace)
    return df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log

# --- 이전 결과 표시 (로딩 중 사라짐 방지) ---
if st.session_state.prev_final_contributions:
    draw_top_dashboard(st.session_state.prev_final_contributions, summary_placeholder, is_updating=True)

# --- 모듈 로드 ---
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except Exception as e:
            st.sidebar.warning(f"⚠️ {item} 로드 실패: {e}")

sorted_modules = sorted(team_modules, key=lambda m: getattr(m, "MEMBER_NAME", m.__name__))
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()}

total_charts = sum(
    len(st.session_state.get(
        f"ms_{getattr(m, 'MEMBER_NAME', m.__name__)}",
        [all_stock_names[idx] for idx in getattr(m, 'TARGET_INDICES', [])]
    ))
    for m in sorted_modules
)

final_contributions = []
total_episodes_run = 0
rendered_count = 0

# [수정] master_pbar를 None으로 초기화해 루프 내 undefined 변수 오류 방지
master_pbar = None
if total_charts > 0:
    master_pbar = master_progress_placeholder.progress(0.0, text="Analyzing Agents...")

st.markdown("### Portfolio Managers (Independent RL Labs)")

for m_config in sorted_modules:
    m_name = getattr(m_config, "MEMBER_NAME", "Unknown")
    with st.container():
        st.subheader(f"{m_name}")
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        ctpt_code, ctpt_desc, ctpt_color = calculate_ctpt_and_color(
            default_p.get("lr", global_lr),
            default_p.get("gamma", global_gamma),
            default_p.get("epsilon", global_epsilon)
        )
        _all_personas = [
            ("PSR", "#607d8b", "보수형"),
            ("PSV", "#ff9800", "탐색형"),
            ("PLR", "#3f51b5", "신중한 장기형"),
            ("PLV", "#e91e63", "유연한 장기형"),
            ("ASR", "#f44336", "단기 민첩형"),
            ("ASV", "#ffc107", "단기 모험형"),
            ("ALR", "#4caf50", "안정적 성장형"),
            ("ALV", "#2196f3", "적응형 모험가"),
        ]
        _badges = ""
        for _code, _color, _desc in _all_personas:
            if _code == ctpt_code:
                _badges += (
                    f"<span style='background:{_color}; color:#fff; border-radius:5px; "
                    f"padding:3px 11px; font-size:0.85em; font-weight:bold; "
                    f"box-shadow:0 0 8px {_color}99; margin:2px;'>"
                    f"{_code}&nbsp;{_desc}</span>"
                )
            else:
                _badges += (
                    f"<span style='background:{_color}18; color:{_color}; "
                    f"border:1px solid {_color}55; border-radius:5px; "
                    f"padding:3px 8px; font-size:0.78em; opacity:0.55; margin:2px;'>"
                    f"{_code}</span>"
                )
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:3px; flex-wrap:wrap; margin-bottom:2px;'>"
            f"<small style='margin-right:4px;'><b>Persona</b></small>{_badges}</div>",
            unsafe_allow_html=True
        )

        selected_stock_names = st.multiselect(
            f"Stocks - {m_name}",
            options=list(all_stock_names.values()),
            default=[all_stock_names[idx] for idx in getattr(m_config, 'TARGET_INDICES', [])],
            key=f"ms_{m_name}"
        )

        mem_s_rets, mem_v_rets, mem_mdds = [], [], []
        if selected_stock_names:
            for stock_name in selected_stock_names:
                ticker    = get_ticker_by_name(stock_name)
                fee_info  = get_fee_info(ticker)
                fee_rate  = fee_info["buy"] + fee_info["sell"]
                stock_idx = name_to_index.get(stock_name)
                p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                hist_key = f"{m_name}_{stock_name}"

                # ── Simulation pending: 슬라이더 렌더링 전에 키 사전 설정 ──
                _sim_pend_key = f"sim_pending_{hist_key}"
                if _sim_pend_key in st.session_state:
                    _pend = st.session_state.pop(_sim_pend_key)
                    st.session_state[f"lr_{m_name}_{stock_name}"]    = _pend["lr"]
                    st.session_state[f"gamma_{m_name}_{stock_name}"] = _pend["gamma"]
                    st.session_state[f"eps_{m_name}_{stock_name}"]   = _pend["epsilon"]
                    st.session_state[f"v_eps_{m_name}_{stock_name}"] = _pend["v_epsilon"]

                # ── 파라미터: 접힌 expander – 2행 구조 ──
                with st.expander(f"⚙️ {stock_name} Parameters", expanded=False):
                    st.caption(f"💸 거래 수수료 — {fee_info['label']}")
                    # ─ 행 1: System Parameters ─
                    st.markdown("<small><b>System Parameters</b></small>", unsafe_allow_html=True)
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    with sc1:
                        l_epi = st.slider(
                            "Trading Days", 10, 500,
                            int(p_settings.get("episodes", global_episodes)),
                            key=f"epi_{m_name}_{stock_name}"
                        )
                    with sc2:
                        l_frame_speed = st.slider(
                            "Frame Speed (sec)", 0.01, 2.0,
                            0.05, step=0.01, format="%.2f",
                            key=f"fspd_{m_name}_{stock_name}",
                            help="시뮬레이션 재생 프레임 간격"
                        )
                    with sc3:
                        l_seed = st.number_input(
                            "Base Seed",
                            value=int(p_settings.get("seed", global_seed)),
                            step=1, key=f"seed_{m_name}_{stock_name}"
                        )
                    with sc4:
                        l_auto_runs = st.number_input(
                            "Auto Run Count", min_value=1, value=1, step=1,
                            key=f"autoruns_{m_name}_{stock_name}",
                            help="Run Evaluation 클릭 시 자동 반복 횟수"
                        )
                    with sc5:
                        l_active_agents = st.multiselect(
                            "Active Agents",
                            options=["Vanilla RL", "STATIC RL"],
                            default=["Vanilla RL", "STATIC RL"],
                            key=f"active_{m_name}_{stock_name}",
                            help="체크 해제된 에이전트는 연산 없이 0% 수평선으로 표시"
                        )
                    # ─ 행 2: RL Hyperparameters ─
                    st.markdown(
                        "<small><b>RL Hyperparameters &nbsp;"
                        "<span style='color:#4a90d9;'>STATIC RL</span>: α / γ / ε(S) &nbsp;|&nbsp; "
                        "<span style='color:#e05050;'>Vanilla RL</span>: ε(V)</b></small>",
                        unsafe_allow_html=True
                    )
                    hc1, hc2, hc3, hc4 = st.columns(4)
                    with hc1:
                        l_lr = st.slider(
                            "Learning Rate (α)", 0.001, 0.1,
                            float(p_settings.get("lr", global_lr)),
                            step=0.001, format="%.3f", key=f"lr_{m_name}_{stock_name}"
                        )
                    with hc2:
                        l_gamma = st.slider(
                            "Discount Factor (γ)", 0.1, 0.99,
                            float(p_settings.get("gamma", global_gamma)),
                            key=f"gamma_{m_name}_{stock_name}"
                        )
                    with hc3:
                        l_epsilon = st.slider(
                            "STATIC ε", 0.01, 0.5,
                            float(p_settings.get("epsilon", global_epsilon)),
                            key=f"eps_{m_name}_{stock_name}",
                            help="STATIC RL 탐험율"
                        )
                    with hc4:
                        l_v_epsilon = st.slider(
                            "Vanilla ε", 0.01, 0.5,
                            float(p_settings.get("v_epsilon", global_epsilon)),
                            key=f"v_eps_{m_name}_{stock_name}",
                            help="Vanilla RL 탐험율 (STATIC과 독립적으로 조정)"
                        )

                # ── Run Evaluation / Simulation 버튼 + 진행률 ──
                btn_col, run_prog_col = st.columns([2, 3])
                with btn_col:
                    st.markdown('<span class="btn-pair-marker"></span>', unsafe_allow_html=True)
                    run_clicked = st.button(
                        "▶ Run Evaluation",
                        key=f"run_btn_{m_name}_{stock_name}",
                        type="primary",
                    )
                    sim_clicked = st.button(
                        "Simulation",
                        key=f"sim_btn_{m_name}_{stock_name}",
                        type="primary",
                    )
                run_prog_slot = run_prog_col.empty()

                # ── 이전 Simulation 결과 배너 ──
                if hist_key in st.session_state.sim_result:
                    sr = st.session_state.sim_result[hist_key]
                    _status = "✅ 목표 달성" if sr.get("found") else "⚠️ 최선값"
                    st.caption(
                        f"🔍 최근 Simulation — {_status}  |  "
                        f"LR={sr['lr']:.4f}  γ={sr['gamma']:.4f}  "
                        f"ε(S)={sr['epsilon']:.4f}  ε(V)={sr['v_epsilon']:.4f}  |  "
                        f"STATIC {sr['s_final']:+.2f}%  Vanilla {sr['v_final']:+.2f}%  "
                        f"Gap {sr['gap']:+.2f}%"
                    )

                # ── Simulation 후 자동 Run Evaluation 트리거 ──
                _auto_run_key = f"auto_run_{hist_key}"
                if st.session_state.get(_auto_run_key, False):
                    st.session_state[_auto_run_key] = False
                    run_clicked = True

                if run_clicked:
                    # 종목별 파라미터를 다시 적용: fallback 목록에서 이 종목 제거
                    st.session_state.stocks_reverted.add(hist_key)
                    trials = st.session_state.stock_trial_history.setdefault(hist_key, [])
                    n_runs = int(l_auto_runs)
                    for run_i in range(n_runs):
                        trial_seed = int(l_seed) + len(trials) + run_i
                        run_prog_slot.progress(
                            run_i / n_runs,
                            text=f"Running trial {run_i + 1} / {n_runs}  (seed={trial_seed})"
                        )
                        _, vt, st_t, mkt, _, _, _ = get_rl_data(
                            ticker, l_lr, l_gamma, l_epsilon, l_epi, trial_seed,
                            v_epsilon=l_v_epsilon, fee_rate=fee_rate
                        )
                        if vt is not None:
                            trials.append({
                                "Trial": len(trials) + 1,
                                "Seed": trial_seed,
                                "Vanilla Final (%)": float(vt[-1]),
                                "STATIC Final (%)":  float(st_t[-1]),
                                "Market Final (%)":  float(mkt.iloc[-1]),
                            })
                    run_prog_slot.success(f"완료: {n_runs}회 Trial 누적")
                    st.rerun()

                # ── Simulation: STATIC > Vanilla 조건 파라미터 탐색 ──
                if sim_clicked:
                    n_iters = max(20, int(l_auto_runs) * 8)
                    phase1  = int(n_iters * 0.4)
                    phase2  = int(n_iters * 0.8)
                    param_bounds = {
                        "lr":        (0.001, 0.1),
                        "gamma":     (0.5,   0.99),
                        "epsilon":   (0.01,  0.5),
                        "v_epsilon": (0.01,  0.5),
                    }
                    best = {
                        "lr": l_lr, "gamma": l_gamma,
                        "epsilon": l_epsilon, "v_epsilon": l_v_epsilon,
                        "gap": -999.0, "s_final": 0.0, "v_final": 0.0
                    }
                    gap_history = []
                    param_hist  = {k: [] for k in param_bounds}
                    sim_display = st.empty()

                    for _i in range(n_iters):
                        if _i < phase1:
                            phase_name, step = "🔴 Exploring", 1.0
                        elif _i < phase2:
                            phase_name, step = "🟡 Narrowing", 0.25
                        else:
                            phase_name, step = "🟢 Converging", 0.05

                        candidate = {}
                        for _k, (_lo, _hi) in param_bounds.items():
                            _rng  = (_hi - _lo) * step
                            _base = best[_k] if _i > 0 else _lo + (_hi - _lo) * random.random()
                            candidate[_k] = float(np.clip(
                                _base + random.uniform(-_rng / 2, _rng / 2), _lo, _hi
                            ))

                        _seed = int(l_seed) + _i
                        _, _vt, _st_t, _, _, _, _ = get_rl_data(
                            ticker,
                            candidate["lr"], candidate["gamma"], candidate["epsilon"],
                            int(l_epi), _seed, v_epsilon=candidate["v_epsilon"],
                            fee_rate=fee_rate
                        )
                        if _vt is not None and _st_t is not None:
                            _gap = float(_st_t[-1]) - float(_vt[-1])
                            candidate["gap"]     = _gap
                            candidate["s_final"] = float(_st_t[-1])
                            candidate["v_final"] = float(_vt[-1])
                            if _gap > best["gap"]:
                                best = candidate.copy()
                        gap_history.append(best["gap"])
                        for _k in param_bounds:
                            param_hist[_k].append(candidate[_k])

                        # 실시간 디스플레이
                        with sim_display.container(border=True):
                            _prog     = (_i + 1) / n_iters
                            _goal_txt = " ✅" if best["gap"] >= 5.0 else ""
                            st.progress(_prog,
                                text=f"{phase_name}  {_i+1}/{n_iters}  |  "
                                     f"STATIC {best['s_final']:+.2f}%  "
                                     f"Vanilla {best['v_final']:+.2f}%  "
                                     f"Gap {best['gap']:+.1f}%{_goal_txt}")
                            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
                            _prev_lr = param_hist["lr"][-2]        if len(param_hist["lr"]) > 1        else candidate["lr"]
                            _prev_g  = param_hist["gamma"][-2]     if len(param_hist["gamma"]) > 1     else candidate["gamma"]
                            _prev_e  = param_hist["epsilon"][-2]   if len(param_hist["epsilon"]) > 1   else candidate["epsilon"]
                            _prev_ve = param_hist["v_epsilon"][-2] if len(param_hist["v_epsilon"]) > 1 else candidate["v_epsilon"]
                            _pc1.metric("α (LR)",    f'{candidate["lr"]:.4f}',        f'{candidate["lr"]        - _prev_lr:+.4f}')
                            _pc2.metric("γ",          f'{candidate["gamma"]:.4f}',    f'{candidate["gamma"]     - _prev_g:+.4f}')
                            _pc3.metric("ε STATIC",   f'{candidate["epsilon"]:.4f}',  f'{candidate["epsilon"]   - _prev_e:+.4f}')
                            _pc4.metric("ε Vanilla",  f'{candidate["v_epsilon"]:.4f}',f'{candidate["v_epsilon"] - _prev_ve:+.4f}')
                            if len(gap_history) > 1:
                                _fig_sim = go.Figure()
                                _fig_sim.add_trace(go.Scatter(
                                    x=list(range(1, len(gap_history) + 1)),
                                    y=gap_history,
                                    mode="lines",
                                    line=dict(color="#4a90d9", width=2),
                                ))
                                _fig_sim.add_hline(y=5.0, line_dash="dash",
                                                   line_color="#50c878",
                                                   annotation_text="목표 +5%")
                                _fig_sim.update_layout(
                                    height=150, margin=dict(l=0, r=0, t=10, b=0),
                                    xaxis_title="Iteration", yaxis_title="Gap (%)",
                                    showlegend=False,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)"
                                )
                                st.plotly_chart(_fig_sim, use_container_width=True,
                                                key=f"sim_chart_{m_name}_{stock_name}_{_i}")

                    # 완료: sim_pending으로 저장 후 rerun (슬라이더 렌더링 전 적용됨)
                    best["found"] = best["gap"] >= 5.0
                    st.session_state.sim_result[hist_key] = best
                    st.session_state[f"sim_pending_{hist_key}"] = {
                        "lr":        best["lr"],
                        "gamma":     best["gamma"],
                        "epsilon":   best["epsilon"],
                        "v_epsilon": best["v_epsilon"],
                    }
                    st.session_state.stocks_reverted.add(hist_key)
                    st.session_state[f"auto_run_{hist_key}"] = True
                    sim_display.empty()
                    st.rerun()

                # ── 유효 파라미터 결정: fallback 활성 여부 ──
                _use_fb = (
                    st.session_state.stock_use_fallback == "ALL"
                    and hist_key not in st.session_state.stocks_reverted
                )
                if _use_fb:
                    fp = st.session_state.fallback_params
                    eff_lr, eff_gamma, eff_eps = fp["lr"],   fp["gamma"],   fp["epsilon"]
                    eff_epi, eff_seed          = fp["episodes"], fp["seed"]
                    eff_v_eps          = fp.get("v_epsilon", fp["epsilon"])
                    eff_active_agents  = fp.get("active_agents", ["Vanilla RL", "STATIC RL"])
                    st.info(
                        f"Fallback 파라미터 적용 중 "
                        f"(LR={eff_lr:.3f} γ={eff_gamma:.2f} ε(S)={eff_eps:.2f} ε(V)={eff_v_eps:.2f} "
                        f"Days={eff_epi} Seed={eff_seed} "
                        f"Agents={', '.join(eff_active_agents) if eff_active_agents else '없음'})",
                        icon="ℹ️"
                    )
                else:
                    eff_lr, eff_gamma, eff_eps, eff_epi, eff_seed = (
                        l_lr, l_gamma, l_epsilon, l_epi, l_seed
                    )
                    eff_v_eps         = l_v_epsilon
                    eff_active_agents = l_active_agents

                # ── 시뮬레이션 실행 (유효 파라미터 기준) ──
                with st.spinner(f"Processing {stock_name}..."):
                    df_stock, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log = get_rl_data(
                        ticker, eff_lr, eff_gamma, eff_eps, eff_epi, eff_seed,
                        v_epsilon=eff_v_eps, fee_rate=fee_rate
                    )

                if df_stock is None:
                    st.warning(f"데이터를 불러올 수 없습니다: {stock_name}")
                    continue

                # ── Active Agents 적용: 비활성 에이전트 → 0% 수평선 ──
                if "Vanilla RL" not in eff_active_agents:
                    v_trace = np.zeros(len(df_stock))
                    v_log   = []
                if "STATIC RL" not in eff_active_agents:
                    s_trace = np.zeros(len(df_stock))
                    s_log   = []
                    s_mdd   = 0.0

                s_final      = float(s_trace[-1])
                v_final      = float(v_trace[-1])
                market_final = float(real_ret_trace.iloc[-1])
                # 마지막 스텝 일별 수익 (delta용)
                v_last_day = float(v_trace[-1] - v_trace[-2]) if len(v_trace) > 1 else 0.0
                s_last_day = float(s_trace[-1] - s_trace[-2]) if len(s_trace) > 1 else 0.0
                m_last_day = float(real_ret_trace.iloc[-1] - real_ret_trace.iloc[-2]) if len(real_ret_trace) > 1 else 0.0

                # ── 메인 2컬럼 레이아웃 ──
                col_left, col_right = st.columns([1, 1])

                # ══════════════════════════════════════════
                # 왼쪽: S&P 500 Performance 스타일
                # ══════════════════════════════════════════
                with col_left:
                    st.markdown(f"#### {stock_name} Performance")

                    # 누적 수익 차트 (구버전 fig_main 스타일)
                    fig_cum = _make_cumulative_fig(stock_name, df_stock, v_trace, s_trace, real_ret_trace)
                    st.plotly_chart(fig_cum, use_container_width=True, key=f"chart_cum_{m_name}_{stock_name}")

                    # 3 지표 카드 – 색상 커스터마이징 (HTML)
                    st.markdown(
                        "<p style='margin:6px 0 2px 0;font-size:12px;font-weight:700;"
                        "color:rgba(180,180,180,0.8);letter-spacing:0.05em;'>Final Cumulative Return</p>",
                        unsafe_allow_html=True
                    )
                    mc1, mc2, mc3 = st.columns(3)
                    def _metric_html(label, value_pct, delta_pct, color):
                        delta_sign = "▲" if delta_pct >= 0 else "▼"
                        delta_color = "#2ecc71" if delta_pct >= 0 else "#e05050"
                        return (
                            f"<div style='padding:8px 4px 4px 4px;'>"
                            f"<div style='font-size:12px;font-weight:700;color:{color};'>{label}</div>"
                            f"<div style='font-size:28px;font-weight:900;color:{color};line-height:1.2;'>"
                            f"{value_pct:.2f}%</div>"
                            f"<div style='font-size:13px;color:{delta_color};margin-top:2px;'>"
                            f"{delta_sign} {abs(delta_pct):.2f}%</div>"
                            f"</div>"
                        )
                    with mc1:
                        st.markdown(_metric_html("Vanilla RL", v_final, v_last_day, "#e05050"), unsafe_allow_html=True)
                    with mc2:
                        st.markdown(_metric_html("STATIC RL", s_final, s_last_day, "#4a90d9"), unsafe_allow_html=True)
                    with mc3:
                        st.markdown(_metric_html("Market (Buy&Hold)", market_final, m_last_day, "#2ecc71"), unsafe_allow_html=True)

                    # Agent Decision Analysis – Action Frequency(좌) + 테이블(우)
                    if v_log and s_log:
                        st.markdown("---")
                        st.markdown("#### Agent Decision Analysis")

                        df_v = pd.DataFrame(v_log).rename(columns={"Action": "Vanilla Action", "Daily_Return(%)": "Vanilla Return(%)"})
                        df_s = pd.DataFrame(s_log).rename(columns={"Action": "STATIC Action",  "Daily_Return(%)": "STATIC Return(%)"})
                        df_log = df_v.merge(df_s, on="Day").set_index("Day")

                        def _style_log(val):
                            if isinstance(val, (int, float)):
                                return 'color: #e05050; font-weight: bold;' if val < 0 else 'font-weight: bold;'
                            if val == "BUY":
                                return 'color: #4a90d9; font-weight: bold;'
                            if val == "CASH":
                                return 'color: #e05050; font-weight: bold;'
                            return 'font-weight: bold;'

                        styled_log = df_log.style.map(_style_log).format(
                            {"Vanilla Return(%)": "{:.2f}", "STATIC Return(%)": "{:.2f}"}
                        )

                        bar_col, tbl_col = st.columns([1, 1.4])
                        with bar_col:
                            action_counts = df_log["STATIC Action"].value_counts().reset_index()
                            action_counts.columns = ["Action", "Count"]
                            for _act in ["BUY", "CASH"]:
                                if _act not in action_counts["Action"].values:
                                    action_counts = pd.concat([action_counts,
                                        pd.DataFrame({"Action": [_act], "Count": [0]})], ignore_index=True)
                            action_counts = action_counts.sort_values("Action").reset_index(drop=True)
                            _bar_colors = {"BUY": "#4a90d9", "CASH": "#e05050"}
                            fig_bar = go.Figure()
                            for _, row in action_counts.iterrows():
                                fig_bar.add_trace(go.Bar(
                                    x=[row["Action"]], y=[row["Count"]],
                                    name=row["Action"],
                                    marker_color=_bar_colors.get(row["Action"], "#888"),
                                    width=0.35,
                                    text=[f"<b>{row['Count']}</b>"],
                                    textposition="outside",
                                    textfont=dict(size=14, color=_bar_colors.get(row["Action"], "#888"))
                                ))
                            fig_bar.update_layout(
                                title=dict(text="<b>STATIC: Action Frequency</b>", font=dict(size=13)),
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                height=250, showlegend=False,
                                margin=dict(t=45, b=25, l=30, r=10),
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=True, range=[0, action_counts["Count"].max() * 1.2])
                            )
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{m_name}_{stock_name}")
                        with tbl_col:
                            st.dataframe(styled_log, height=250, use_container_width=True)

                # ══════════════════════════════════════════
                # 오른쪽: Trial History Statistical Analysis 스타일
                # ══════════════════════════════════════════
                with col_right:
                    st.markdown("### Trial History: Statistical Analysis (Alpha Performance)")
                    trials = st.session_state.stock_trial_history.get(hist_key, [])

                    if not trials:
                        st.markdown("""
<div style='background:var(--secondary-background-color);padding:40px;border-radius:10px;
border:1px solid rgba(128,128,128,0.3);text-align:center;margin-top:20px;'>
<h4 style='color:rgba(128,128,128,0.6);margin-bottom:8px;'>Trial History 없음</h4>
<p style='color:rgba(128,128,128,0.4);font-size:14px;'>상단 <b>▶ Run Evaluation</b>을 클릭하여<br>Trial History를 누적하세요</p>
</div>""", unsafe_allow_html=True)
                    else:
                        df_h = pd.DataFrame(trials)
                        v_mean = df_h['Vanilla Final (%)'].mean()
                        s_mean = df_h['STATIC Final (%)'].mean()
                        v_std  = df_h['Vanilla Final (%)'].std(ddof=0) if len(df_h) > 1 else 0.0
                        s_std  = df_h['STATIC Final (%)'].std(ddof=0) if len(df_h) > 1 else 0.0
                        v_min, v_max = df_h['Vanilla Final (%)'].min(), df_h['Vanilla Final (%)'].max()
                        s_min, s_max = df_h['STATIC Final (%)'].min(), df_h['STATIC Final (%)'].max()
                        avg_mkt = df_h['Market Final (%)'].mean()

                        # Alpha 배너
                        st.success(
                            f"시장 평균 대비 **Alpha 기대치(Expected Value)**: "
                            f"STATIC **{s_mean - avg_mkt:.2f}%p** | Vanilla **{v_mean - avg_mkt:.2f}%p**"
                        )

                        # Trial-by-Trial 추이 차트
                        st.plotly_chart(_make_trend_fig(df_h), use_container_width=True,
                                        key=f"trend_{m_name}_{stock_name}")

                        # 박스 플롯 + 통계 카드 2열
                        box_col, stat_col = st.columns([1.1, 1])
                        with box_col:
                            st.plotly_chart(_make_trial_box_fig(df_h), use_container_width=True,
                                            key=f"tribox_{m_name}_{stock_name}")
                        with stat_col:
                            st.markdown(f"""
<div style='background:var(--secondary-background-color);padding:12px 14px;border-radius:10px;
border:1px solid rgba(128,128,128,0.3);'>
<h4 style='margin-top:0;margin-bottom:8px;font-weight:900;font-size:14px;'>통계 요약 (Expected &amp; Risk)</h4>
<div style='display:flex;gap:12px;'>
  <div style='flex:1;border-right:1px solid rgba(128,128,128,0.3);padding-right:10px;'>
    <ul style='font-size:12px;margin:0;padding-left:12px;line-height:1.9;list-style:none;'>
    <li><b style='color:#e05050;'>Vanilla 평균:</b> {v_mean:.2f}% (σ={v_std:.2f}%)</li>
    <li><b style='color:#e05050;'>Vanilla 범위:</b> {v_min:.2f}% ~ {v_max:.2f}%</li>
    </ul>
  </div>
  <div style='flex:1;padding-left:2px;'>
    <ul style='font-size:12px;margin:0;padding-left:12px;line-height:1.9;list-style:none;'>
    <li><b style='color:#4a90d9;'>STATIC 평균:</b> {s_mean:.2f}% (σ={s_std:.2f}%)</li>
    <li><b style='color:#4a90d9;'>STATIC 범위:</b> {s_min:.2f}% ~ {s_max:.2f}%</li>
    </ul>
  </div>
</div></div>""", unsafe_allow_html=True)

                            def _color_neg(val):
                                if isinstance(val, (int, float)) and val < 0:
                                    return 'color: #e05050; font-weight: bold;'
                                return 'font-weight: bold;'

                            st.dataframe(
                                df_h.set_index("Trial").style.map(_color_neg).format(
                                    {"Vanilla Final (%)": "{:.2f}", "STATIC Final (%)": "{:.2f}",
                                     "Market Final (%)": "{:.2f}", "Seed": "{:.0f}"}
                                ), height=320, use_container_width=True
                            )

                total_episodes_run += l_epi
                rendered_count += 1
                if master_pbar is not None:
                    pct = min(rendered_count / total_charts, 1.0)
                    master_pbar.progress(pct, text=f"Analyzing Agents... ({int(pct * 100)}%)")

                mem_s_rets.append(s_final)
                mem_v_rets.append(v_final)
                mem_mdds.append(s_mdd)

        if mem_s_rets:
            avg_s, avg_v = np.mean(mem_s_rets), np.mean(mem_v_rets)
            final_contributions.append({
                "Member": m_name,
                "Final_Capital": 1.0 * (1 + avg_s / 100),
                "Profit_Dollar": (1.0 * (1 + avg_s / 100)) - 1.0,
                "Vanilla_Profit": (1.0 * (1 + avg_v / 100)) - 1.0,
                "Avg_Return": avg_s,
                "Vanilla_Return": avg_v,
                "Avg_MDD": np.mean(mem_mdds),
                "CTPT_Code": ctpt_code
            })
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

if final_contributions:
    master_progress_placeholder.empty()
    # summary_placeholder.empty() 제거 — draw_top_dashboard 내부 with container:가 원자적으로 교체함
    current_summary = draw_top_dashboard(final_contributions, summary_placeholder)

    # 게이지를 최종 값으로 한 번만 업데이트 (루프 안에서는 호출하지 않음)
    st.session_state.prev_final_contributions = final_contributions
    st.session_state.prev_summary = current_summary
    st.session_state.prev_episodes_run = total_episodes_run

# Render gauge exactly once per script run
update_gauge(
    total_episodes_run if final_contributions else st.session_state.prev_episodes_run,
    gauge_placeholder
)
