import streamlit as st
import importlib
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation
from common.evaluator import calculate_ctpt_and_color, calculate_mdd

# 루트 경로 설정
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

st.set_page_config(page_title="Chainers Master Fund", layout="wide", initial_sidebar_state="collapsed")

# --- UI 세션 상태 ---
if 'prev_summary' not in st.session_state:
    st.session_state.prev_summary = {}
if 'prev_final_contributions' not in st.session_state:
    st.session_state.prev_final_contributions = []
if 'prev_episodes_run' not in st.session_state:
    st.session_state.prev_episodes_run = 0

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
update_gauge(st.session_state.prev_episodes_run, gauge_placeholder)

st.sidebar.markdown("---")
with st.sidebar.expander("Fallback Parameters", expanded=False):
    global_episodes = st.slider("Episodes", 10, 500, 100)
    global_seed = st.number_input("Random Seed", value=2026, step=1)
    global_lr = st.slider("Learning Rate (alpha)", 0.001, 0.1, 0.01, step=0.001)
    global_gamma = st.slider("Discount Factor (gamma)", 0.1, 0.99, 0.98)
    global_epsilon = st.slider("Exploration (epsilon)", 0.01, 0.5, 0.10)

st.title("Chainers Master Fund: Performance Monitoring Dashboard")
st.markdown("---")
st.markdown("## Master Fund Portfolio Report")

master_progress_placeholder = st.empty()
summary_placeholder = st.empty()
st.markdown("---")

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=v_trace, mode='lines+markers', name='<b>Vanilla RL</b>',
        line=dict(color='#e05050', width=2), marker=dict(symbol='circle-open', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=s_trace, mode='lines+markers', name='<b>RL with STATIC</b>',
        line=dict(color='#4a90d9', width=2), marker=dict(symbol='square-open', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=real_ret_trace, mode='lines+markers', name='<b>Market</b>',
        line=dict(color='green', width=2, dash='dot'), marker=dict(symbol='diamond-open', size=6)
    ))
    fig.update_layout(
        title=dict(text=f"<b>{stock_name} (Lookback: {len(df)} Days)</b>", font=dict(size=22)),
        xaxis=dict(title=dict(text="<b>Trading Days</b>", font=dict(size=16)), showgrid=True),
        yaxis=dict(title=dict(text="<b>Total Cumulative Return (%)</b>", font=dict(size=16)), showgrid=True),
        legend=dict(font=dict(size=14), x=0.01, y=0.99,
                    bgcolor='rgba(128,128,128,0.15)', bordercolor='rgba(128,128,128,0.3)', borderwidth=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=420, margin=dict(t=70, b=70, l=70, r=30)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def _make_stats_fig(stock_name, df, v_trace, s_trace, real_ret_trace):
    """구버전 'Trial History: Statistical Analysis' fig_box 스타일: Daily Return Distribution"""
    v_arr = np.array(v_trace)
    s_arr = np.array(s_trace)
    real_arr = np.array(real_ret_trace)

    if len(v_arr) < 2:
        return go.Figure()

    v_daily = np.diff(v_arr)
    s_daily = np.diff(s_arr)
    market_daily = np.diff(real_arr)

    v_mean, s_mean = float(np.mean(v_daily)), float(np.mean(s_daily))
    med_v, med_s = float(np.median(v_daily)), float(np.median(s_daily))
    avg_market = float(np.mean(market_daily))

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=v_daily, x0=1.0, name='<b>Vanilla RL</b>',
        line=dict(color='#e05050', width=3), fillcolor='rgba(224,80,80,0.05)',
        boxmean=True, width=0.5
    ))
    fig.add_trace(go.Box(
        y=s_daily, x0=2.25, name='<b>STATIC RL (Ours)</b>',
        line=dict(color='#4a90d9', width=3), fillcolor='rgba(74,144,217,0.05)',
        boxmean=True, width=0.5
    ))

    fig.add_annotation(x=0.75, y=v_mean, text=f"<b>Mean: {v_mean:.2f}%</b>",
                       showarrow=False, xshift=-4, yshift=8, xanchor='right',
                       font=dict(color='#e05050', size=12, family="Arial Black"))
    fig.add_annotation(x=0.75, y=med_v, text=f"<b>Median: {med_v:.2f}%</b>",
                       showarrow=False, xshift=-4, yshift=-8, xanchor='right',
                       font=dict(color='#e05050', size=12, family="Arial Black"))
    fig.add_annotation(x=2.5, y=med_s, text=f"<b>Median: {med_s:.2f}%</b>",
                       showarrow=False, xshift=4, yshift=8, xanchor='left',
                       font=dict(color='#4a90d9', size=12, family="Arial Black"))
    fig.add_annotation(x=2.5, y=s_mean, text=f"<b>Mean: {s_mean:.2f}%</b>",
                       showarrow=False, xshift=4, yshift=-8, xanchor='left',
                       font=dict(color='#4a90d9', size=12, family="Arial Black"))

    fig.add_hline(y=avg_market, line_width=2.5, line_dash="dot", line_color="green")
    fig.add_annotation(
        x=1.625, xref="x", y=avg_market,
        text=f"<b>Market Daily Avg<br>{avg_market:.2f}%</b>",
        showarrow=False, yshift=18, xanchor='center', align='center',
        font=dict(color="green", size=12, family="Arial Black"), bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        title=dict(text=f"<b>{stock_name} (Lookback: {len(df)} Days)</b>",
                   font=dict(size=22, family="Arial Black")),
        yaxis=dict(title=dict(text="<b>Daily Return (%)</b>", font=dict(size=16, family="Arial Black"))),
        xaxis=dict(
            title=dict(text="<b>Performance Metrics</b>", font=dict(size=16, family="Arial Black")),
            tickmode='array', tickvals=[1.0, 2.25],
            ticktext=['<b>Vanilla RL</b>', '<b>STATIC RL (Ours)</b>'],
            range=[0, 3.0]
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=420, margin=dict(t=80, b=80, l=60, r=60)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed, chart_type='cumulative'):
    df_full = fetch_stock_data(ticker, period="2y")
    if df_full.empty or len(df_full) < 10:
        return go.Figure(), 0.0, 0.0, 0.0

    # Trading Days 만큼 최근 데이터를 잘라 차트 범위로 사용
    df = df_full.tail(episodes).copy()
    real_ret_trace = (df['Close'] / df['Close'].iloc[0] - 1) * 100

    v_trace = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    s_trace = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)
    s_mdd = calculate_mdd(s_trace)

    if chart_type == 'stats':
        fig = _make_stats_fig(stock_name, df, v_trace, s_trace, real_ret_trace)
    else:
        fig = _make_cumulative_fig(stock_name, df, v_trace, s_trace, real_ret_trace)

    return fig, s_trace[-1], v_trace[-1], s_mdd

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
        st.markdown(
            f"**Persona:** <span style='color:{ctpt_color}; font-weight:bold;'>{ctpt_code}</span> ({ctpt_desc})",
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
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    stock_idx = name_to_index.get(stock_name)
                    p_settings = m_params.get(stock_idx, m_params.get("default", {}))

                    with st.expander(f"{stock_name} Parameters", expanded=True):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            l_epi = st.slider("Trading Days", 10, 500,
                                              int(p_settings.get("episodes", global_episodes)),
                                              key=f"epi_{m_name}_{stock_name}")
                            l_seed = st.number_input("Seed",
                                                     value=int(p_settings.get("seed", global_seed)),
                                                     step=1, key=f"seed_{m_name}_{stock_name}")
                        with sc2:
                            l_lr = st.slider("LR", 0.001, 0.1,
                                             float(p_settings.get("lr", global_lr)),
                                             step=0.001, format="%.3f", key=f"lr_{m_name}_{stock_name}")
                            l_gamma = st.slider("Gamma", 0.1, 0.99,
                                                float(p_settings.get("gamma", global_gamma)),
                                                key=f"gamma_{m_name}_{stock_name}")
                            l_epsilon = st.slider("Epsilon", 0.01, 0.5,
                                                  float(p_settings.get("epsilon", global_epsilon)),
                                                  key=f"eps_{m_name}_{stock_name}")

                    # 짝수 컬럼(왼쪽): Cumulative Return 스타일, 홀수 컬럼(오른쪽): Stats 스타일
                    chart_type = 'cumulative' if j % 2 == 0 else 'stats'
                    with st.spinner(f"📡 Processing {stock_name}..."):
                        fig, s_final, v_final, s_mdd = create_real_rl_chart(
                            stock_name, ticker, l_lr, l_gamma, l_epsilon, l_epi, l_seed,
                            chart_type=chart_type
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_name}_{stock_name}")

                    total_episodes_run += l_epi
                    rendered_count += 1

                    # [핵심 수정] update_gauge를 루프 안에서 호출하지 않음 (DuplicateElementId 방지)
                    # 진행 상황은 메인 영역의 progress bar로만 표시
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
    summary_placeholder.empty()

    # [핵심 수정] with summary_placeholder.container() 제거:
    # draw_top_dashboard 내부에서 이미 'with container:'를 사용하므로 중복 중첩 불필요
    current_summary = draw_top_dashboard(final_contributions, summary_placeholder)

    # 게이지를 최종 값으로 한 번만 업데이트 (루프 안에서는 호출하지 않음)
    update_gauge(total_episodes_run, gauge_placeholder)

    st.session_state.prev_final_contributions = final_contributions
    st.session_state.prev_summary = current_summary
    st.session_state.prev_episodes_run = total_episodes_run
