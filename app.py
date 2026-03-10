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
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Master Fund", layout="wide", initial_sidebar_state="collapsed")

# --- UI 세션 상태 유지 (깜빡임 방지용 캐시) ---
if 'prev_summary' not in st.session_state: st.session_state.prev_summary = {}
if 'prev_final_contributions' not in st.session_state: st.session_state.prev_final_contributions = []
if 'prev_episodes_run' not in st.session_state: st.session_state.prev_episodes_run = 0

# ==========================================
# 🚀 1. 실시간 시스템 상태 계기판 (에러 해결 버전)
# ==========================================
def update_gauge(episodes_run, placeholder):
    max_load = 6000 
    load_pct = min((episodes_run / max_load) * 100, 100)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=load_pct, 
        number={'suffix': "%", 'valueformat': ".1f", 'font': {'weight': 'bold'}},
        title={'text': "Real-time Load", 'font': {'weight': 'bold'}},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#2196f3"},
               'steps': [{'range': [0, 50], 'color': "#333"}, {'range': [50, 80], 'color': "#ff9800"}, {'range': [80, 100], 'color': "#ff4b4b"}]}
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    
    # 🌟 [에러 해결]: 중복 키 에러를 방지하기 위해 plotly_chart에서 key를 제거합니다.
    # placeholder(st.empty)가 이미 위치를 고정하고 있으므로 key 없이도 안전하게 갱신됩니다.
    placeholder.plotly_chart(fig_gauge, use_container_width=True)

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

st.title("🌐 Chainers Master Fund: Performance Monitoring Dashboard")
st.markdown("---")
st.markdown("## 📊 Master Fund Portfolio Report")

master_progress_placeholder = st.empty()
summary_placeholder = st.empty()
st.markdown("---")

# ==========================================
# 📊 2. 통합 대시보드 렌더링 함수 (Bold & Red Text 반영)
# ==========================================
def draw_top_dashboard(final_contribs, container, is_updating=False):
    df_contrib = pd.DataFrame(final_contribs)
    if df_contrib.empty: return {}
    
    df_contrib = df_contrib.sort_values(by="Member").reset_index(drop=True)
    distinct_colors = px.colors.qualitative.Plotly 
    df_contrib['Unique_Color'] = [distinct_colors[i % len(distinct_colors)] for i in range(len(df_contrib))]
    
    total_fund_capital = df_contrib['Final_Capital'].sum()

    # 도넛 차트 (텍스트 굵게)
    fig_donut = go.Figure(go.Pie(
        labels=df_contrib['Member'], values=df_contrib['Final_Capital'], hole=0.6,
        marker=dict(colors=df_contrib['Unique_Color']), textinfo="percent", 
        texttemplate="<b>%{percent}</b><br><b>%{value:.2f}$</b>", # 🌟 숫자/텍스트 굵게
        sort=False 
    ))
    title_text = "<b>Master Fund Contribution</b>"
    if is_updating: title_text += " <span style='color:#ff9800;'>(Updating...)</span>"
    
    fig_donut.update_layout(
        title=title_text, height=350, margin=dict(l=0, r=0, t=40, b=0),
        annotations=[dict(text=f"Total Capital<br><b>{total_fund_capital:.2f} $</b>", x=0.5, y=0.5, font_size=18, showarrow=False)],
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=-0.4) 
    )

    # 수익 바 차트 (텍스트 굵게)
    fig_profit = go.Figure(go.Bar(
        x=df_contrib['Member'], y=df_contrib['Profit_Dollar'], 
        marker=dict(color=df_contrib['Unique_Color']),
        text=df_contrib['Profit_Dollar'].apply(lambda x: f"<b>{x:.2f} $</b>"), # 🌟 숫자 굵게
        textposition='outside'
    ))
    fig_profit.update_yaxes(range=[min(0, df_contrib['Profit_Dollar'].min()*1.2), df_contrib['Profit_Dollar'].max() * 1.4])
    fig_profit.update_layout(title="<b>Portfolio Profit ($)</b>", height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

    # 성과 테이블 (음수 빨간색 텍스트 강조)
    table_data = []
    current_summary = {}
    prev_data = st.session_state.prev_summary

    for i, row in df_contrib.iterrows():
        m_name, c_ret, c_mdd = row['Member'], row['Avg_Return'], row['Avg_MDD']
        current_summary[m_name] = {'return': c_ret, 'mdd': c_mdd}
        ret_arrow = "(↑)" if m_name in prev_data and c_ret > prev_data[m_name]['return'] else "(↓)" if m_name in prev_data and c_ret < prev_data[m_name]['return'] else "(-)"
        mdd_arrow = "(↑)" if m_name in prev_data and c_mdd > prev_data[m_name]['mdd'] else "(↓)" if m_name in prev_data and c_mdd < prev_data[m_name]['mdd'] else "(-)"
        
        table_data.append({
            "Member": m_name, "Persona": row['CTPT_Code'], "Capital ($)": f"{row['Final_Capital']:.2f} $",
            "Return (%)": f"{c_ret:.2f} {ret_arrow}", "MDD (%)": f"{c_mdd:.2f} {mdd_arrow}"
        })
        
    def highlight_negatives(val):
        # 🌟 배경색 변경 없이 텍스트만 밝은 빨강(#FF4B4B) 적용
        if isinstance(val, str) and val.strip().startswith('-'): 
            return 'color: #FF4B4B; font-weight: bold;'
        return ''

    styled_table = pd.DataFrame(table_data).style.map(highlight_negatives)

    with container:
        col1, col2, col3 = st.columns([1, 1, 1.4])
        with col1: st.plotly_chart(fig_donut, use_container_width=True)
        with col2: st.plotly_chart(fig_profit, use_container_width=True)
        with col3: 
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Persona & Performance Report")
            st.dataframe(styled_table, use_container_width=True, hide_index=True)
            
    return current_summary

if st.session_state.prev_final_contributions:
    draw_top_dashboard(st.session_state.prev_final_contributions, summary_placeholder, is_updating=True)

# ==========================================
# 👨‍💻 3. 메인 연산 및 진행률 관리
# ==========================================
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try: team_modules.append(importlib.import_module(f"members.{item}.config"))
        except: pass

sorted_modules = sorted(team_modules, key=lambda m: getattr(m, "MEMBER_NAME", m.__name__))
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()}

total_charts = sum(len(st.session_state.get(f"ms_{getattr(m, 'MEMBER_NAME', m.__name__)}", 
                   [all_stock_names[idx] for idx in getattr(m, 'TARGET_INDICES', [])])) for m in sorted_modules)

final_contributions, total_episodes_run, rendered_count = [], 0, 0
if total_charts > 0: master_pbar = master_progress_placeholder.progress(0.0, text="🚀 Global Fund Analysis Starting...")

st.markdown("### Portfolio Managers (Independent RL Labs)")

for m_config in sorted_modules:
    m_name = getattr(m_config, "MEMBER_NAME", "Unknown")
    with st.container():
        st.subheader(f"📍 {m_name}")
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        ctpt_code, ctpt_desc, ctpt_color = calculate_ctpt_and_color(default_p.get("lr", global_lr), default_p.get("gamma", global_gamma), default_p.get("epsilon", global_epsilon))
        st.markdown(f"**Persona:** <span style='color:{ctpt_color}; font-weight:bold;'>{ctpt_code}</span> ({ctpt_desc})", unsafe_allow_html=True)

        selected_stock_names = st.multiselect(f"Stocks - {m_name}", options=list(all_stock_names.values()), 
                                              default=[all_stock_names[idx] for idx in getattr(m_config, 'TARGET_INDICES', [])], key=f"ms_{m_name}")
        
        member_returns, member_mdds = [], []
        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker, stock_idx = get_ticker_by_name(stock_name), name_to_index.get(stock_name)
                    p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                    
                    with st.expander(f"⚙️ {stock_name} Parameters", expanded=True):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            l_epi = st.slider("Trading Days", 10, 500, int(p_settings.get("episodes", global_episodes)), key=f"epi_{m_name}_{stock_name}")
                            l_seed = st.number_input("Seed", value=int(p_settings.get("seed", global_seed)), step=1, key=f"seed_{m_name}_{stock_name}")
                        with sc2:
                            l_lr = st.slider("LR", 0.001, 0.1, float(p_settings.get("lr", global_lr)), step=0.001, key=f"lr_{m_name}_{stock_name}")
                            l_gamma = st.slider("Gamma", 0.1, 0.99, float(p_settings.get("gamma", global_gamma)), key=f"gamma_{m_name}_{stock_name}")
                            l_epsilon = st.slider("Epsilon", 0.01, 0.5, float(p_settings.get("epsilon", global_epsilon)), key=f"eps_{m_name}_{stock_name}")

                    with st.spinner(f"📡 Processing {stock_name}..."):
                        df = fetch_stock_data(ticker, period="2y")
                        v_ret = run_rl_simulation(df, l_lr, l_gamma, l_epsilon, episodes=l_epi, use_static=False, seed=l_seed)
                        s_ret = run_rl_simulation(df, l_lr, l_gamma, l_epsilon, episodes=l_epi, use_static=True, seed=l_seed)
                        
                        fig = go.Figure()
                        m_ret = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                        fig.add_trace(go.Scatter(x=df.index, y=m_ret, mode='lines', name='Market', line=dict(color='#4caf50', width=5)))
                        fig.add_trace(go.Scatter(x=df.index, y=v_ret, mode='lines+markers', name='Vanilla', line=dict(color='#ff4b4b', width=1), marker=dict(symbol='square-open', size=5)))
                        fig.add_trace(go.Scatter(x=df.index, y=s_ret, mode='lines+markers', name='STATIC', line=dict(color='#2196f3', width=2.5), marker=dict(symbol='circle-open', size=6)))
                        fig.update_layout(title=f"<b>{stock_name}</b>", height=320, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_name}_{stock_name}")
                    
                    total_episodes_run += l_epi
                    update_gauge(total_episodes_run, gauge_placeholder)
                    rendered_count += 1
                    master_pbar.progress(rendered_count / total_charts, text=f"🚀 Analyzing Agents... ({int((rendered_count/total_charts)*100)}%)")
                    member_returns.append(s_ret[-1])
                    member_mdds.append(calculate_mdd(s_ret))

        if member_returns:
            avg_r = np.mean(member_returns)
            final_cap = 1.0*(1+avg_r/100)
            final_contributions.append({
                "Member": m_name, "Final_Capital": final_cap, "Profit_Dollar": final_cap-1.0,
                "Avg_Return": avg_r, "Avg_MDD": np.mean(member_mdds), "CTPT_Code": ctpt_code
            })
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

# --- 최종 갱신 ---
if final_contributions:
    master_progress_placeholder.empty()
    summary_placeholder.empty() 
    with summary_placeholder.container():
        current_summary = draw_top_dashboard(final_contributions, summary_placeholder)
    st.session_state.prev_final_contributions = final_contributions
    st.session_state.prev_summary = current_summary
    st.session_state.prev_episodes_run = total_episodes_run