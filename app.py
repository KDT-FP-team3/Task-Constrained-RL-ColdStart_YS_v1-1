import streamlit as st
import importlib
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation
from common.evaluator import calculate_metrics, calculate_ctpt_and_color, calculate_mdd

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")

# ==========================================
# 🎛️ 사이드바: 서버 부하 계기판 & 숨겨진 글로벌 파라미터
# ==========================================
st.sidebar.markdown("### 🖥️ AI Server Load")
# 나중에 계산된 부하량을 넣기 위해 빈 공간(placeholder) 생성
load_placeholder = st.sidebar.empty() 

st.sidebar.markdown("---")
# 글로벌 파라미터는 여전히 필요하므로 접어두기(expander)로 숨김
with st.sidebar.expander("⚙️ Global Fallback Params", expanded=False):
    global_episodes = st.slider("Episodes", 10, 500, 100)
    global_seed = st.number_input("Base Seed", value=2026, step=1)
    global_lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    global_gamma = st.slider("Discount Factor", 0.1, 0.99, 0.98)
    global_epsilon = st.slider("Exploration", 0.01, 0.5, 0.10)

st.title("🌐 Chainers Master Fund: Multi-Agent Global Portfolio Monitoring")

st.markdown("---")
summary_placeholder = st.empty()
st.markdown("---")

members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except: pass

# --- 실제 데이터 기반 차트 생성 함수 (MDD 반환 추가) ---
def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50:
        return go.Figure(), 0.0, 0.0

    dates = df.index
    prices = df['Close'].values
    real_return_percent = (prices / prices[0] - 1) * 100

    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)

    # MDD 계산
    static_mdd = calculate_mdd(static_return)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=real_return_percent, mode='lines', name=f'Market', line=dict(color='#4caf50', width=5)))
    fig.add_trace(go.Scatter(x=dates, y=vanilla_return, mode='lines+markers', name='Vanilla RL', line=dict(color='#ff4b4b', width=1), marker=dict(symbol='square-open', size=5, color='#ff4b4b')))
    fig.add_trace(go.Scatter(x=dates, y=static_return, mode='lines+markers', name='STATIC RL (Ours)', line=dict(color='#2196f3', width=2.5), marker=dict(symbol='circle-open', size=6, color='#2196f3')))
    
    fig.update_layout(
        title=f"<b>{stock_name}</b> (Epi:{episodes}, Seed:{seed} | LR:{lr:.3f}, γ:{gamma:.2f})",
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig, static_return[-1] if len(static_return) > 0 else 0.0, static_mdd

final_contributions_data = []
total_compute_episodes = 0 # 서버 부하 계산용 변수

# --- 1. 팀원별 독립 워크스페이스 렌더링 루프 ---
st.markdown("### 👨‍💻 Individual Member Labs")
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()}

for m_config in team_modules:
    with st.container():
        st.subheader(f"📍 {m_config.MEMBER_NAME}")
        
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        ctpt_code, ctpt_color, ctpt_desc = calculate_ctpt_and_color(
            default_p.get("lr", global_lr), default_p.get("gamma", global_gamma), default_p.get("epsilon", global_epsilon)
        )
        st.markdown(f"**Persona:** <span style='color:{ctpt_color}; font-weight:bold;'>{ctpt_code}</span> ({ctpt_desc})", unsafe_allow_html=True)

        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        selected_stock_names = st.multiselect(f"차트 추가 (최대 10개) - {m_config.MEMBER_NAME}", options=list(all_stock_names.values()), default=default_names, max_selections=10, key=f"ms_{m_config.MEMBER_NAME}")
        
        member_final_returns = [] 
        member_mdds = [] # MDD 수집용 리스트

        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    stock_idx = name_to_index.get(stock_name)
                    p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                    
                    def_lr = p_settings.get("learning_rate", p_settings.get("lr", global_lr))
                    def_gamma = p_settings.get("discount_factor", p_settings.get("gamma", global_gamma))
                    def_epsilon = p_settings.get("exploration_rate", p_settings.get("epsilon", global_epsilon))
                    def_epi = p_settings.get("episodes", global_episodes)
                    def_seed = p_settings.get("seed", global_seed)

                    with st.expander(f"⚙️ {stock_name} 파라미터 실시간 조절", expanded=True):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            local_epi = st.slider("Trading Days", 10, 500, int(def_epi), key=f"epi_{m_config.MEMBER_NAME}_{stock_name}")
                            local_seed = st.number_input("Random Seed", value=int(def_seed), step=1, key=f"seed_{m_config.MEMBER_NAME}_{stock_name}")
                        with sc2:
                            local_lr = st.slider("Learning Rate", 0.001, 0.1, float(def_lr), step=0.001, format="%.3f", key=f"lr_{m_config.MEMBER_NAME}_{stock_name}")
                            local_gamma = st.slider("Discount Factor", 0.1, 0.99, float(def_gamma), key=f"gamma_{m_config.MEMBER_NAME}_{stock_name}")
                            local_epsilon = st.slider("Exploration", 0.01, 0.5, float(def_epsilon), key=f"eps_{m_config.MEMBER_NAME}_{stock_name}")

                    # 그래프 생성 시 누적된 에피소드 횟수를 계산하여 서버 부하량 측정
                    total_compute_episodes += local_epi
                    
                    fig, final_ret, local_mdd = create_real_rl_chart(stock_name, ticker, local_lr, local_gamma, local_epsilon, local_epi, local_seed)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_config.MEMBER_NAME}_{stock_name}")
                    
                    member_final_returns.append(final_ret)
                    member_mdds.append(local_mdd)

        if member_final_returns:
            avg_return = np.mean(member_final_returns)
            avg_mdd = np.mean(member_mdds)
            final_capital = 1.0 * (1 + avg_return / 100) 
            final_contributions_data.append({
                "Member": m_config.MEMBER_NAME,
                "Final_Capital": final_capital,
                "Avg_Return": avg_return,
                "Avg_MDD": avg_mdd,
                "CTPT_Code": ctpt_code,
                "CTPT_Color": ctpt_color
            })
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ==========================================
# 🚀 --- 2. 사이드바 계기판 (서버 부하량 렌더링) ---
# ==========================================
# (팀원 6명이 각각 그래프 2개씩 띄우고 500 에피소드를 돌릴 때 = 최대 6000 정도의 부하)
max_capacity = 6000 
load_percentage = min((total_compute_episodes / max_capacity) * 100, 100)

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = load_percentage,
    number = {'suffix': "%"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Computing Load", 'font': {'size': 16}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "#2196f3"},
        'steps' : [
            {'range': [0, 50], 'color': "#1e1e1e"},
            {'range': [50, 80], 'color': "#4b4b4b"},
            {'range': [80, 100], 'color': "#ff4b4b"}],
        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
    }
))
fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
load_placeholder.plotly_chart(fig_gauge, use_container_width=True)


# ==========================================
# 📊 --- 3. 최상단 통합 대시보드 렌더링 (테이블 업데이트) ---
# ==========================================
if final_contributions_data:
    df_contrib = pd.DataFrame(final_contributions_data)
    total_team_capital = df_contrib['Final_Capital'].sum()
    df_contrib['Contribution_Weight'] = df_contrib['Final_Capital'] / total_team_capital
    
    # [수정됨] 요약 테이블에 수익률(Return)과 최대낙폭(MDD) 컬럼 추가
    sum_table = df_contrib[['Member', 'CTPT_Code', 'Avg_Return', 'Avg_MDD', 'Final_Capital']].copy()
    sum_table['Avg_Return'] = sum_table['Avg_Return'].map("{:,.2f} %".format)
    sum_table['Avg_MDD'] = sum_table['Avg_MDD'].map("{:,.2f} %".format)
    sum_table['Final_Capital'] = sum_table['Final_Capital'].map("{:,.2f} $".format)
    sum_table.columns = ['Member', 'Persona', 'Return(%)', 'MDD(%)', 'Capital($)']

    with summary_placeholder.container():
        st.markdown("### 📊 Team Alpha Fund Global Monitoring")
        col_donut, col_table = st.columns([1, 1])
        
        with col_donut:
            fig_donut = go.Figure()
            fig_donut.add_trace(go.Pie(
                labels=df_contrib['Member'] + " (" + df_contrib['CTPT_Code'] + ")",
                values=df_contrib['Contribution_Weight'],
                hole=0.5, 
                marker=dict(colors=df_contrib['CTPT_Color']), 
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Contribution: %{percent}<br>Capital: %{value:.2f} $ <extra></extra>"
            ))
            fig_donut.update_layout(title="<b>Chainers Master Fund 기여도</b>", height=380, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_table:
            st.markdown("#### Portfolio Performance Report")
            st.dataframe(sum_table, use_container_width=True)
else:
    with summary_placeholder:
        st.warning("활성화된 팀원 에이전트가 없어 데이터를 산출할 수 없습니다.")