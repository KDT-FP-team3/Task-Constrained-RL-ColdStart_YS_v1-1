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
from common.evaluator import calculate_metrics, calculate_ctpt_and_color

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")

# ==========================================
# 🎛️ 사이드바: 글로벌 기본값
# ==========================================
st.sidebar.markdown("### Global Default Parameters")
global_episodes = st.sidebar.slider("Episodes (Trading Days)", 10, 500, 100)
global_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Default RL Hyperparameters")
global_lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
global_gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.98)
global_epsilon = st.sidebar.slider("Exploration (ε)", 0.01, 0.5, 0.10)

st.title("🌐 Chainers Master Fund: Multi-Agent Global Portfolio Monitoring")

st.markdown("---")
summary_placeholder = st.empty()
st.markdown("---")

members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            mod = importlib.import_module(f"members.{item}.config")
            team_modules.append(mod)
        except: pass

# --- 실제 데이터 기반 차트 생성 함수 ---
def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50:
        return go.Figure(), 0.0

    dates = df.index
    prices = df['Close'].values
    real_return_percent = (prices / prices[0] - 1) * 100

    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)

    fig = go.Figure()
    
    # 1. 실제 주가 (Market): 녹색 실선, 가장 굵게 (width=5)
    # 다른 선들이 겹쳐도 배경처럼 넓게 보이도록 굵기를 키웠습니다.
    fig.add_trace(go.Scatter(
        x=dates, y=real_return_percent, 
        mode='lines', 
        name=f'Market', 
        line=dict(color='#4caf50', width=5)
    ))
    
    # 2. Vanilla RL: 붉은색 실선, 가장 가늘게 (width=1) + 빈 네모 심벌 (square-open)
    fig.add_trace(go.Scatter(
        x=dates, y=vanilla_return, 
        mode='lines+markers', 
        name='Vanilla RL', 
        line=dict(color='#ff4b4b', width=1),
        marker=dict(symbol='square-open', size=5, color='#ff4b4b')
    ))
    
    # 3. STATIC RL: 파란색 실선, 중간 굵기 (width=2.5) + 빈 원형 심벌 (circle-open)
    fig.add_trace(go.Scatter(
        x=dates, y=static_return, 
        mode='lines+markers', 
        name='STATIC RL (Ours)', 
        line=dict(color='#2196f3', width=2.5),
        marker=dict(symbol='circle-open', size=6, color='#2196f3')
    ))
    
    fig.update_layout(
        title=f"<b>{stock_name}</b> (Epi:{episodes}, Seed:{seed} | LR:{lr:.3f}, γ:{gamma:.2f})",
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig, static_return[-1] if len(static_return) > 0 else 0.0

final_contributions_data = []

# --- 1. 팀원별 독립 워크스페이스 렌더링 루프 ---
st.markdown("### 👨‍💻 Individual Member Labs")

# [매핑 준비] 종목 인덱스와 이름을 상호 변환하기 위한 딕셔너리 생성
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()} # "엔비디아" -> 4 변환용

for m_config in team_modules:
    with st.container():
        st.subheader(f"📍 {m_config.MEMBER_NAME}")
        
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        c_lr = default_p.get("lr", global_lr)
        c_gamma = default_p.get("gamma", global_gamma)
        c_epsilon = default_p.get("epsilon", global_epsilon)

        ctpt_code, ctpt_color, ctpt_desc = calculate_ctpt_and_color(c_lr, c_gamma, c_epsilon)
        st.markdown(f"**Persona:** <span style='color:{ctpt_color}; font-weight:bold;'>{ctpt_code}</span> ({ctpt_desc})", unsafe_allow_html=True)

        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        
        selected_stock_names = st.multiselect(
            f"차트 추가 (최대 10개) - {m_config.MEMBER_NAME}",
            options=list(all_stock_names.values()),
            default=default_names,
            max_selections=10,
            key=f"ms_{m_config.MEMBER_NAME}"
        )
        
        member_final_returns = [] 

        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    
                    # 💡 [핵심 매핑 로직]: 화면의 글씨("엔비디아")를 인덱스 번호(4)로 변환합니다.
                    stock_idx = name_to_index.get(stock_name)
                    
                    # 변환된 인덱스 번호로 config.py의 RL_PARAMS에서 설정값을 쏙 빼옵니다.
                    # 만약 해당 인덱스의 설정이 없다면 "default" 값을 가져옵니다.
                    p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                    
                    def_lr = p_settings.get("learning_rate", p_settings.get("lr", global_lr))
                    def_gamma = p_settings.get("discount_factor", p_settings.get("gamma", global_gamma))
                    def_epsilon = p_settings.get("exploration_rate", p_settings.get("epsilon", global_epsilon))
                    def_epi = p_settings.get("episodes", global_episodes)
                    def_seed = p_settings.get("seed", global_seed)

                    with st.expander(f"⚙️ {stock_name} 파라미터 실시간 조절", expanded=True):
                        st.caption("막대를 움직이면 그래프가 즉시 재학습되어 업데이트됩니다.")
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            local_epi = st.slider("Trading Days", 10, 500, int(def_epi), key=f"epi_{m_config.MEMBER_NAME}_{stock_name}")
                            local_seed = st.number_input("Random Seed", value=int(def_seed), step=1, key=f"seed_{m_config.MEMBER_NAME}_{stock_name}")
                        with sc2:
                            local_lr = st.slider("Learning Rate", 0.001, 0.1, float(def_lr), step=0.001, format="%.3f", key=f"lr_{m_config.MEMBER_NAME}_{stock_name}")
                            local_gamma = st.slider("Discount Factor", 0.1, 0.99, float(def_gamma), key=f"gamma_{m_config.MEMBER_NAME}_{stock_name}")
                            local_epsilon = st.slider("Exploration", 0.01, 0.5, float(def_epsilon), key=f"eps_{m_config.MEMBER_NAME}_{stock_name}")

                    fig, final_ret = create_real_rl_chart(stock_name, ticker, local_lr, local_gamma, local_epsilon, local_epi, local_seed)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_config.MEMBER_NAME}_{stock_name}")
                    member_final_returns.append(final_ret)

        if member_final_returns:
            avg_return = np.mean(member_final_returns)
            final_capital = 1.0 * (1 + avg_return / 100) 
            final_contributions_data.append({
                "Member": m_config.MEMBER_NAME,
                "Final_Capital": final_capital,
                "CTPT_Code": ctpt_code,
                "CTPT_Color": ctpt_color
            })

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ==========================================
# 📊 --- 2. 최상단 통합 대시보드 렌더링 ---
# ==========================================
if final_contributions_data:
    df_contrib = pd.DataFrame(final_contributions_data)
    total_team_capital = df_contrib['Final_Capital'].sum()
    df_contrib['Contribution_Weight'] = df_contrib['Final_Capital'] / total_team_capital
    
    sum_table = df_contrib[['Member', 'CTPT_Code', 'Final_Capital']].copy()
    sum_table['Final_Capital'] = sum_table['Final_Capital'].map("{:,.2f} $".format)

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
            fig_donut.update_layout(
                title="<b>Chainers Master Fund 기여도 (최종 자본 가중치)</b>",
                height=380, margin=dict(l=10, r=10, t=60, b=10),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_table:
            st.markdown("#### Member Persona & Performance")
            st.dataframe(sum_table, use_container_width=True)
else:
    with summary_placeholder:
        st.warning("활성화된 팀원 에이전트가 없어 종합 모니터링 데이터를 산출할 수 없습니다.")