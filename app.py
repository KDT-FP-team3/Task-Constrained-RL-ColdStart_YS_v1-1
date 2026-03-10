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

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")

# ==========================================
# 🎛️ 사이드바: 글로벌 기본값 (불필요 변수 제거)
# ==========================================
st.sidebar.markdown("### Global Default Parameters")
global_episodes = st.sidebar.slider("Episodes (Trading Days)", 10, 500, 100)
global_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Default RL Hyperparameters")
global_lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
global_gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.98)
global_epsilon = st.sidebar.slider("Exploration (ε)", 0.01, 0.5, 0.10)

st.title("🌐 Multi-Agent Global Portfolio Monitoring")

# --- 팀원 모듈 탐색 ---
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except: pass

# 전역 포트폴리오 데이터를 모을 딕셔너리
global_portfolio_data = {}
summary_table_data = []

def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50:
        return go.Figure(), None

    dates = df.index
    prices = df['Close'].values
    real_return = (prices / prices[0] - 1) * 100

    # 훈련 횟수(episodes)와 시드(seed) 적용
    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)

    # 전역 포트폴리오 계산을 위해 데이터 반환
    global_portfolio_data[stock_name] = {"dates": dates, "static_return": static_return, "market_return": real_return}
    
    # 요약 테이블용 데이터 적재
    summary_table_data.append({
        "Stock": stock_name,
        "Market Return (%)": round(real_return[-1], 2),
        "STATIC Return (%)": round(static_return[-1], 2),
        "Alpha (%)": round(static_return[-1] - real_return[-1], 2)
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=real_return, mode='lines', name=f'Market', line=dict(color='#4caf50', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=vanilla_return, mode='lines', name='Vanilla RL', line=dict(color='#ff4b4b', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=dates, y=static_return, mode='lines', name='STATIC RL', line=dict(color='#2196f3', width=2)))
    
    fig.update_layout(
        title=f"<b>{stock_name}</b> (Epi:{episodes}, Seed:{seed} | LR:{lr}, γ:{gamma}, ε:{epsilon})",
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig, static_return

st.divider()
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}

# --- 1. 팀원별 독립 워크스페이스 렌더링 ---
st.markdown("### 👨‍💻 Individual Member Workspaces")
for m_config in team_modules:
    with st.container():
        st.subheader(f"📍 {m_config.MEMBER_NAME}")
        
        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        
        selected_stock_names = st.multiselect(
            f"차트 추가 (최대 10개) - {m_config.MEMBER_NAME}",
            options=list(all_stock_names.values()),
            default=default_names,
            max_selections=10,
            key=f"ms_{m_config.MEMBER_NAME}"
        )
        
        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    
                    # [핵심 로직] 종목별 독립 파라미터 추출
                    member_params = getattr(m_config, "RL_PARAMS", {})
                    # 1순위: 종목 이름으로 지정된 전용 파라미터, 2순위: 멤버 디폴트, 3순위: 사이드바 글로벌
                    specific_params = member_params.get(stock_name, member_params.get("default", {}))
                    
                    p_lr = specific_params.get("lr", global_lr)
                    p_gamma = specific_params.get("gamma", global_gamma)
                    p_epsilon = specific_params.get("epsilon", global_epsilon)
                    p_episodes = specific_params.get("episodes", global_episodes)
                    p_seed = specific_params.get("seed", global_seed)

                    fig, _ = create_real_rl_chart(stock_name, ticker, p_lr, p_gamma, p_epsilon, p_episodes, p_seed)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_config.MEMBER_NAME}_{stock_name}")

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

# --- 2. 최상단 광역 통합 모니터링 렌더링 (데이터가 모인 후 최상단에 그리기 위해 placeholder 활용 고려, 여기서는 구조상 맨 아래 또는 맨 위 배치) ---
# Streamlit은 위에서 아래로 실행되므로, 전체 데이터가 취합된 후 상단에 표시하려면 st.empty()를 쓰거나, 위치를 아래로 둡니다.
# 시각적 흐름을 위해 요약 뷰를 하단에 배치하거나, 코드를 리팩토링하여 최상단 빈 공간(st.empty)에 밀어넣을 수 있습니다.