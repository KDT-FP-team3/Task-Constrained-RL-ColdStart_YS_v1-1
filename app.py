import streamlit as st
import importlib
import os
import sys
import numpy as np
import plotly.graph_objects as go

from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")

# ==========================================
# 🎛️ 첨부 이미지 기반 사이드바 UI 구현
# ==========================================
st.sidebar.markdown("### System Parameters")
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, 500, 100)
frame_speed = st.sidebar.slider("Frame Speed (sec)", 0.01, 1.0, 0.03)
base_seed = st.sidebar.number_input("Base Random Seed", value=2026, step=1)
auto_run = st.sidebar.number_input("Auto Run Count", value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### RL Hyperparameters (Logic: STATIC)")
global_lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
global_gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.98)
global_epsilon = st.sidebar.slider("Exploration (ε)", 0.01, 0.5, 0.10)

st.title("🌐 Multi-Agent Global Portfolio Monitoring")

# --- 실제 데이터 기반 차트 생성 함수 ---
def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50:
        return go.Figure().update_layout(title=f"⚠️ {stock_name} 데이터 부족")

    dates = df.index
    # 실제 주가 누적 수익률
    prices = df['Close'].values
    real_return = (prices / prices[0] - 1) * 100

    # 실제 강화학습 시뮬레이션 구동 (app.py -> base_agent.py)
    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, use_static=True, seed=seed)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=real_return, mode='lines', name=f'Market ({stock_name})', line=dict(color='#4caf50', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=vanilla_return, mode='lines', name='Vanilla RL', line=dict(color='#ff4b4b', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=dates, y=static_return, mode='lines', name='STATIC RL (Ours)', line=dict(color='#2196f3', width=2)))
    
    fig.update_layout(
        title=f"<b>{stock_name} (LR:{lr:.3f}, γ:{gamma:.2f}, ε:{epsilon:.2f})</b>",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 팀원 모듈 탐색 ---
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except: pass

st.divider()

# --- 팀원별 작업 공간 ---
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}

for m_config in team_modules:
    with st.container():
        st.subheader(f"📍 {m_config.MEMBER_NAME}'s Workspace")
        
        # 각 팀원이 설정한 파라미터 불러오기 (설정이 없으면 사이드바 글로벌 값 사용)
        rl_params = getattr(m_config, "RL_PARAMS", {})
        m_lr = rl_params.get("learning_rate", global_lr)
        m_gamma = rl_params.get("discount_factor", global_gamma)
        m_epsilon = rl_params.get("exploration_rate", global_epsilon)
        
        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        
        selected_stock_names = st.multiselect(
            f"📈 차트 추가 (최대 10개) - 현재 적용된 파라미터 (α:{m_lr}, γ:{m_gamma})",
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
                    # 실제 RL 알고리즘에 팀원 고유의 파라미터를 주입하여 그래프 생성!
                    fig = create_real_rl_chart(stock_name, ticker, m_lr, m_gamma, m_epsilon, base_seed)
                    st.plotly_chart(fig, use_container_width=True)
                    
        st.markdown("<br><hr><br>", unsafe_allow_html=True)