import streamlit as st
import importlib
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from common.stock_registry import STOCK_REGISTRY

# 루트 경로 설정
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")
st.title("🌐 Multi-Agent Global Portfolio Monitoring")

# --- 모의 강화학습 그래프 생성 함수 (나중에 실제 agent.py 로직으로 교체) ---
def create_rl_comparison_chart(stock_name):
    """Vanilla RL과 STATIC RL의 누적 수익률을 비교하는 더미(Dummy) 차트 생성"""
    days = np.arange(100)
    # Vanilla: 제약이 없어 변동성이 큼
    vanilla = np.cumsum(np.random.normal(0.001, 0.02, 100)) * 100
    # STATIC: 제약 조건(EMA 등)으로 하방이 방어되어 안정적임
    static = np.cumsum(np.random.normal(0.002, 0.01, 100)) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=vanilla, mode='lines', name='Vanilla RL', line=dict(color='#ff4b4b', width=2)))
    fig.add_trace(go.Scatter(x=days, y=static, mode='lines', name='STATIC RL (Ours)', line=dict(color='#2196f3', width=2)))
    
    fig.update_layout(
        title=f"<b>{stock_name} 누적 수익률 비교</b>",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 1. 팀원 모듈 자동 탐색 ---
members_dir = os.path.join(root_path, "members")
team_modules = []

for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            mod = importlib.import_module(f"members.{item}.config")
            team_modules.append(mod)
        except Exception as e:
            st.error(f"Error loading {item}: {e}")

# --- 2. 최상단 글로벌 모니터링 ---
st.markdown("### 📊 Global Performance Summary")
st.info("각 팀원 에이전트들의 성과가 취합되어 광역 포트폴리오 수익률이 여기에 표시됩니다.")
st.divider()

# --- 3. 팀원별 독립 워크스페이스 (스크롤 뷰 & 동적 그래프 추가) ---
# 전체 종목 이름 리스트 (선택기용)
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}

for m_config in team_modules:
    # 각 팀원별 컨테이너(구역) 생성
    with st.container():
        st.subheader(f"📍 {m_config.MEMBER_NAME}'s Workspace")
        
        # 파라미터 표시 영역
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            st.write("**Core RL Params:**", getattr(m_config, "RL_PARAMS", {}))
        with col_param2:
            st.write("**Custom Settings:**", getattr(m_config, "CUSTOM_PARAMS", {}))
            
        # 버튼형 다중 선택기: config.py의 TARGET_INDICES를 기본값으로 하되, 웹에서 자유롭게 추가/제거 가능
        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        
        selected_stock_names = st.multiselect(
            f"📈 차트에 추가할 종목을 클릭하여 선택하세요 (최대 10개)",
            options=list(all_stock_names.values()),
            default=default_names,
            max_selections=10,
            key=f"ms_{m_config.MEMBER_NAME}" # 각 팀원별 독립된 키 부여
        )
        
        # 선택된 종목 수에 맞춰 그래프를 2열(Grid)로 동적 렌더링
        if selected_stock_names:
            # 2개씩 짝지어서 화면에 출력하기 위해 컬럼 분리
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]: # 0, 1, 0, 1 순서로 컬럼에 배치
                    # 실제 RL 로직이 들어갈 자리에 더미 차트 연결
                    fig = create_rl_comparison_chart(stock_name)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("선택된 종목이 없습니다. 위에서 종목을 선택해 주세요.")
            
        st.markdown("<br><hr><br>", unsafe_allow_html=True) # 팀원 간 구분선