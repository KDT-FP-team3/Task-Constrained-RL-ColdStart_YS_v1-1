import streamlit as st
import importlib
import os
import sys
import pandas as pd
import plotly.graph_objects as go
from common.stock_registry import get_stock_by_index
from common.data_loader import fetch_stock_data

# 루트 경로 설정
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Global Portfolio", layout="wide")
st.title("🌐 Multi-Agent Global Portfolio Monitoring")

# --- 1. 팀원 모듈 자동 탐색 (Auto-Discovery) ---
members_dir = os.path.join(root_path, "members")
team_modules = []

for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            mod = importlib.import_module(f"members.{item}.config")
            team_modules.append(mod)
        except Exception as e:
            st.error(f"Error loading {item}: {e}")

# --- 2. 최상단 글로벌 모니터링 (Global Aggregation) ---
st.markdown("### 📊 Global Performance Summary")
global_returns = []

# 모든 팀원의 종목 성과를 취합하여 광역 강화학습 성과를 산출하는 로직
for config in team_modules:
    # 인덱스 기반 종목 자동 선택
    for idx in config.TARGET_INDICES:
        stock_info = get_stock_by_index(idx)
        if stock_info:
            # 여기에서 실제 강화학습 시뮬레이션을 돌려 성과지표를 취합합니다.
            global_returns.append({"Member": config.MEMBER_NAME, "Stock": stock_info['name'], "Return": 15.0}) # 예시 값

if global_returns:
    st.dataframe(pd.DataFrame(global_returns), use_container_width=True)

# --- 3. 팀원별 독립 섹션 (Individual Workspaces) ---
st.divider()
if team_modules:
    tabs = st.tabs([m.MEMBER_NAME for m in team_modules])
    for i, tab in enumerate(tabs):
        with tab:
            m_config = team_modules[i]
            st.subheader(f"📍 {m_config.MEMBER_NAME}'s Workspace")
            
            # 팀원이 추가한 임의의 파라미터를 동적으로 표시 (유연성 확보)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Core RL Params:**", m_config.RL_PARAMS)
            with col2:
                # hasattr를 사용하여 팀원이 추가한 변수를 동적으로 감지
                custom_params = getattr(m_config, "CUSTOM_PARAMS", {})
                st.write("**Custom Settings:**", custom_params)
            
            # 선택된 종목 시각화 (예시 그래프 창 2개)
            g1, g2 = st.columns(2)
            for j, s_idx in enumerate(m_config.TARGET_INDICES[:2]):
                s_info = get_stock_by_index(s_idx)
                with [g1, g2][j]:
                    st.info(f"{s_info['name']} ({s_info['ticker']}) Chart Area")