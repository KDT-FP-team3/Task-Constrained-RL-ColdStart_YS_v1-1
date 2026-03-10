import streamlit as st
import importlib
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가하여 절대 경로 임포트 문제 해결
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

from common.stock_registry import get_stock_info

st.set_page_config(page_title="Global RL Portfolio", layout="wide")
st.title("Global Portfolio RL Dashboard")

# 1. 자동 디스커버리 (Auto-Discovery)
members_dir = os.path.join(root_path, "members")
team_configs = []

for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            # 팀원별 config 모듈 동적 로드
            module = importlib.import_module(f"members.{item}.config")
            team_configs.append(module)
        except Exception as e:
            st.error(f"{item} 모듈 로드 실패: {e}")

# 2. 동적 UI 렌더링
if team_configs:
    tabs = st.tabs([config.MEMBER_NAME for config in team_configs])
    
    for i, tab in enumerate(tabs):
        with tab:
            config = team_configs[i]
            
            # 중앙 레지스트리 기반 종목 정보 매핑
            stocks = get_stock_info(config.TARGET_STOCK_INDICES)
            stock_names = [f"{s['name']} ({s['ticker']})" for s in stocks]
            st.markdown(f"### 📊 대상 종목: {', '.join(stock_names)}")
            
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧠 기본 파라미터 (RL_PARAMS)")
                if hasattr(config, "RL_PARAMS"):
                    for key, val in config.RL_PARAMS.items():
                        st.write(f"- **{key}**: {val}")
                        
            with col2:
                # 팀원이 추가한 확장 파라미터를 동적으로 읽어 UI에 반영
                st.markdown("#### 🛠 커스텀 파라미터 (CUSTOM_PARAMS)")
                if hasattr(config, "CUSTOM_PARAMS") and config.CUSTOM_PARAMS:
                    for key, val in config.CUSTOM_PARAMS.items():
                        st.write(f"- **{key}**: {val}")
                else:
                    st.write("설정된 커스텀 파라미터가 없습니다.")
            
            # 팀원이 커스텀 함수를 선언했는지 객체 성찰(Reflection)로 확인
            st.markdown("#### ⚙️ 주입된 커스텀 로직")
            if hasattr(config, "custom_reward_function"):
                st.success("✅ `custom_reward_function`이 정상적으로 로드되어 에이전트에 주입될 준비가 되었습니다.")
            
            # 차트 영역 (기본 그래프 창 2개 배치 로직 예시)
            st.markdown("#### 📈 시뮬레이션 결과 (Vanilla vs STATIC)")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.info(f"{stock_names[0]} 그래프 렌더링 영역")
            with chart_col2:
                if len(stock_names) > 1:
                    st.info(f"{stock_names[1]} 그래프 렌더링 영역")
else:
    st.warning("맴버 설정 파일이 발견되지 않았습니다.")