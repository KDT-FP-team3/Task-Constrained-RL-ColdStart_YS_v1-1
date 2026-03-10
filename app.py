import streamlit as st
import importlib
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 작성해둔 공통 모듈 불러오기
from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation
# evaluator.py에 추가한 신규 함수 임포트
from common.evaluator import calculate_metrics, calculate_ctpt_and_color

# 루트 경로 설정
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

st.title("🌐 Chainers Master Fund: Multi-Agent Globa Portfolio Monitoring")

# 🌟 최상단 통합 대시보드 영역 (데이터 수집 전 빈 공간 확보)
st.markdown("---")
summary_placeholder = st.empty()
st.markdown("---")

# --- 팀원 모듈 탐색 ---
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            mod = importlib.import_module(f"members.{item}.config")
            team_modules.append(mod)
        except: pass

# --- 실제 데이터 기반 차트 생성 함수 (evaluate 모듈 활용) ---
def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50:
        return go.Figure(), 0.0

    dates = df.index
    prices = df['Close'].values
    # 실제 주가 누적 수익률 (%)
    real_return_percent = (prices / prices[0] - 1) * 100

    # 실제 강화학습 시뮬레이션 구동 (에피소드 파라미터 추가 반영)
    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)

    # common/evaluator.py를 활용한 지표 계산
    total_market_return, market_vol = calculate_metrics(real_return_percent)
    total_static_return, static_vol = calculate_metrics(static_return)

    fig = go.Figure()
    # 실제 주가 (녹색 선)
    fig.add_trace(go.Scatter(x=dates, y=real_return_percent, mode='lines', name=f'Market', line=dict(color='#4caf50', width=2)))
    # STATIC RL (안정적, 파란색 실선)
    fig.add_trace(go.Scatter(x=dates, y=static_return, mode='lines', name='STATIC RL', line=dict(color='#2196f3', width=2)))
    
    fig.update_layout(
        title=f"<b>{stock_name}</b> (Epi:{episodes}, Seed:{seed} | LR:{lr}, γ:{gamma})",
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig, total_static_return # 기여도 계산을 위해 최종 수익률(%) 반환

# --- 글로벌 기여도 계산을 위한 데이터 적재소 ---
final_contributions_data = []

# --- 1. 팀원별 독립 워크스페이스 렌더링 루프 ---
st.markdown("### 👨‍💻 Individual Member Labs")
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}

for m_config in team_modules:
    with st.container():
        # getattr를 사용하여 새로운 파라미터가 추가되어도 코드 수정 없이 반영 (유연성 확보)
        # hasattr 성찰(Reflection) 기법 활용 가능
        st.subheader(f"📍 {m_config.MEMBER_NAME}")
        
        # [핵심] 성향 산출에 사용할 최종 파라미터 결정 logic
        # 1순위: 멤버 디폴트 파라미터, 2순위: 사이드바 전역 값
        m_params = getattr(m_config, "RL_PARAMS", {})
        c_lr = m_params.get("learning_rate", global_lr)
        c_gamma = m_params.get("discount_factor", global_gamma)
        c_epsilon = m_params.get("exploration_rate", global_epsilon)

        # evaluator.py를 활용한 CTPT 산출 및 색상 배정
        ctpt_code, ctpt_color, ctpt_desc = calculate_ctpt_and_color(c_lr, c_gamma, c_epsilon)

        # 성향 표시 UI
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
        
        member_final_returns = [] # 멤버 내 종목들의 수익률 평균을 구하기 위함

        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    
                    # 종목별 독립 파라미터 추출 logic (지난 번 구현 유지)
                    # members/member_X/config.py에서 "엔비디아": {...} 처럼 종목 이름을 키로 가지는 파라미터를 찾습니다.
                    # 만약 없으면 RL_PARAMS 내부의 "learning_rate" 등을 찾고, 그것도 없으면 전역 슬라이더 값을 씁니다.
                    # [수정된 부분] member_params -> m_params 로 이름 통일
                    # 종목 이름으로 된 설정이 없으면 default를 찾고, 그것도 없으면 전체 m_params를 씁니다.
                    p_settings = m_params.get(stock_name, m_params.get("default", m_params))
                    
                    # config.py에서 "learning_rate" 또는 "lr" 어떤 이름으로 작성해도 인식하도록 처리
                    p_lr = p_settings.get("learning_rate", p_settings.get("lr", global_lr))
                    p_gamma = p_settings.get("discount_factor", p_settings.get("gamma", global_gamma))
                    p_epsilon = p_settings.get("exploration_rate", p_settings.get("epsilon", global_epsilon))
                    p_episodes = p_settings.get("episodes", global_episodes)
                    p_seed = p_settings.get("seed", global_seed)

                    fig, final_ret = create_real_rl_chart(stock_name, ticker, p_lr, p_gamma, p_epsilon, p_episodes, p_seed)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_config.MEMBER_NAME}_{stock_name}")
                    member_final_returns.append(final_ret)

        # 해당 팀원의 기여도 데이터 적재 (전제: 각 팀원은 동일한 자본금 1.0으로 시작했다고 가정)
        if member_final_returns:
            avg_return = np.mean(member_final_returns)
            final_capital = 1.0 * (1 + avg_return / 100) # 평균 수익률을 반영한 최종 자본
            final_contributions_data.append({
                "Member": m_config.MEMBER_NAME,
                "Final_Capital": final_capital,
                "CTPT_Code": ctpt_code,
                "CTPT_Color": ctpt_color
            })

        st.markdown("<br><hr><br>", unsafe_allow_html=True) # 팀원 간 구분선

# ==========================================
# 📊 --- 2. 최상단 통합 대시보드 렌더링 (루프 종료 후) ---
# 데이터가 다 모였으므로 처음에 만들어둔 빈 공간(summary_placeholder)을 채웁니다.
# ==========================================
if final_contributions_data:
    df_contrib = pd.DataFrame(final_contributions_data)
    
    # 원형 링 도넛 그래프 구현 (박사님 요청 사항)
    # 기여도 비율 계산: (개인 최종 자본) / (전체 최종 자본 합계)
    total_team_capital = df_contrib['Final_Capital'].sum()
    df_contrib['Contribution_Weight'] = df_contrib['Final_Capital'] / total_team_capital
    
    # 요약 테이블 생성
    # Alpha 기대치 및 중앙값 수익률 등 앞선 그림 데이터의 통계적 유의성 분석 결과를 여기에 활용할 수 있습니다.
    sum_table = df_contrib[['Member', 'CTPT_Code', 'Final_Capital']].copy()
    sum_table['Final_Capital'] = sum_table['Final_Capital'].map("{:,.2f} $".format)

    with summary_placeholder.container():
        st.markdown("### 📊 Team Alpha Fund Global Monitoring")
        col_donut, col_table = st.columns([1, 1])
        
        with col_donut:
            # Plotly 원형 링 도넛 그래프
            fig_donut = go.Figure()
            fig_donut.add_trace(go.Pie(
                labels=df_contrib['Member'] + " (" + df_contrib['CTPT_Code'] + ")",
                values=df_contrib['Contribution_Weight'],
                hole=0.5, # 도넛 모양 (링)
                marker=dict(colors=df_contrib['CTPT_Color']), # 성향별 고유 색상 적용!
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