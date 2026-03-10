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

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path: sys.path.append(root_path)

st.set_page_config(page_title="Chainers Master Fund", layout="wide", initial_sidebar_state="collapsed")

if 'prev_summary' not in st.session_state:
    st.session_state.prev_summary = {}

st.sidebar.markdown("### Computing Load")
gauge_placeholder = st.sidebar.empty() 

st.sidebar.markdown("---")
with st.sidebar.expander("Fallback Parameters", expanded=False):
    global_episodes = st.slider("Episodes", 10, 500, 100)
    global_seed = st.number_input("Random Seed", value=2026, step=1)
    global_lr = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
    global_gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.98)
    global_epsilon = st.slider("Exploration (ε)", 0.01, 0.5, 0.10)

st.title("🌐 Chainers Master Fund: Performance Monitoring Dashboard")

st.markdown("---")
st.markdown("## 📊 Master Fund Portfolio Report")
summary_placeholder = st.empty()
softmax_placeholder = st.empty() # Softmax 게이지 차트용 공간 추가
st.markdown("---")

members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except: pass

def create_real_rl_chart(stock_name, ticker, lr, gamma, epsilon, episodes, seed):
    df = fetch_stock_data(ticker, period="2y")
    if df.empty or len(df) < 50: return go.Figure(), 0.0, 0.0

    dates = df.index
    real_return_percent = (df['Close'] / df['Close'].iloc[0] - 1) * 100

    vanilla_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=False, seed=seed)
    static_return = run_rl_simulation(df, lr, gamma, epsilon, episodes=episodes, use_static=True, seed=seed)
    static_mdd = calculate_mdd(static_return)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=real_return_percent, mode='lines', name=f'Market', line=dict(color='#4caf50', width=5)))
    fig.add_trace(go.Scatter(x=dates, y=vanilla_return, mode='lines+markers', name='Vanilla RL', line=dict(color='#ff4b4b', width=1), marker=dict(symbol='square-open', size=5, color='#ff4b4b')))
    fig.add_trace(go.Scatter(x=dates, y=static_return, mode='lines+markers', name='STATIC RL', line=dict(color='#2196f3', width=2.5), marker=dict(symbol='circle-open', size=6, color='#2196f3')))
    
    fig.update_layout(title=f"<b>{stock_name}</b> (Epi:{episodes} | α:{lr:.3f}, γ:{gamma:.2f})", height=320, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, static_return[-1] if len(static_return) > 0 else 0.0, static_mdd

final_contributions = []
total_episodes_run = 0 

st.markdown("### 👨‍💻 Portfolio Managers (Independent RL Labs)")
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()}
sorted_modules = sorted(team_modules, key=lambda m: getattr(m, "MEMBER_NAME", m.__name__))

for m_config in sorted_modules:
    with st.container():
        m_name = getattr(m_config, "MEMBER_NAME", "Unknown")
        st.subheader(f"📍 {m_name}")
        
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        ctpt_code, ctpt_desc, ctpt_color = calculate_ctpt_and_color(
            default_p.get("lr", global_lr), default_p.get("gamma", global_gamma), default_p.get("epsilon", global_epsilon)
        )
        st.markdown(f"**Persona:** <span style='color:{ctpt_color}; font-weight:bold;'>{ctpt_code}</span> ({ctpt_desc})", unsafe_allow_html=True)

        default_indices = getattr(m_config, "TARGET_INDICES", [])
        default_names = [all_stock_names[idx] for idx in default_indices if idx in all_stock_names]
        selected_stock_names = st.multiselect(f"차트 추가 - {m_name}", options=list(all_stock_names.values()), default=default_names, max_selections=10, key=f"ms_{m_name}")
        
        member_returns, member_mdds = [], []

        if selected_stock_names:
            cols = st.columns(2)
            for j, stock_name in enumerate(selected_stock_names):
                with cols[j % 2]:
                    ticker = get_ticker_by_name(stock_name)
                    stock_idx = name_to_index.get(stock_name)
                    p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                    
                    with st.expander(f"⚙️ {stock_name} 파라미터 조절", expanded=True):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            local_epi = st.slider("Trading Days", 10, 500, int(p_settings.get("episodes", global_episodes)), key=f"epi_{m_name}_{stock_name}")
                            local_seed = st.number_input("Random Seed", value=int(p_settings.get("seed", global_seed)), step=1, key=f"seed_{m_name}_{stock_name}")
                        with sc2:
                            local_lr = st.slider("Learning Rate", 0.001, 0.1, float(p_settings.get("lr", global_lr)), step=0.001, format="%.3f", key=f"lr_{m_name}_{stock_name}")
                            local_gamma = st.slider("Discount Factor", 0.1, 0.99, float(p_settings.get("gamma", global_gamma)), key=f"gamma_{m_name}_{stock_name}")
                            local_epsilon = st.slider("Exploration", 0.01, 0.5, float(p_settings.get("epsilon", global_epsilon)), key=f"eps_{m_name}_{stock_name}")

                    total_episodes_run += local_epi
                    fig, final_ret, local_mdd = create_real_rl_chart(stock_name, ticker, local_lr, local_gamma, local_epsilon, local_epi, local_seed)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{m_name}_{stock_name}")
                    member_returns.append(final_ret)
                    member_mdds.append(local_mdd)

        if member_returns:
            avg_return = np.mean(member_returns)
            final_capital = 1.0 * (1 + avg_return / 100)
            final_contributions.append({
                "Member": m_name, "Final_Capital": final_capital, "Profit_Dollar": final_capital - 1.0,
                "Avg_Return": avg_return, "Avg_MDD": np.mean(member_mdds), "CTPT_Code": ctpt_code
            })
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ==========================================
# 📊 --- 2. 최상단 통합 대시보드 렌더링 ---
# ==========================================
if final_contributions:
    df_contrib = pd.DataFrame(final_contributions)
    df_contrib = df_contrib.sort_values(by="Member").reset_index(drop=True)
    
    # 🌟 멤버별로 고유하고 뚜렷한 색상 팔레트 강제 배정 (파란색 통일 문제 해결)
    distinct_colors = px.colors.qualitative.Plotly 
    df_contrib['Unique_Color'] = [distinct_colors[i % len(distinct_colors)] for i in range(len(df_contrib))]
    
    total_fund_capital = df_contrib['Final_Capital'].sum()
    total_fund_profit = df_contrib['Profit_Dollar'].sum()
    df_contrib['Contribution_Weight'] = df_contrib['Final_Capital'] / total_fund_capital
    
    # 🌟 Softmax 알고리즘 구현 (Temperature 적용하여 극단값 방지)
    tau = 20.0 
    z_scaled = df_contrib['Avg_Return'].values / tau
    exp_z = np.exp(z_scaled - np.max(z_scaled)) # 오버플로우 방지
    softmax_weights = exp_z / np.sum(exp_z)
    df_contrib['Softmax_Weight'] = softmax_weights

    # 1) 도넛 그래프 (텍스트 간소화 및 범례 순서 고정)
    fig_donut = go.Figure(go.Pie(
        labels=df_contrib['Member'], 
        values=df_contrib['Final_Capital'], 
        hole=0.6,
        marker=dict(colors=df_contrib['Unique_Color']), 
        textinfo="percent", # 🌟 텍스트에서 Member 이름 제거
        texttemplate="%{percent}<br>%{value:.2f}$",
        hovertemplate="<b>%{label}</b><br>Capital: %{value:.2f} $ <extra></extra>",
        sort=False # 🌟 멤버 1번부터 차례대로 범례와 조각이 정렬되도록 고정
    ))
    fig_donut.update_layout(
        title="<b>Master Fund Contribution</b>", height=350, margin=dict(l=0, r=0, t=40, b=0),
        annotations=[dict(text=f"Total Capital<br><b>{total_fund_capital:.2f} $</b>", x=0.5, y=0.5, font_size=18, showarrow=False)],
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=-0.4, traceorder="normal") # 좌측 상단 범례
    )

    # 2) 수익 바 차트
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Bar(x=["Total Fund"], y=[total_fund_profit], name="Total", marker=dict(color='#ff4b4b')))
    fig_profit.add_trace(go.Bar(x=df_contrib['Member'], y=df_contrib['Profit_Dollar'], name="Members", marker=dict(color=df_contrib['Unique_Color'])))
    fig_profit.update_layout(title="<b>Portfolio Profit ($)</b>", height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

    # 3) 성과 테이블
    table_data = []
    current_summary = {}
    prev_data = st.session_state.prev_summary

    for i, row in df_contrib.iterrows():
        m_name = row['Member']
        c_ret, c_mdd = row['Avg_Return'], row['Avg_MDD']
        current_summary[m_name] = {'return': c_ret, 'mdd': c_mdd}
        
        ret_arrow = "(↑)" if m_name in prev_data and c_ret > prev_data[m_name]['return'] else "(↓)" if m_name in prev_data and c_ret < prev_data[m_name]['return'] else "(-)"
        mdd_arrow = "(↑)" if m_name in prev_data and c_mdd > prev_data[m_name]['mdd'] else "(↓)" if m_name in prev_data and c_mdd < prev_data[m_name]['mdd'] else "(-)"

        table_data.append({
            "Member": m_name, "Persona": row['CTPT_Code'],
            "Capital ($)": f"{row['Final_Capital']:.2f} $",
            "Return (%)": f"{c_ret:.2f} {ret_arrow}",
            "MDD (%)": f"{c_mdd:.2f} {mdd_arrow}",
            "Opt. Weight": f"{row['Softmax_Weight']*100:.1f} %" # Softmax 비중 추가
        })
        
    st.session_state.prev_summary = current_summary
    styled_table = pd.DataFrame(table_data).style.map(lambda val: 'color: #ff4b4b; font-weight: bold;' if isinstance(val, str) and ('-' in val and '(-)' not in val) else '')

    with summary_placeholder.container():
        col1, col2, col3 = st.columns([1, 1, 1.4])
        with col1: st.plotly_chart(fig_donut, use_container_width=True)
        with col2: st.plotly_chart(fig_profit, use_container_width=True)
        with col3: 
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Persona & Profit / Loss")
            st.dataframe(styled_table, use_container_width=True, hide_index=True)
            
    # 🌟 4) [신규] Softmax 기반 최적 포트폴리오 비중 게이지 (Horizontal Stacked Bar)
    with softmax_placeholder.container():
        fig_softmax = go.Figure()
        for i, row in df_contrib.iterrows():
            # [신규 계산]: 전체 펀드 자본금을 Softmax 비중대로 나누었을 때의 목표 할당 금액
            target_allocation_dollar = total_fund_capital * row['Softmax_Weight']
            
            # 막대기 안에 표시될 두 줄짜리 텍스트 (비율 % \n 금액 $)
            bar_text = f"{row['Softmax_Weight']*100:.1f}%<br>{target_allocation_dollar:.2f} $"
            
            fig_softmax.add_trace(go.Bar(
                y=['Optimal Allocation'], 
                x=[row['Softmax_Weight'] * 100], 
                name=row['Member'], 
                orientation='h', 
                marker=dict(color=row['Unique_Color']),
                text=bar_text, 
                textposition='inside', # 막대기 안쪽에 텍스트 배치
                insidetextanchor='middle', # 텍스트 중앙 정렬
                customdata=[target_allocation_dollar], # hover 데이터용
                hovertemplate="<b>%{name}</b><br>Suggested Weight: %{x:.2f}%<br>Target Capital: %{customdata[0]:.2f} $<extra></extra>"
            ))
            
        fig_softmax.update_layout(
            barmode='stack', height=180, margin=dict(l=20, r=20, t=40, b=20),
            title="<b>🎯 Target Portfolio Allocation (by Softmax Function)</b>",
            xaxis=dict(showgrid=False, range=[0, 100], ticksuffix="%"),
            yaxis=dict(showticklabels=False), showlegend=False
        )
        st.plotly_chart(fig_softmax, use_container_width=True)

# ==========================================
# 🚀 --- 3. 사이드바 서버 부하 계기판 ---
# ==========================================
max_load = 6000 
load_pct = min((total_episodes_run / max_load) * 100, 100)
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=load_pct, number={'suffix': "%"},
    title={'text': "Computing Load"},
    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#2196f3"},
           'steps': [{'range': [0, 50], 'color': "#333"}, {'range': [50, 80], 'color': "#ff9800"}, {'range': [80, 100], 'color': "#ff4b4b"}]}
))
fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True)