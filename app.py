import streamlit as st
import importlib
import inspect
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from common.stock_registry import STOCK_REGISTRY, get_ticker_by_name, get_fee_info
from common.data_loader import fetch_stock_data
from common.base_agent import run_rl_simulation_with_log, run_neural_rl
from common.evaluator import calculate_ctpt_and_color, calculate_mdd, calculate_softmax_weights
from common.heuristic import PGActorCriticOptimizer

# 루트 경로 설정
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

# ── 실행 환경 감지 ──────────────────────────────────────────
# Streamlit Cloud: HOME=/home/appuser 또는 STREAMLIT_SHARING_MODE 환경변수
_IS_CLOUD = (
    os.environ.get("HOME", "") == "/home/appuser"
    or os.environ.get("STREAMLIT_SHARING_MODE", "") != ""
    or os.environ.get("IS_CLOUD", "") == "1"
)

# GPU 감지 (로컬 전용 — 클라우드는 항상 CPU)
_HAS_CUDA = False
_CUDA_DEVICE = "CPU"
if not _IS_CLOUD:
    try:
        import torch
        _HAS_CUDA = torch.cuda.is_available()
        if _HAS_CUDA:
            _CUDA_DEVICE = torch.cuda.get_device_name(0)
    except ImportError:
        pass

st.set_page_config(page_title="Chainers Master Fund", layout="wide", initial_sidebar_state="collapsed")

# ── 반응형 레이아웃 & 다크/라이트 모드 공통 CSS ──
st.markdown("""
<style>
/* 상단 여백 최소화 */
.block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }
/* metric 카드 compact */
[data-testid="stMetric"] { padding: 4px 8px !important; }
/* expander 내부 여백 축소 */
[data-testid="stExpander"] > div:last-child { padding: 0.5rem 0.75rem !important; }
/* plotly 차트 마진 제거 */
[data-testid="stPlotlyChart"] { margin-bottom: 0 !important; }
/* 다크/라이트 모드 div 테두리 */
.st-summary-card { border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; padding: 12px 14px; }
/* ── sticky-header-marker 숨김 ── */
.element-container:has(.sticky-header-marker) {
    display: none !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
}
/* ── 헤더 컨테이너 sticky 고정: > .element-container 직접 자식 확인으로 범위 제한 ── */
[data-testid="stVerticalBlock"]:has(> .element-container .sticky-header-marker) {
    position: sticky !important;
    top: 3.75rem !important;
    z-index: 999 !important;
    background-color: var(--background-color, #0e1117) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    padding-top: 0.5rem !important;
    padding-bottom: 2px !important;
}
/* ── 헤더 제목 소형화 ── */
.sticky-main-title {
    font-size: 1.4rem !important; font-weight: 700 !important;
    margin: 0.1rem 0 !important; line-height: 1.3 !important;
}
.sticky-sub-title {
    font-size: 1.1rem !important; font-weight: 700 !important;
    margin: 0.05rem 0 !important;
}
.sticky-divider { margin: 0.15rem 0 !important; border-color: rgba(128,128,128,0.3) !important; }
/* ── 버튼 공통: 텍스트 맞춤 너비, 균일 높이 ── */
[data-testid="stButton"] button[kind="primary"] {
    min-height: 2.4rem !important; height: 2.4rem !important;
    padding-left: 1.25rem !important; padding-right: 1.25rem !important;
    white-space: nowrap !important;
}
/* ── sim-btn-marker 컨테이너 숨김 ── */
.element-container:has(.sim-btn-marker) {
    display: none !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
}
/* ── 버튼 쌍 column gap 최소화: b2 직계 구조로 inner stHorizontalBlock만 타겟 ── */
[data-testid="stHorizontalBlock"]:has(> [data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container .sim-btn-marker) {
    gap: 4px !important;
}
/* ── Simulation 버튼 보라색: 중첩 컬럼 안쪽(column 포함 안 된) sim-btn-marker 컬럼 ── */
[data-testid="column"]:not(:has([data-testid="column"])):has(.sim-btn-marker) button[kind="primary"] {
    background-color: #7B2FBE !important;
    border-color: #7B2FBE !important;
}
[data-testid="column"]:not(:has([data-testid="column"])):has(.sim-btn-marker) button[kind="primary"]:hover {
    background-color: #6322A3 !important;
    border-color: #6322A3 !important;
}
/* ── stop-btn-marker: 인터럽트 ■ 버튼 (작은 흰색 정사각형) ── */
.element-container:has(.stop-btn-marker) {
    display: none !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
}
[data-testid="column"]:not(:has([data-testid="column"])):has(.stop-btn-marker) button {
    background-color: #f0f0f0 !important;
    color: #1a1a2e !important;
    border: 1.5px solid rgba(200,200,200,0.7) !important;
    min-width: 2.4rem !important; max-width: 2.4rem !important;
    width: 2.4rem !important; height: 2.4rem !important;
    padding: 0 !important; font-size: 1.0rem !important;
    border-radius: 4px !important;
}
[data-testid="column"]:not(:has([data-testid="column"])):has(.stop-btn-marker) button:hover {
    background-color: #ffcccc !important;
    border-color: #ff4b4b !important;
    color: #cc0000 !important;
}
</style>
""", unsafe_allow_html=True)

# --- UI 세션 상태 ---
if 'prev_summary' not in st.session_state:
    st.session_state.prev_summary = {}
if 'prev_final_contributions' not in st.session_state:
    st.session_state.prev_final_contributions = []
if 'prev_episodes_run' not in st.session_state:
    st.session_state.prev_episodes_run = 0
if 'master_pbar_pct' not in st.session_state:
    st.session_state.master_pbar_pct = 0.0
if 'stock_trial_history' not in st.session_state:
    st.session_state.stock_trial_history = {}  # key: f"{m_name}_{stock_name}" → list of trial dicts
if 'fallback_params' not in st.session_state:
    st.session_state.fallback_params = {}       # 마지막 "All 적용" 시 스냅샷
if 'fallback_prev_state' not in st.session_state:
    st.session_state.fallback_prev_state = {}   # 되돌리기용 이전 슬라이더 상태 스냅샷
if 'stock_use_fallback' not in st.session_state:
    st.session_state.stock_use_fallback = None  # "ALL" 이면 전체 fallback 활성
if 'stocks_reverted' not in st.session_state:
    st.session_state.stocks_reverted = set()    # Run Evaluation 클릭으로 되돌아온 종목
if 'sim_result' not in st.session_state:
    st.session_state.sim_result = {}            # key: hist_key → best params dict
if 'ghost_data' not in st.session_state:
    st.session_state.ghost_data = {}
    # key: hist_key → {'v_trace': np.array, 's_trace': np.array, 'params': dict, 'gap': float}
if 'member_traces' not in st.session_state:
    st.session_state.member_traces = {}
    # key: member_name → {'s_trace': np.array, 'dates': index, 'stocks': list[str]}
# [P1] Team Fund 배분 설정
if 'fund_temperature' not in st.session_state:
    st.session_state.fund_temperature = 1.0   # Softmax 온도 (높을수록 균등 배분)
if 'fund_max_weight' not in st.session_state:
    st.session_state.fund_max_weight = 0.40   # 단일 종목 최대 비중 (1.0 = 무제한)
# [P2] 학습된 정책 캐시 (State Analysis Dashboard용)
if 'policy_cache' not in st.session_state:
    st.session_state.policy_cache = {}
    # key: f"{m_name}_{stock_name}" → {'theta': np.array, 'q_table': np.array, 'n_states': int}
# [P3] 변동성 상태 신호 활성화
if 'use_vol_feature' not in st.session_state:
    st.session_state.use_vol_feature = False  # True: 8상태 (변동성 포함), False: 4상태 (기본)
# [P4] Rolling Window 재학습 설정
if 'roll_period_active' not in st.session_state:
    st.session_state.roll_period_active = False  # True: OOS 주기 재학습 활성화
if 'roll_period_val' not in st.session_state:
    st.session_state.roll_period_val = 20        # 재학습 주기 (봉 수)
# ══════════════════════════════════════════════════════════════════════
# 시뮬레이션 파라미터 영구 저장: config.py 재작성
# ══════════════════════════════════════════════════════════════════════
def _save_sim_params_to_config(m_config, stock_idx, new_params):
    """시뮬레이션 최적 파라미터를 해당 멤버의 config.py에 영구 저장.
    lr/gamma/epsilon/v_epsilon은 new_params로 덮어쓰고,
    episodes/train_episodes/seed/use_vol/roll_period는 현재 값 유지."""
    config_path = inspect.getfile(m_config)
    rl = {k: dict(v) for k, v in m_config.RL_PARAMS.items()}

    # 대상 종목 키 결정 (int 인덱스 또는 "default")
    target_key = stock_idx if stock_idx in rl else "default"
    rl[target_key]["lr"]        = round(float(new_params["lr"]),        6)
    rl[target_key]["gamma"]     = round(float(new_params["gamma"]),     6)
    rl[target_key]["epsilon"]   = round(float(new_params["epsilon"]),   6)
    rl[target_key]["v_epsilon"] = round(float(new_params["v_epsilon"]), 6)
    if "algorithm" in new_params:
        rl[target_key]["algorithm"] = str(new_params["algorithm"])

    # "default" 키도 동기화 (lr/gamma/epsilon/v_epsilon/algorithm)
    if target_key != "default" and "default" in rl:
        rl["default"]["lr"]        = rl[target_key]["lr"]
        rl["default"]["gamma"]     = rl[target_key]["gamma"]
        rl["default"]["epsilon"]   = rl[target_key]["epsilon"]
        rl["default"]["v_epsilon"] = rl[target_key]["v_epsilon"]
        if "algorithm" in new_params:
            rl["default"]["algorithm"] = rl[target_key]["algorithm"]

    # 종목명 주석용
    _tgt_name = STOCK_REGISTRY.get(m_config.TARGET_INDICES[0], {}).get("name", "")
    _gap_str = (f"gap={new_params.get('gap', 0.0):.4f}, "
                f"s_final={new_params.get('s_final', 0.0):.2f}%, "
                f"v_final={new_params.get('v_final', 0.0):.2f}%")

    lines = [
        f'MEMBER_NAME = "{m_config.MEMBER_NAME}"\n',
        f'TARGET_INDICES = {m_config.TARGET_INDICES} # {_tgt_name}\n',
        '\n',
        f'# [파라미터 — Simulation 저장: {_gap_str}]\n',
        'RL_PARAMS = {\n',
    ]
    for k, v in rl.items():
        key_str = 'TARGET_INDICES[0]' if k != "default" else '"default"'
        lines.append(f'    {key_str}: {{\n')
        lines.append(
            f'        "lr": {v["lr"]}, "gamma": {v["gamma"]}, '
            f'"epsilon": {v["epsilon"]}, "v_epsilon": {v["v_epsilon"]},\n'
        )
        lines.append(
            f'        "episodes": {v["episodes"]}, '
            f'"train_episodes": {v.get("train_episodes", 300)}, "seed": {v["seed"]},\n'
        )
        lines.append(
            f'        "use_vol": {v.get("use_vol", False)}, '
            f'"roll_period": {v.get("roll_period", None)},\n'
        )
        _alg = v.get("algorithm", "STATIC")
        lines.append(
            f'        "algorithm": "{_alg}"\n'
        )
        lines.append('    },\n')
    lines.append('}\n')

    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 저장 후 모듈 캐시 갱신: 같은 세션 내 p_settings가 최신 값을 반환하도록
    try:
        importlib.reload(m_config)
    except Exception:
        pass  # reload 실패 시 무시 (다음 서버 재시작 시 반영됨)


if 'run_all_queue' not in st.session_state:
    st.session_state.run_all_queue = []   # [(m_name, stock_name), ...] 순차 처리 큐
if 'sim_all_queue' not in st.session_state:
    st.session_state.sim_all_queue = []   # [(m_name, stock_name), ...] 순차 처리 큐
if 'sim_auto_save' not in st.session_state:
    st.session_state.sim_auto_save = False  # Simul. All 시 자동 저장 플래그
if 'interrupt_requested' not in st.session_state:
    st.session_state.interrupt_requested = False

# ==========================================
# 1. 실시간 시스템 상태: 단순 수평 막대
# ==========================================
def update_load_bar(episodes_run, placeholder, is_loading=False):
    """컬러 그라디언트 수평 막대로 Real-time Load 표시.
    50% 시 파랑→주황, 100% 시 파랑→주황→빨강 (background-size 트릭으로 비율 반영).
    max_load = 6멤버 × 500에피소드 = 3000 (전체 실행 완료 시 100%)."""
    max_load = 3000  # 6 members × 500 episodes
    load_pct = min(episodes_run / max_load, 1.0)
    pct_val = load_pct * 100
    bar_width = max(pct_val, 1.5) if pct_val > 0 else 0
    # background-size를 (100/load_pct)%로 설정 → 채워진 div 내에서 그라디언트의 앞부분만 보임
    # 예: 50% 채움 → background-size=200% → 그라디언트 0~50% 구간(파랑→주황)만 표시
    if load_pct > 0.005:
        bg_size = f"{100 / load_pct:.1f}% 100%"
    else:
        bg_size = "100% 100%"
    if is_loading:
        label = "⏳ Loading..."
    elif pct_val >= 100.0:
        label = f"Last Run: 완료 ({pct_val:.0f}%)"  # 실행 중 아님 — 마지막 실행 완료 상태
    elif pct_val > 0:
        label = f"Last Run: {pct_val:.1f}%"
    else:
        label = "Idle (실행 대기)"
    with placeholder:
        st.markdown(
            f"<div style='margin-bottom:6px;font-size:12px;font-weight:600;"
            f"color:rgba(210,210,210,0.95);'>{label}</div>"
            f"<div style='background:rgba(50,50,60,0.6);border-radius:5px;height:14px;"
            f"width:100%;overflow:hidden;border:1px solid rgba(120,120,140,0.3);'>"
            f"  <div style='height:100%;width:{bar_width:.1f}%;border-radius:5px;"
            f"background:linear-gradient(90deg,#2196f3 0%,#ff9800 50%,#ff4b4b 100%);"
            f"background-size:{bg_size};background-position:left center;"
            f"transition:width 0.3s ease;'></div>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:9px;"
            f"color:rgba(140,140,150,0.7);margin-top:2px;'>"
            f"<span>0%</span><span>50%</span><span>100%</span></div>",
            unsafe_allow_html=True
        )

def _render_master_pbar_html(pct, placeholder=None):
    """Analyzing Agents 진행률 바를 커스텀 HTML로 렌더링 (CSS injection 없이 즉시 색상 적용)."""
    bar_pct = min(int(pct * 100), 100)
    bar_color = "#AAFF00" if pct >= 1.0 else "#1C83E1"
    html = (
        f"<div style='margin-bottom:4px;font-size:13px;font-weight:600;"
        f"color:rgba(210,210,210,0.95);'>Analyzing Agents... ({bar_pct}%)</div>"
        f"<div style='background:rgba(50,50,60,0.6);border-radius:5px;height:14px;"
        f"width:100%;overflow:hidden;border:1px solid rgba(120,120,140,0.3);'>"
        f"<div style='height:100%;width:{bar_pct}%;border-radius:5px;"
        f"background:{bar_color};transition:width 0.3s ease;'></div>"
        f"</div>"
    )
    if placeholder is not None:
        placeholder.markdown(html, unsafe_allow_html=True)
    return html


st.sidebar.markdown("### System Status")
# ── 실행 환경 배지 ──────────────────────────────
_env_icon  = "☁️ Cloud" if _IS_CLOUD else "🖥️ Local"
_gpu_icon  = f"⚡ GPU ({_CUDA_DEVICE})" if _HAS_CUDA else "🔲 CPU"
st.sidebar.caption(f"{_env_icon} &nbsp;|&nbsp; {_gpu_icon}")
# ─────────────────────────────────────────────────
master_progress_placeholder = st.sidebar.empty()
gauge_placeholder = st.sidebar.empty()
# 스크립트 재실행 시 즉시 이전 값으로 렌더링 → 공백(사라짐) 방지
_render_master_pbar_html(st.session_state.master_pbar_pct, master_progress_placeholder)
update_load_bar(st.session_state.prev_episodes_run, gauge_placeholder)

st.sidebar.markdown("---")

# ── 1행: [▶ Eval. All] [⚙ Simul. All] [■ 중단] ──
_rq_cnt = len(st.session_state.run_all_queue)
_sq_cnt = len(st.session_state.sim_all_queue)
_sb_r1c1, _sb_r1c2, _sb_r1c3 = st.sidebar.columns([5, 5, 2])
with _sb_r1c1:
    _re_label = f"▶ 실행중({_rq_cnt})" if _rq_cnt > 0 else "▶ Eval. All"
    run_eval_all_btn = st.button(
        _re_label, key="sidebar_run_eval_all", type="primary",
        use_container_width=True,
        help="전체 멤버·종목에 Run Evaluation을 순차 실행합니다"
    )
with _sb_r1c2:
    _si_label = f"⚙ 실행중({_sq_cnt})" if _sq_cnt > 0 else "⚙ Simul. All"
    sim_all_btn = st.button(
        _si_label, key="sidebar_sim_all", type="primary",
        use_container_width=True,
        help="전체 멤버·종목에 Simulation(PG Actor-Critic)을 순차 실행합니다"
    )
with _sb_r1c3:
    st.markdown('<span class="stop-btn-marker"></span>', unsafe_allow_html=True)
    _sb_stop = st.button("■", key="sidebar_interrupt",
                         help="실행 중인 Eval. All / Simul. All 큐를 즉시 중단합니다",
                         use_container_width=True)

if _sb_stop:
    st.session_state.run_all_queue = []
    st.session_state.sim_all_queue = []
    st.session_state.sim_auto_save = False
    st.session_state.interrupt_requested = True

# ── 2행: [🔁 All 적용] [↩ 되돌리기] [🗑 초기화] ──
_fb_active = st.session_state.stock_use_fallback == "ALL"
_fb_label  = "✅ Fallback 적용 중" if _fb_active else "적용"
_has_prev  = bool(st.session_state.fallback_prev_state)
_sb_r2c1, _sb_r2c2, _sb_r2c3 = st.sidebar.columns([3, 2, 2])
apply_all_clicked = _sb_r2c1.button(
    _fb_label, key="sidebar_apply_all", type="primary",
    use_container_width=True,
    help="아래 설정값을 모든 멤버·종목 파라미터에 일괄 적용합니다"
)
revert_all_clicked = _sb_r2c2.button(
    "↩ 되돌리기", key="sidebar_revert_all",
    use_container_width=True,
    disabled=not _has_prev,
    help="All 적용 이전 상태로 모든 파라미터를 복원합니다"
)
if _sb_r2c3.button("🗑 초기화", key="btn_reset_dashboard",
                   use_container_width=True,
                   help="포트폴리오 차트 및 누적 리턴 데이터를 초기화합니다."):
    st.session_state.member_traces = {}
    st.session_state.prev_final_contributions = []
    st.rerun()

with st.sidebar.expander("Fund & Agent Settings", expanded=False):
    st.markdown("<div style='margin:0 0 2px 0'><small><b>[P1] Team Fund 배분 설정</b></small></div>", unsafe_allow_html=True)
    _fund_temp = st.slider(
        "Softmax Temperature (T)", 1.0, 5.0,
        float(st.session_state.fund_temperature), step=0.5,
        key="sb_fund_temp",
        help="낮을수록 성과 우수 종목 집중 / 높을수록 균등 배분 (T=5.0 ≈ 1/N)"
    )
    st.session_state.fund_temperature = _fund_temp
    _fund_cap = st.slider(
        "Max Single Weight (%)", 10, 100,
        int(st.session_state.fund_max_weight * 100), step=5,
        key="sb_fund_cap",
        help="단일 종목 최대 비중 상한 (100% = 무제한, 30% = 분산 강제)"
    ) / 100.0
    st.session_state.fund_max_weight = _fund_cap

    st.markdown("<hr style='margin:4px 0;border-color:rgba(128,128,128,0.3)'>", unsafe_allow_html=True)
    st.markdown("<div style='margin:2px 0'><small><b>[P3] 상태 공간 확장 — 변동성 신호</b></small></div>", unsafe_allow_html=True)
    _use_vol = st.toggle(
        "8-State Mode (변동성 신호 추가)",
        value=st.session_state.use_vol_feature,
        key="sb_use_vol",
        help="ON: 4상태→8상태 (Rolling_Std 이분 신호 추가). 기존 파라미터 최적화 재탐색 권장."
    )
    st.session_state.use_vol_feature = _use_vol

    st.markdown("<hr style='margin:4px 0;border-color:rgba(128,128,128,0.3)'>", unsafe_allow_html=True)
    st.markdown("<div style='margin:2px 0'><small><b>[P4] Rolling Window 재학습</b></small></div>", unsafe_allow_html=True)
    _roll_active = st.toggle(
        "Rolling Retrain (OOS 주기 재학습)",
        value=st.session_state.roll_period_active,
        key="sb_roll_active",
        help="ON: OOS 구간에서 roll_period 봉마다 최근 데이터로 재학습. STATIC RL 전용."
    )
    st.session_state.roll_period_active = _roll_active
    if _roll_active:
        _roll_val = st.number_input(
            "Roll Period (봉)", min_value=5, max_value=100,
            value=int(st.session_state.roll_period_val), step=5,
            key="sb_roll_val",
            help="OOS 구간에서 매 N봉마다 재학습 (예: 20 = 약 1개월 일봉)"
        )
        st.session_state.roll_period_val = int(_roll_val)

with st.sidebar.expander("Fallback Parameters", expanded=False):
    st.markdown("<small><b>System Parameters &nbsp;—&nbsp; ☑ 체크한 항목만 일괄 적용</b></small>", unsafe_allow_html=True)
    _fb_tf_options = ["15 min.", "1 hour", "1 day", "1 week", "1 month"]
    _fb_tf_map     = {"15 min.": "15m", "1 hour": "1h", "1 day": "1d", "1 week": "1wk", "1 month": "1mo"}
    _fb_lbl_map    = {"15m": "Bars (15min)", "1h": "Bars (1h)", "1d": "Trading Days", "1wk": "Trading Weeks", "1mo": "Trading Months"}
    _fb_min_map    = {"15m": 20, "1h": 20, "1d": 10, "1wk": 10, "1mo": 6}
    _fb_max_map    = {"15m": 400, "1h": 500, "1d": 500, "1wk": 200, "1mo": 60}
    _fb_def_map    = {"15m": 80, "1h": 120, "1d": 300 if _IS_CLOUD else 500, "1wk": 105, "1mo": 24}

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_timeframe", label_visibility="collapsed")
    with _wg:
        fb_tf_sel = st.selectbox(
            "Timeframe", _fb_tf_options, index=2,
            key="fb_timeframe",
            help="데이터 봉 단위 (15분/1시간: 최근 60일/730일 제한)"
        )
    fb_interval = _fb_tf_map[fb_tf_sel]

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_episodes", label_visibility="collapsed")
    with _wg:
        global_episodes = st.slider(
            _fb_lbl_map[fb_interval],
            _fb_min_map[fb_interval], _fb_max_map[fb_interval], _fb_def_map[fb_interval],
            key=f"fb_epi_{fb_interval}",
            help="시장 데이터 봉 수 (데이터 창 크기)"
        )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_train_epi", label_visibility="collapsed")
    with _wg:
        global_train_episodes = st.slider(
            "Train Episodes", 10, 500, 150 if _IS_CLOUD else 300,
            key="fb_train_epi",
            help="RL 학습 반복 횟수 (같은 훈련 데이터를 몇 번 반복 학습할지)"
        )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_frame", label_visibility="collapsed")
    with _wg:
        global_frame = st.slider("Frame Speed (sec)", 0.01, 2.0, 0.03,
                                 step=0.01, format="%.2f", key="fb_frame")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_seed", label_visibility="collapsed")
    with _wg:
        global_seed = st.number_input("Base Seed", value=2026, step=1, key="fb_seed")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_auto", label_visibility="collapsed")
    with _wg:
        global_auto_runs = st.number_input("Auto Run Count", min_value=1,
                                           value=5 if _IS_CLOUD else 10,
                                           step=1, key="fb_auto")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_sim_min", label_visibility="collapsed")
    with _wg:
        global_sim_min = st.number_input(
            "Sim Min Steps", min_value=5, max_value=200,
            value=20 if _IS_CLOUD else 30, step=5, key="fb_sim_min",
            help="시뮬레이션 최소 탐색 step 수 (n_iters 하한)"
        )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_sim_mult", label_visibility="collapsed")
    with _wg:
        global_sim_mult = st.number_input(
            "Sim Step Mult.", min_value=1, max_value=30,
            value=6 if _IS_CLOUD else 10, step=1, key="fb_sim_mult",
            help="n_iters = max(Min Steps, Auto Run Count × Mult.)"
        )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_active", label_visibility="collapsed")
    with _wg:
        global_active_agents = st.multiselect(
            "Active Agents",
            options=["Vanilla RL", "STATIC RL"],
            default=["Vanilla RL", "STATIC RL"],
            key="fb_active_agents",
            help="체크 해제된 에이전트는 연산 없이 0% 수평선으로 표시"
        )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_algo", label_visibility="collapsed")
    with _wg:
        global_algorithm = st.selectbox(
            "RL Algorithm",
            options=["STATIC", "A2C", "A3C", "PPO", "SAC", "DDPG"],
            index=0,
            key="fb_algorithm",
            help="STATIC=기존 tabular Actor-Critic / 나머지=NumPy 신경망 RL"
        )

    st.markdown(
        "<small><b>RL Hyperparameters &nbsp;"
        "<span style='color:#4a90d9;'>STATIC RL</span>: α / γ / ε(S) &nbsp;|&nbsp; "
        "<span style='color:#e05050;'>Vanilla RL</span>: ε(V)</b></small>",
        unsafe_allow_html=True
    )

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_lr", label_visibility="collapsed")
    with _wg:
        global_lr = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01,
                              step=0.001, format="%.3f", key="fb_lr")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_gamma", label_visibility="collapsed")
    with _wg:
        global_gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.98, key="fb_gamma")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_eps", label_visibility="collapsed")
    with _wg:
        global_epsilon = st.slider("STATIC ε", 0.01, 0.5, 0.10, key="fb_eps",
                                   help="STATIC RL 탐험율")

    _ck, _wg = st.columns([1, 5])
    with _ck:
        st.checkbox("", value=False, key="fb_chk_v_eps", label_visibility="collapsed")
    with _wg:
        global_v_epsilon = st.slider("Vanilla ε", 0.01, 0.5, 0.10, key="fb_v_eps",
                                     help="Vanilla RL 탐험율 (STATIC과 독립적으로 조정)")

# 버튼 클릭 시 현재 슬라이더 값 스냅샷 저장 (슬라이더가 위에서 이미 렌더됨)
if apply_all_clicked:
    _chks = {
        "timeframe": bool(st.session_state.get("fb_chk_timeframe", False)),
        "episodes":  bool(st.session_state.get("fb_chk_episodes",  False)),
        "train_epi": bool(st.session_state.get("fb_chk_train_epi", False)),
        "frame":     bool(st.session_state.get("fb_chk_frame",     False)),
        "seed":      bool(st.session_state.get("fb_chk_seed",      False)),
        "auto":      bool(st.session_state.get("fb_chk_auto",      False)),
        "sim_min":   bool(st.session_state.get("fb_chk_sim_min",   False)),
        "sim_mult":  bool(st.session_state.get("fb_chk_sim_mult",  False)),
        "active":    bool(st.session_state.get("fb_chk_active",    False)),
        "algo":      bool(st.session_state.get("fb_chk_algo",      False)),
        "lr":        bool(st.session_state.get("fb_chk_lr",        False)),
        "gamma":     bool(st.session_state.get("fb_chk_gamma",     False)),
        "eps":       bool(st.session_state.get("fb_chk_eps",       False)),
        "v_eps":     bool(st.session_state.get("fb_chk_v_eps",     False)),
    }
    st.session_state.fallback_params = {
        "timeframe":       fb_tf_sel,
        "interval":        fb_interval,
        "episodes":        global_episodes,
        "train_episodes":  global_train_episodes,
        "frame_speed":     global_frame,
        "seed":            int(global_seed),
        "auto_runs":       int(global_auto_runs),
        "sim_min":         int(global_sim_min),
        "sim_mult":        int(global_sim_mult),
        "active_agents":   global_active_agents,
        "algorithm":       global_algorithm,
        "lr":              global_lr,
        "gamma":           global_gamma,
        "epsilon":         global_epsilon,
        "v_epsilon":       global_v_epsilon,
        "checked":         _chks,
    }
    st.session_state.stock_use_fallback = "ALL"
    st.session_state.stocks_reverted    = set()

with st.container():
    st.markdown('<span class="sticky-header-marker"></span>', unsafe_allow_html=True)
    st.markdown('<h1 class="sticky-main-title">Chainers Master Fund: Performance Monitoring Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="sticky-divider">', unsafe_allow_html=True)
    st.markdown('<h2 class="sticky-sub-title">Master Fund Portfolio Report</h2>', unsafe_allow_html=True)
    st.markdown('<hr class="sticky-divider">', unsafe_allow_html=True)

# 차트+테이블 영역: sticky 컨테이너 밖, st.container()로 항상 안정 렌더
summary_placeholder = st.container()

# ==========================================
# 2. 통합 대시보드 (Alpha 비교 + 팀 펀드 에쿼티)
# ==========================================
def draw_top_dashboard(final_contribs, container, member_traces_snap=None, is_updating=False):
    df_contrib = pd.DataFrame(final_contribs)
    if df_contrib.empty:
        return {}

    df_contrib = df_contrib.sort_values(by="Member").reset_index(drop=True)
    distinct_colors = px.colors.qualitative.Plotly
    df_contrib['Unique_Color'] = [distinct_colors[i % len(distinct_colors)] for i in range(len(df_contrib))]

    # 도넛 범례: "Member N (종목명)", 바 차트 x축: "Member N<br>(종목명)"
    if 'Stocks' in df_contrib.columns:
        df_contrib['Member_Donut_Label'] = df_contrib.apply(
            lambda r: f"{r['Member']} ({r['Stocks']})"
            if str(r.get('Stocks', '-')) not in ('', '-') else r['Member'], axis=1
        )
        df_contrib['Member_Bar_Label'] = df_contrib.apply(
            lambda r: f"{r['Member']}<br>({r['Stocks']})"
            if str(r.get('Stocks', '-')) not in ('', '-') else r['Member'], axis=1
        )
    else:
        df_contrib['Member_Donut_Label'] = df_contrib['Member']
        df_contrib['Member_Bar_Label']   = df_contrib['Member']

    total_fund_capital = df_contrib['Final_Capital'].sum()

    # (1) 도넛 그래프
    fig_donut = go.Figure(go.Pie(
        labels=df_contrib['Member_Donut_Label'], values=df_contrib['Final_Capital'], hole=0.6,
        marker=dict(colors=df_contrib['Unique_Color']), textinfo="percent",
        texttemplate="<b>%{percent}</b><br><b>%{value:.2f}$</b>",
        textfont=dict(size=15), sort=False
    ))
    title_text = "<b>Master Fund Contribution (STATIC)</b>"
    if is_updating:
        title_text += " <span style='color:#ff9800;'>(Updating...)</span>"

    fig_donut.update_layout(
        title=title_text, height=350, margin=dict(l=0, r=0, t=40, b=0),
        annotations=[dict(
            text=f"Total Capital<br><b>{total_fund_capital:.2f} $</b>",
            x=0.5, y=0.5, font_size=18, showarrow=False
        )],
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=-0.4, traceorder="normal",
                    bgcolor='rgba(0,0,0,0)')
    )

    # (2) 수익 바 차트 (Vanilla vs STATIC 비교)
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Bar(
        x=df_contrib['Member_Bar_Label'], y=df_contrib['Vanilla_Profit'],
        orientation='v',
        name="Vanilla RL", marker_color="#ff4b4b", opacity=0.7,
        text=df_contrib['Vanilla_Profit'].apply(lambda x: f"<b>{x:.2f}$</b>"), textposition='outside'
    ))
    fig_profit.add_trace(go.Bar(
        x=df_contrib['Member_Bar_Label'], y=df_contrib['Profit_Dollar'],
        orientation='v',
        name="STATIC RL", marker_color="#2196f3",
        text=df_contrib['Profit_Dollar'].apply(lambda x: f"<b>{x:.2f}$</b>"), textposition='outside'
    ))
    _profit_vals = list(df_contrib['Vanilla_Profit']) + list(df_contrib['Profit_Dollar'])
    _ymax = max(_profit_vals)
    _ymin = min(_profit_vals)
    _ypad = (_ymax - _ymin) * 0.28  # 라벨과 제목 사이 여백 확보
    fig_profit.update_layout(
        title="<b>Profit Comparison ($): Vanilla vs STATIC</b>", barmode='group',
        height=380, margin=dict(l=0, r=0, t=75, b=60),
        yaxis=dict(title="<b>Profit ($)</b>", range=[_ymin - _ypad, _ymax + _ypad]),
        xaxis=dict(title="", tickangle=-30, automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5)
    )

    # (3) 성과 테이블
    table_data = []
    current_summary = {}

    for _, row in df_contrib.iterrows():
        m_name, c_ret, v_ret = row['Member'], row['Avg_Return'], row['Vanilla_Return']
        current_summary[m_name] = {'return': c_ret}
        delta = c_ret - v_ret
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        _stocks_no_trace = row.get('Stocks', '-') if 'Stocks' in row.index else '-'
        table_data.append({
            "Member": m_name, "Stocks": _stocks_no_trace,
            "Persona": row['CTPT_Code'], "Capital ($)": f"{row['Final_Capital']:.2f}$",
            "STATIC (%)": f"{c_ret:.2f}", "Vanilla (%)": f"{v_ret:.2f}",
            "Alpha (Gap)": f"{delta_str}", "STATIC MDD": f"{row['Avg_MDD']:.2f}%"
        })

    def color_negative_red(val):
        if isinstance(val, str) and val.strip().startswith('-'):
            return 'color: #FF4B4B; font-weight: bold;'
        return ''

    styled_table = (
        pd.DataFrame(table_data).style
        .map(color_negative_red)
        .set_properties(**{"text-align": "right"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "right")]}])
    )

    # [핵심 수정] is_updating 값에 따라 고유 key suffix를 부여해 DuplicateElementId 방지
    key_suffix = "upd" if is_updating else "fin"

    with container:
        has_traces = bool(member_traces_snap and len(member_traces_snap) >= 1)

        # ── Softmax 계산 (traces 있을 때) ──
        if has_traces:
            member_names_sorted = sorted(member_traces_snap.keys())
            scores_map = {row['Member']: row['Avg_Return'] / (1.0 + abs(row['Avg_MDD']))
                          for _, row in df_contrib.iterrows()}
            scores_arr = np.array([scores_map.get(mn, 0.0) for mn in member_names_sorted], dtype=float)
            # [P1] session_state의 온도·비중상한 사용 (구: temperature=1.0 하드코딩)
            _T   = float(st.session_state.get('fund_temperature', 1.0))
            _cap = float(st.session_state.get('fund_max_weight',  1.0))
            weights_arr = calculate_softmax_weights(scores_arr, temperature=_T)
            # [P1] Weight Cap: 반복 수렴 클리핑 (단일 패스는 재정규화 후 캡 재초과 가능)
            # 예) M3=59.6% → 40% 클립 후 재정규화 시 49.7%로 재초과 → 반복 필요
            if _cap < 1.0:
                for _ in range(20):
                    weights_arr = np.minimum(weights_arr, _cap)
                    _wsum = weights_arr.sum()
                    if _wsum > 0:
                        weights_arr /= _wsum
                    if np.all(weights_arr <= _cap + 1e-9):
                        break

            traces_list = [member_traces_snap[mn]['s_trace'] for mn in member_names_sorted]
            min_len = min(len(t) for t in traces_list)
            aligned    = np.array([t[:min_len] for t in traces_list])
            team_curve = np.dot(weights_arr, aligned)
            tf_final   = float(team_curve[-1]) if len(team_curve) > 0 else 0.0
            first_mn   = member_names_sorted[0]
            dates_ref  = list(member_traces_snap[first_mn]['dates'])[:min_len]

        # ── Row 1: donut | bar | All Members 차트 (항상 3열) ──
        col1, col2, col3 = st.columns([1, 1.1, 1.6])

        with col1:
            st.plotly_chart(fig_donut, use_container_width=True, key=f"top_donut_{key_suffix}")
        with col2:
            st.plotly_chart(fig_profit, use_container_width=True, key=f"top_profit_{key_suffix}")

        if has_traces:
            with col3:
                fig_members = go.Figure()
                for i, mn in enumerate(member_names_sorted):
                    color = distinct_colors[i % len(distinct_colors)]
                    t = member_traces_snap[mn]['s_trace'][:min_len]
                    stocks_label = ", ".join(member_traces_snap[mn].get('stocks', []))
                    fig_members.add_trace(go.Scatter(
                        x=dates_ref, y=t, mode='lines',
                        name=f'<b>{mn}</b>' + (f' ({stocks_label})' if stocks_label else ''),
                        line=dict(color=color, width=2)
                    ))
                fig_members.add_trace(go.Scatter(
                    x=dates_ref, y=team_curve, mode='lines',
                    name='<b>Team Fund (Softmax)</b>',
                    line=dict(color='#ffffff', width=3.5, dash='solid')
                ))
                fig_members.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
                tf_color = "#4a90d9" if tf_final >= 0 else "#ff4b4b"
                # Team Fund 라벨: 제목과 그래프 최상단 사이 우측
                fig_members.add_annotation(
                    x=0.98, y=1.0, xref="paper", yref="paper",
                    text=f"<b>Team Fund: {tf_final:+.2f}%</b>",
                    showarrow=False, font=dict(color=tf_color, size=13),
                    bgcolor='rgba(20,20,35,0.75)',
                    bordercolor=tf_color, borderwidth=1, borderpad=5,
                    align="right", xanchor="right", yanchor="bottom"
                )
                # All Members Y축 범위: 팀펀드 포함 전체 데이터 기준
                _am_vals = list(team_curve)
                for _mn in member_names_sorted:
                    _am_vals += list(member_traces_snap[_mn]['s_trace'][:min_len])
                _am_ymin = min(0.0, float(np.min(_am_vals))) if _am_vals else 0.0
                _am_ymax = max(float(np.max(_am_vals)), 1.0) * 1.12 if _am_vals else 10.0

                fig_members.update_layout(
                    title=dict(text="<b>All Members: STATIC RL Cumulative Returns + Team Fund</b>",
                               font=dict(size=14)),
                    xaxis=dict(title="<b>Trading Days</b>", showgrid=True),
                    yaxis=dict(title="<b>Cumulative Return (%)</b>", showgrid=True,
                               range=[_am_ymin, _am_ymax]),
                    # 범례: 그래프 내부 좌측 상단
                    legend=dict(font=dict(size=10), orientation="v",
                                yanchor="top", y=0.99, xanchor="left", x=0.01,
                                bgcolor='rgba(20,20,35,0.75)',
                                bordercolor='rgba(128,128,128,0.3)', borderwidth=1),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=370, margin=dict(t=45, b=15, l=55, r=15)
                )
                st.plotly_chart(fig_members, use_container_width=True,
                                key=f"team_fund_chart_{key_suffix}")

        # ── Row 2: 테이블 (Portfolio Alpha + Softmax 비중 통합 또는 기본) ──
        st.markdown("---")
        st.markdown("#### Portfolio Alpha Strategy Report")

        if has_traces:
            merged_rows = []
            for _, row in df_contrib.iterrows():
                m_name = row['Member']
                c_ret, v_ret = row['Avg_Return'], row['Vanilla_Return']
                delta = c_ret - v_ret
                delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
                idx = member_names_sorted.index(m_name) if m_name in member_names_sorted else -1
                score_str  = f"{scores_arr[idx]:.3f}" if idx >= 0 else "-"
                weight_str = f"{weights_arr[idx]*100:.1f}%" if idx >= 0 else "-"
                stocks_str = ", ".join(member_traces_snap[m_name].get('stocks', [])) if m_name in member_traces_snap else "-"
                merged_rows.append({
                    "Member": m_name, "Stocks": stocks_str,
                    "Persona": row['CTPT_Code'],
                    "Capital ($)": f"{row['Final_Capital']:.2f}$",
                    "STATIC (%)": f"{c_ret:.2f}", "Vanilla (%)": f"{v_ret:.2f}",
                    "Alpha (Gap)": delta_str, "MDD": f"{row['Avg_MDD']:.2f}%",
                    "Score": score_str, "Weight %": weight_str,
                })

            def color_neg(val):
                if isinstance(val, str) and val.strip().startswith('-'):
                    return 'color: #FF4B4B; font-weight: bold;'
                return ''

            st.dataframe(
                pd.DataFrame(merged_rows).style
                .map(color_neg)
                .set_properties(**{"text-align": "right"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "right")]}]),
                use_container_width=True, hide_index=True
            )
        else:
            # traces 없을 때 col3에 안내 메시지
            with col3:
                st.markdown(
                    "<div style='height:340px;display:flex;align-items:center;justify-content:center;"
                    "border:1px dashed rgba(120,120,140,0.3);border-radius:8px;"
                    "color:rgba(160,160,180,0.6);font-size:13px;text-align:center;'>"
                    "▶ Run Evaluation 또는<br>Eval. All 실행 후<br>All Members 차트가 표시됩니다</div>",
                    unsafe_allow_html=True
                )
            st.dataframe(styled_table, use_container_width=True, hide_index=True)

    return current_summary


# ==========================================
# 3. 시뮬레이션 및 차트 생성
# ==========================================
def _make_cumulative_fig(stock_name, df, v_trace, s_trace, real_ret_trace,
                          opt_v_trace=None, opt_s_trace=None, algo_name="STATIC RL"):
    """누적 수익률 비교 차트. Ghost Line (Optimal) 파라미터 선택적 표시."""
    height     = 400
    title_size = 18
    axis_size  = 14
    legend_size = 12

    fig = go.Figure()

    # ── Ghost Lines 먼저 (배경) — 메인 라인보다 아래 레이어 ──
    if opt_v_trace is not None and len(opt_v_trace) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[:len(opt_v_trace)], y=opt_v_trace,
            mode='lines', name='<b>Vanilla RL (Optimal ✦)</b>',
            line=dict(color='#e05050', width=1.2, dash='dash'),
            opacity=0.35
        ))
    if opt_s_trace is not None and len(opt_s_trace) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[:len(opt_s_trace)], y=opt_s_trace,
            mode='lines', name=f'<b>{algo_name} (Optimal ✦)</b>',
            line=dict(color='#4a90d9', width=1.2, dash='dash'),
            opacity=0.35
        ))

    # ── 메인 라인 나중에 (전경) — Ghost 위에 표시 ──
    fig.add_trace(go.Scatter(
        x=list(df.index), y=list(v_trace), mode='lines', name='<b>Vanilla RL</b>',
        line=dict(color='#e05050', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=list(df.index), y=list(s_trace), mode='lines', name=f'<b>{algo_name}</b>',
        line=dict(color='#4a90d9', width=3, dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=list(df.index), y=list(real_ret_trace), mode='lines', name='<b>Market (Buy&Hold)</b>',
        line=dict(color='#2ecc71', width=1.5, dash='dot')
    ))

    # Y축 범위: 메인 데이터 기준으로만 산출 (Ghost Line 제외 — 극단값 왜곡 방지)
    _all_vals = list(s_trace) + list(v_trace) + list(real_ret_trace)
    _y_min = min(0.0, float(np.min(_all_vals))) if _all_vals else 0.0
    _y_max = max(float(np.max(_all_vals)), 1.0) * 1.15 if _all_vals else 10.0

    fig.update_layout(
        title=dict(text=f"<b>Cumulative Return Comparison ({stock_name})</b>", font=dict(size=title_size)),
        xaxis=dict(title=dict(text="<b>Trading Days</b>", font=dict(size=axis_size)), showgrid=True),
        yaxis=dict(title=dict(text="<b>Total Cumulative Return (%)</b>", font=dict(size=axis_size)),
                   showgrid=True, range=[_y_min, _y_max]),
        legend=dict(font=dict(size=legend_size), x=0.01, y=0.99,
                    bgcolor='rgba(128,128,128,0.15)', bordercolor='rgba(128,128,128,0.3)', borderwidth=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=height, margin=dict(t=50, b=50, l=60, r=30)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def _make_trend_fig(df_h):
    """Trial-by-Trial Return Progression & Stability"""
    v_mean = df_h['Vanilla Final (%)'].mean()
    v_max  = df_h['Vanilla Final (%)'].max()
    v_min  = df_h['Vanilla Final (%)'].min()
    s_mean = df_h['STATIC Final (%)'].mean()
    s_max  = df_h['STATIC Final (%)'].max()
    s_min  = df_h['STATIC Final (%)'].min()

    _t_min = int(df_h['Trial'].min())
    _t_max = int(df_h['Trial'].max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['Vanilla Final (%)'],
        mode='lines+markers', name='<b>Vanilla Return</b>',
        line=dict(color='#e05050', width=2),
        marker=dict(size=10, symbol='circle', color='#e05050',
                    line=dict(color='white', width=1.5))))
    fig.add_trace(go.Scatter(x=df_h['Trial'], y=df_h['STATIC Final (%)'],
        mode='lines+markers', name='<b>STATIC Return (Ours)</b>',
        line=dict(color='#4a90d9', width=2),
        marker=dict(size=10, symbol='square', color='#4a90d9',
                    line=dict(color='white', width=1.5))))

    for y, dash, color, label, pos in [
        (v_mean, "solid",  "#e05050", "Vanilla Mean", "top right"),
        (v_max,  "dot",    "#e05050", "Vanilla Max",  "top right"),
        (v_min,  "dot",    "#e05050", "Vanilla Min",  "bottom right"),
        (s_mean, "solid",  "#4a90d9", "STATIC Mean",  "top left"),
        (s_max,  "dot",    "#4a90d9", "STATIC Max",   "top left"),
        (s_min,  "dot",    "#4a90d9", "STATIC Min",   "bottom left"),
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position=pos)

    fig.update_layout(
        title=dict(text="<b>Trial-by-Trial Return Progression & Stability</b>",
                   font=dict(size=20, family="Arial Black")),
        xaxis=dict(title="<b>Trial Number</b>", tickmode='linear', dtick=1,
                   range=[_t_min - 0.5, _t_max + 0.5]),
        yaxis=dict(title="<b>Final Return (%)</b>"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=320, margin=dict(t=45, b=25, l=40, r=80)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def _make_trial_box_fig(df_h):
    """Return Distribution across Trials"""
    v_mean     = df_h['Vanilla Final (%)'].mean()
    s_mean     = df_h['STATIC Final (%)'].mean()
    med_v      = df_h['Vanilla Final (%)'].median()
    med_s      = df_h['STATIC Final (%)'].median()
    avg_market = df_h['Market Final (%)'].mean()

    _n = len(df_h)
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df_h['Vanilla Final (%)'],
        x=['Vanilla RL'] * _n,
        name='<b>Vanilla RL</b>', line=dict(color='#e05050', width=2),
        fillcolor='rgba(224,80,80,0.20)', boxmean=True, width=0.4,
        boxpoints='all', jitter=0.4, pointpos=0,
        marker=dict(color='#e05050', size=9, opacity=0.9,
                    line=dict(color='white', width=1.5))))
    fig.add_trace(go.Box(
        y=df_h['STATIC Final (%)'],
        x=['STATIC RL (Ours)'] * _n,
        name='<b>STATIC RL (Ours)</b>', line=dict(color='#4a90d9', width=2),
        fillcolor='rgba(74,144,217,0.20)', boxmean=True, width=0.4,
        boxpoints='all', jitter=0.4, pointpos=0,
        marker=dict(color='#4a90d9', size=9, opacity=0.9,
                    line=dict(color='white', width=1.5))))

    fig.add_annotation(x='Vanilla RL', y=v_mean, text=f"<b>Mean: {v_mean:.2f}%</b>",
        showarrow=False, xshift=-50, yshift=8, xanchor='right',
        font=dict(color='#e05050', size=13, family="Arial Black"))
    fig.add_annotation(x='Vanilla RL', y=med_v, text=f"<b>Median: {med_v:.2f}%</b>",
        showarrow=False, xshift=-50, yshift=-8, xanchor='right',
        font=dict(color='#e05050', size=13, family="Arial Black"))
    fig.add_annotation(x='STATIC RL (Ours)', y=med_s, text=f"<b>Median: {med_s:.2f}%</b>",
        showarrow=False, xshift=50, yshift=8, xanchor='left',
        font=dict(color='#4a90d9', size=13, family="Arial Black"))
    fig.add_annotation(x='STATIC RL (Ours)', y=s_mean, text=f"<b>Mean: {s_mean:.2f}%</b>",
        showarrow=False, xshift=50, yshift=-8, xanchor='left',
        font=dict(color='#4a90d9', size=13, family="Arial Black"))

    fig.add_hline(y=avg_market, line_width=2.5, line_dash="dot", line_color="green")
    fig.add_annotation(x='Vanilla RL', xref="x", y=avg_market,
        text=f"<b>Market (Buy&Hold) {avg_market:.2f}%</b>",
        showarrow=False, yshift=18, xanchor='left',
        font=dict(color="green", size=13, family="Arial Black"), bgcolor="rgba(0,0,0,0)")

    _box_vals = (list(df_h['Vanilla Final (%)']) + list(df_h['STATIC Final (%)'])
                 + [avg_market, v_mean, s_mean, med_v, med_s])
    _by_max = max(_box_vals)
    _by_min = min(_box_vals)
    _by_pad = (_by_max - _by_min) * 0.35   # 제목과 최고점 사이 여백 확보
    fig.update_layout(
        title=dict(text="<b>Return Distribution across Trials</b>",
                   font=dict(size=22, family="Arial Black")),
        yaxis=dict(
            title=dict(text="<b>Final Return (%)</b>", font=dict(size=18, family="Arial Black")),
            range=[_by_min - _by_pad, _by_max + _by_pad],
        ),
        xaxis=dict(
            title=dict(text="<b>Performance Metrics</b>", font=dict(size=18, family="Arial Black")),
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=510, margin=dict(t=100, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5)
    )
    fig.add_hline(y=0, line_width=2, line_color="rgba(150,150,150,0.8)")
    return fig


def get_rl_data(ticker, lr, gamma, epsilon, n_bars, train_episodes, seed, v_epsilon=None, fee_rate=0.0, interval="1d",
                use_vol=False, roll_period=None, algorithm="STATIC"):
    """시뮬레이션을 1회만 실행하여 원시 데이터 + 일별 행동 로그 + 학습된 정책을 반환.

    Parameters (신규)
    ─────────────────
    use_vol     : [P3] True이면 8상태 변동성 신호 활성화 (Rolling_Std 컬럼 필요).
    roll_period : [P4] OOS 구간 재학습 주기 (봉 수). None이면 기존 동작.
    algorithm   : 'STATIC' | 'A2C' | 'A3C' | 'PPO' | 'SAC' | 'DDPG'
                  STATIC → 기존 tabular Actor-Critic / 나머지 → 신경망 RL

    Returns (9개 — 기존 7개 + s_theta, v_qtable)
    ──────────────────────────────────────────────
    df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable
      s_theta  : STATIC RL theta 또는 TinyMLP actor (신경망 알고리즘 시 TinyMLP).
      v_qtable : Vanilla RL Q-table (2×2). 항상 반환.
    """
    df_full = fetch_stock_data(ticker, interval=interval)
    if df_full.empty or len(df_full) < 10:
        return None, None, None, None, 0.0, [], [], None, None
    df = df_full.tail(n_bars).copy()
    real_ret_trace = (df['Close'] / df['Close'].iloc[0] - 1) * 100
    _v_eps       = v_epsilon if v_epsilon is not None else epsilon
    _v_train_epi = max(train_episodes * 2, 200)  # improve 4-3: Vanilla 2× 학습

    # [P2] Vanilla RL은 항상 실행 (비교 기준선)
    v_trace, v_log, v_qtable = run_rl_simulation_with_log(
        df, lr, gamma, _v_eps, episodes=_v_train_epi,
        use_static=False, seed=seed, fee_rate=fee_rate,
        return_policy=True
    )

    if algorithm == "STATIC":
        # [P3] 변동성 신호: use_vol=True이면 df의 Rolling_Std 컬럼 자동 전달
        _vols_arr = df['Rolling_Std'].values if (use_vol and 'Rolling_Std' in df.columns) else None
        # [P2] return_policy=True → theta 반환 (Explainable RL 시각화용)
        s_trace, s_log, s_theta = run_rl_simulation_with_log(
            df, lr, gamma, epsilon, episodes=train_episodes,
            use_static=True, seed=seed, fee_rate=fee_rate,
            vols=_vols_arr, vol_threshold=None,
            roll_period=roll_period,
            return_policy=True
        )
    else:
        # 신경망 RL 알고리즘 (A2C / A3C / PPO / SAC / DDPG)
        s_trace, s_log, s_theta = run_neural_rl(
            df, lr=lr, gamma=gamma, epsilon=epsilon,
            episodes=train_episodes, algorithm=algorithm,
            seed=seed, fee_rate=fee_rate
        )

    s_mdd = calculate_mdd(s_trace)
    return df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable

# --- 포트폴리오 리포트: 항상 렌더 (st.container() 기반) ---
# st.container()는 st.empty()와 달리 복잡한 레이아웃(columns 등)을 안정적으로 표시
_prev_contribs = st.session_state.prev_final_contributions
_prev_traces   = st.session_state.member_traces
_is_updating   = bool(st.session_state.run_all_queue or st.session_state.sim_all_queue)

if _prev_contribs:
    draw_top_dashboard(
        _prev_contribs,
        summary_placeholder,
        member_traces_snap=_prev_traces if _prev_traces else None,
        is_updating=_is_updating,
    )
else:
    with summary_placeholder:
        st.info("ℹ️ Run Evaluation을 실행하면 포트폴리오 차트와 테이블이 여기에 표시됩니다.")

# --- 모듈 로드 ---
members_dir = os.path.join(root_path, "members")
team_modules = []
for item in sorted(os.listdir(members_dir)):
    if item.startswith("member_") and os.path.isdir(os.path.join(members_dir, item)):
        try:
            team_modules.append(importlib.import_module(f"members.{item}.config"))
        except Exception as e:
            st.sidebar.warning(f"⚠️ {item} 로드 실패: {e}")

sorted_modules = sorted(team_modules, key=lambda m: getattr(m, "MEMBER_NAME", m.__name__))
all_stock_names = {idx: info["name"] for idx, info in STOCK_REGISTRY.items()}
name_to_index = {info["name"]: idx for idx, info in STOCK_REGISTRY.items()}

# ── Run Eval. All / Simul. All 버튼 클릭 시 큐 채우기 ──
def _build_all_queue():
    """현재 선택된 전체 멤버·종목 목록으로 큐 생성"""
    _q = []
    for _mc in sorted_modules:
        _mn = getattr(_mc, 'MEMBER_NAME', _mc.__name__)
        _default = [all_stock_names[i] for i in getattr(_mc, 'TARGET_INDICES', []) if i in all_stock_names]
        _sel = st.session_state.get(f"ms_{_mn}", _default)
        for _sn in _sel:
            _q.append((_mn, _sn))
    return _q

if run_eval_all_btn:
    st.session_state.run_all_queue = _build_all_queue()
if sim_all_btn:
    st.session_state.sim_all_queue = _build_all_queue()
    st.session_state.sim_auto_save = True   # 자동 저장 모드 활성화

# ── All 적용: 모든 멤버·종목 슬라이더 키 일괄 업데이트 ──
_ALL_INTERVALS = ["15m", "1h", "1d", "1wk", "1mo"]
_ALL_CHK_KEYS  = ["timeframe","episodes","train_epi","frame","seed","auto","active","algo","lr","gamma","eps","v_eps","sim_min","sim_mult"]
if apply_all_clicked and st.session_state.fallback_params:
    _fp   = st.session_state.fallback_params
    _chks = _fp.get("checked", {k: True for k in _ALL_CHK_KEYS})
    _new_tf = _fp.get("timeframe", "1 week")
    _new_iv = _fp.get("interval", "1wk")
    # 이전 상태 스냅샷 저장 (체크된 파라미터만)
    _prev = {}
    for _mc in sorted_modules:
        _mn = getattr(_mc, 'MEMBER_NAME', _mc.__name__)
        _def_stks = [all_stock_names[i] for i in getattr(_mc, 'TARGET_INDICES', []) if i in all_stock_names]
        _sel_stks = st.session_state.get(f"ms_{_mn}", _def_stks)
        for _sn in _sel_stks:
            _snap = []
            if _chks.get("timeframe"): _snap.append(f"tf_{_mn}_{_sn}")
            if _chks.get("frame"):     _snap.append(f"fspd_{_mn}_{_sn}")
            if _chks.get("seed"):      _snap.append(f"seed_{_mn}_{_sn}")
            if _chks.get("auto"):      _snap.append(f"autoruns_{_mn}_{_sn}")
            if _chks.get("active"):    _snap.append(f"active_{_mn}_{_sn}")
            if _chks.get("lr"):        _snap.append(f"lr_{_mn}_{_sn}")
            if _chks.get("gamma"):     _snap.append(f"gamma_{_mn}_{_sn}")
            if _chks.get("eps"):       _snap.append(f"eps_{_mn}_{_sn}")
            if _chks.get("v_eps"):     _snap.append(f"v_eps_{_mn}_{_sn}")
            for _k in _snap:
                if _k in st.session_state:
                    _prev[_k] = st.session_state[_k]
            if _chks.get("episodes"):
                for _iv in _ALL_INTERVALS:
                    _ek = f"epi_{_mn}_{_sn}_{_iv}"
                    if _ek in st.session_state:
                        _prev[_ek] = st.session_state[_ek]
            if _chks.get("train_epi"):
                _tk = f"train_epi_{_mn}_{_sn}"
                if _tk in st.session_state:
                    _prev[_tk] = st.session_state[_tk]
    st.session_state.fallback_prev_state = _prev
    # 새 값 일괄 적용 (체크된 파라미터만)
    for _mc in sorted_modules:
        _mn = getattr(_mc, 'MEMBER_NAME', _mc.__name__)
        _def_stks = [all_stock_names[i] for i in getattr(_mc, 'TARGET_INDICES', []) if i in all_stock_names]
        _sel_stks = st.session_state.get(f"ms_{_mn}", _def_stks)
        for _sn in _sel_stks:
            if _chks.get("timeframe"):
                st.session_state[f"tf_{_mn}_{_sn}"]            = _new_tf
            if _chks.get("episodes"):
                st.session_state[f"epi_{_mn}_{_sn}_{_new_iv}"] = _fp["episodes"]
            if _chks.get("train_epi"):
                st.session_state[f"train_epi_{_mn}_{_sn}"] = _fp["train_episodes"]
            if _chks.get("frame"):
                st.session_state[f"fspd_{_mn}_{_sn}"]          = _fp["frame_speed"]
            if _chks.get("seed"):
                st.session_state[f"seed_{_mn}_{_sn}"]          = _fp["seed"]
            if _chks.get("auto"):
                st.session_state[f"autoruns_{_mn}_{_sn}"]      = _fp["auto_runs"]
            if _chks.get("active"):
                st.session_state[f"active_{_mn}_{_sn}"]        = _fp.get("active_agents", ["Vanilla RL", "STATIC RL"])
            if _chks.get("lr"):
                st.session_state[f"lr_{_mn}_{_sn}"]            = _fp["lr"]
            if _chks.get("gamma"):
                st.session_state[f"gamma_{_mn}_{_sn}"]         = _fp["gamma"]
            if _chks.get("eps"):
                st.session_state[f"eps_{_mn}_{_sn}"]           = _fp["epsilon"]
            if _chks.get("v_eps"):
                st.session_state[f"v_eps_{_mn}_{_sn}"]         = _fp["v_epsilon"]
    st.rerun()

# ── 되돌리기: 이전 상태 복원 ──
if revert_all_clicked:
    _prev = st.session_state.get('fallback_prev_state', {})
    for _k, _v in _prev.items():
        st.session_state[_k] = _v
    st.session_state.fallback_prev_state = {}
    st.session_state.stock_use_fallback  = None
    st.session_state.stocks_reverted     = set()
    st.rerun()

total_charts = sum(
    len(st.session_state.get(
        f"ms_{getattr(m, 'MEMBER_NAME', m.__name__)}",
        [all_stock_names[idx] for idx in getattr(m, 'TARGET_INDICES', [])]
    ))
    for m in sorted_modules
)

final_contributions = []
total_episodes_run = 0
rendered_count = 0
_gauge_loading_set = False  # 로딩 표시 중복 방지 플래그

# 멤버별 에쿼티 곡선 누적 버퍼 (팀 펀드 합성용)
_member_trace_buf = {}   # member_name → {'traces': [], 'dates': index, 'stock_names': []}

_pct0 = st.session_state.master_pbar_pct
_render_master_pbar_html(_pct0, master_progress_placeholder)
master_pbar = master_progress_placeholder

st.markdown("### Portfolio Managers (Independent RL Labs)")

for m_config in sorted_modules:
    m_name = getattr(m_config, "MEMBER_NAME", "Unknown")
    with st.container():
        st.subheader(f"{m_name}")
        m_params = getattr(m_config, "RL_PARAMS", {})
        default_p = m_params.get("default", {})
        ctpt_code, ctpt_desc, ctpt_color = calculate_ctpt_and_color(
            default_p.get("lr", global_lr),
            default_p.get("gamma", global_gamma),
            default_p.get("epsilon", global_epsilon)
        )
        _all_personas = [
            ("PSR", "#607d8b", "보수형"),
            ("PSV", "#ff9800", "탐색형"),
            ("PLR", "#3f51b5", "신중한 장기형"),
            ("PLV", "#e91e63", "유연한 장기형"),
            ("ASR", "#f44336", "단기 민첩형"),
            ("ASV", "#ffc107", "단기 모험형"),
            ("ALR", "#4caf50", "안정적 성장형"),
            ("ALV", "#2196f3", "적응형 모험가"),
        ]
        _badges = ""
        for _code, _color, _desc in _all_personas:
            if _code == ctpt_code:
                _badges += (
                    f"<span style='background:{_color}; color:#fff; border-radius:5px; "
                    f"padding:3px 11px; font-size:0.85em; font-weight:bold; "
                    f"box-shadow:0 0 8px {_color}99; margin:2px;'>"
                    f"{_code}&nbsp;{_desc}</span>"
                )
            else:
                _badges += (
                    f"<span style='background:{_color}18; color:{_color}; "
                    f"border:1px solid {_color}55; border-radius:5px; "
                    f"padding:3px 8px; font-size:0.78em; opacity:0.55; margin:2px;'>"
                    f"{_code}</span>"
                )
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:3px; flex-wrap:wrap; margin-bottom:2px;'>"
            f"<small style='margin-right:4px;'><b>Persona</b></small>{_badges}</div>",
            unsafe_allow_html=True
        )

        selected_stock_names = st.multiselect(
            f"Stocks - {m_name}",
            options=list(all_stock_names.values()),
            default=[all_stock_names[idx] for idx in getattr(m_config, 'TARGET_INDICES', [])],
            key=f"ms_{m_name}"
        )

        mem_s_rets, mem_v_rets, mem_mdds = [], [], []
        if selected_stock_names:
            for stock_name in selected_stock_names:
                ticker    = get_ticker_by_name(stock_name)
                fee_info  = get_fee_info(ticker)
                fee_rate  = fee_info["buy"] + fee_info["sell"]
                stock_idx = name_to_index.get(stock_name)
                p_settings = m_params.get(stock_idx, m_params.get("default", {}))
                hist_key = f"{m_name}_{stock_name}"
                # [P3/P4] 멤버별 기본값(config.py) + 전역 session_state 오버라이드
                # use_vol  : 전역 ON이면 강제 True / OFF이면 config 값 사용
                # roll_period: 전역 roll_period_active ON이면 전역 값 / OFF이면 config 값 사용
                _use_vol_now = (bool(st.session_state.get('use_vol_feature', False))
                                or bool(p_settings.get('use_vol', False)))
                _roll_period_now = (int(st.session_state.get('roll_period_val', 20))
                                    if st.session_state.get('roll_period_active', False)
                                    else p_settings.get('roll_period', None))

                # ── Simulation pending: 슬라이더 렌더링 전에 키 사전 설정 ──
                _sim_pend_key = f"sim_pending_{hist_key}"
                if _sim_pend_key in st.session_state:
                    _pend = st.session_state.pop(_sim_pend_key)
                    st.session_state[f"lr_{m_name}_{stock_name}"]    = _pend["lr"]
                    st.session_state[f"gamma_{m_name}_{stock_name}"] = _pend["gamma"]
                    st.session_state[f"eps_{m_name}_{stock_name}"]   = _pend["epsilon"]
                    st.session_state[f"v_eps_{m_name}_{stock_name}"] = _pend["v_epsilon"]
                    if "algorithm" in _pend:
                        _pend_algo = _pend["algorithm"]
                        if _pend_algo in ["STATIC", "A2C", "A3C", "PPO", "SAC", "DDPG"]:
                            st.session_state[f"algo_{m_name}_{stock_name}"] = _pend_algo

                # ── Simulation 완료 후 저장 확인 키 (버튼 행에서 처리) ──
                _sim_confirm_key = f"sim_confirm_{hist_key}"

                # ── 파라미터: 접힌 expander – 2행 구조 ──
                with st.expander(f"⚙️ {stock_name} Parameters  |  💸 거래 수수료 — {fee_info['label']}", expanded=False):
                    st.markdown("<small><b>System Parameters</b></small>", unsafe_allow_html=True)
                    # ── Timeframe 설정 ──
                    _tf_options  = ["15 min.", "1 hour", "1 day", "1 week", "1 month"]
                    _tf_map      = {"15 min.": "15m", "1 hour": "1h", "1 day": "1d", "1 week": "1wk", "1 month": "1mo"}
                    _tf_lbl_map  = {"15m": "Bars (15min)", "1h": "Bars (1h)", "1d": "Trading Days", "1wk": "Trading Weeks", "1mo": "Trading Months"}
                    _tf_min_map  = {"15m": 20, "1h": 20, "1d": 10, "1wk": 10, "1mo": 6}
                    _tf_max_map  = {"15m": 400, "1h": 500, "1d": 500, "1wk": 200, "1mo": 60}
                    _tf_def_map  = {"15m": 80, "1h": 120, "1d": 500, "1wk": 105, "1mo": 24}
                    sc0, sc1, sc1b, sc2, sc3, sc4, sc5 = st.columns(7)
                    with sc0:
                        _tf_sel = st.selectbox(
                            "Timeframe", _tf_options, index=2,
                            key=f"tf_{m_name}_{stock_name}",
                            help="데이터 봉 단위 선택 (15분/1시간: 최근 60일/730일 제한)"
                        )
                    l_interval = _tf_map[_tf_sel]
                    with sc1:
                        _tf_min = _tf_min_map[l_interval]
                        _tf_max = _tf_max_map[l_interval]
                        _tf_def = _tf_def_map[l_interval]
                        _epi_val = min(max(int(p_settings.get("episodes", _tf_def)), _tf_min), _tf_max)
                        l_epi = st.slider(
                            _tf_lbl_map[l_interval], _tf_min, _tf_max, _epi_val,
                            key=f"epi_{m_name}_{stock_name}_{l_interval}",
                            # [RL] 데이터 창 크기(n_bars): 학습+평가에 사용할 봉 수.
                            # 일봉 500 = 약 2년 데이터. 첫 70%(350봉)로 학습, 전체(500봉)로 평가.
                            # 너무 작으면 OOS 구간 부족, 너무 크면 시장 체제 변화 포함 위험.
                            help="시장 데이터 봉 수 (n_bars). 첫 70%=학습, 전체=평가(워크포워드). 일봉 500≈2년."
                        )
                    with sc1b:
                        _train_epi_val = int(p_settings.get("train_episodes", 300))
                        if _IS_CLOUD:
                            _train_epi_val = min(_train_epi_val, 300)  # 클라우드 부하 제한
                        l_train_epi = st.slider(
                            "Train Episodes", 10, 500, _train_epi_val,
                            key=f"train_epi_{m_name}_{stock_name}",
                            # [RL] 훈련 에피소드 수 (epoch): 같은 훈련 데이터를 반복 학습하는 횟수.
                            # 300 에피소드 × 350봉 = 105,000 TD 업데이트 step.
                            # 많을수록 정책 수렴 향상, 과적합(in-sample 최적화) 위험 증가.
                            help="RL 학습 에피소드 수 (epoch). 훈련 데이터 반복 횟수. 많을수록 수렴↑/과적합↑."
                        )
                    with sc2:
                        l_frame_speed = st.slider(
                            "Frame Speed (sec)", 0.01, 2.0,
                            0.05, step=0.01, format="%.2f",
                            key=f"fspd_{m_name}_{stock_name}",
                            help="시뮬레이션 수렴 차트 업데이트 간격(초). 낮을수록 빠른 재생."
                        )
                    with sc3:
                        l_seed = st.number_input(
                            "Base Seed",
                            value=int(p_settings.get("seed", global_seed)),
                            step=1, key=f"seed_{m_name}_{stock_name}",
                            # [RL] 훈련 재현성 시드: np.random.seed(seed)로 ε-greedy 탐험 경로 고정.
                            # 동일 seed → 동일 탐험 경로 → 동일 학습 궤적 → 동일 정책 → 동일 성과.
                            # Trial History의 각 Trial은 seed + trial_idx×37 로 독립 시드 사용.
                        )
                    with sc4:
                        l_auto_runs = st.number_input(
                            "Auto Run Count", min_value=1,
                            value=5, step=1,
                            key=f"autoruns_{m_name}_{stock_name}",
                            # [RL] Run Evaluation 자동 반복 횟수: 다양한 시드로 성과 분포 측정.
                            # trial_seed = base_seed + run_i × 37 (소수 간격으로 시드 독립성 확보).
                            # Simulation에서는 _n_eval = min(4, max(3, auto_runs//2)) 평가 시드 결정.
                            help="Run Evaluation 자동 반복 횟수. 다양한 시드로 Trial 분포 측정. Sim _n_eval = min(4, max(3, count//2))."
                        )
                    with sc5:
                        l_active_agents = st.multiselect(
                            "Active Agents",
                            options=["Vanilla RL", "STATIC RL"],
                            default=["Vanilla RL", "STATIC RL"],
                            key=f"active_{m_name}_{stock_name}",
                            help="비활성화된 에이전트는 연산 생략, 수익 0%로 표시. 단독 비교 시 사용."
                        )
                    # ─ 행 2: RL Hyperparameters ─
                    st.markdown(
                        "<small><b>RL Hyperparameters &nbsp;"
                        "<span style='color:#4a90d9;'>STATIC RL</span>: α / γ / ε(S) &nbsp;|&nbsp; "
                        "<span style='color:#e05050;'>Vanilla RL</span>: ε(V)</b></small>",
                        unsafe_allow_html=True
                    )
                    hc0, hc1, hc2, hc3, hc4, hc5, hc6 = st.columns(7)
                    with hc0:
                        _algo_default = p_settings.get("algorithm", "STATIC")
                        l_algorithm = st.selectbox(
                            "RL Algorithm",
                            options=["STATIC", "A2C", "A3C", "PPO", "SAC", "DDPG"],
                            index=["STATIC", "A2C", "A3C", "PPO", "SAC", "DDPG"].index(
                                _algo_default if _algo_default in ["STATIC", "A2C", "A3C", "PPO", "SAC", "DDPG"] else "STATIC"
                            ),
                            key=f"algo_{m_name}_{stock_name}",
                            help="STATIC=기존 tabular Actor-Critic / 나머지=NumPy 신경망 RL"
                        )
                    with hc1:
                        l_lr = st.slider(
                            "Learning Rate (α)", 0.001, 0.1,
                            float(p_settings.get("lr", global_lr)),
                            step=0.001, format="%.3f", key=f"lr_{m_name}_{stock_name}",
                            # [RL] 학습률 α: θ += α·δ·∇logπ (Actor), V += α·δ (Critic), Q += α·δ (Q-Learning).
                            # 크면 빠른 수렴/불안정, 작으면 느린 수렴/안정. 탐색 범위: 0.005~0.10.
                        )
                    with hc2:
                        l_gamma = st.slider(
                            "Discount Factor (γ)", 0.1, 0.99,
                            float(p_settings.get("gamma", global_gamma)),
                            key=f"gamma_{m_name}_{stock_name}",
                            # [RL] 할인율 γ: V(s) = E[r₀ + γr₁ + γ²r₂ + ...].
                            # γ=0.99: 100봉 후 보상을 0.99^100≈0.37배로 할인 (장기 중시).
                            # γ=0.85: 빠른 할인 (단기 수익 집중). 권장: 0.85~0.99.
                        )
                    with hc3:
                        l_epsilon = st.slider(
                            "STATIC ε", 0.01, 0.5,
                            float(p_settings.get("epsilon", global_epsilon)),
                            key=f"eps_{m_name}_{stock_name}",
                            # [RL] STATIC RL 탐험율 ε: ε 확률로 무작위 행동, (1-ε)로 정책 행동.
                            # 상수 ε (annealing 없음). 너무 높으면 정책 수렴 방해.
                            help="STATIC RL ε-greedy 탐험율. ε=0.16 → 16% 확률 무작위 행동. 권장: 0.01~0.25."
                        )
                    with hc4:
                        l_v_epsilon = st.slider(
                            "Vanilla ε", 0.01, 0.5,
                            float(p_settings.get("v_epsilon", global_epsilon)),
                            key=f"v_eps_{m_name}_{stock_name}",
                            # [RL] Vanilla RL 탐험율 (STATIC과 독립 최적화).
                            # Vanilla는 epsilon annealing 적용: 훈련 초반 2ε → 후반 ε.
                            help="Vanilla RL ε-greedy 탐험율. STATIC과 독립 설정. 훈련 시 2ε→ε annealing 적용."
                        )
                    with hc5:
                        l_sim_min = st.number_input(
                            "Sim Min Steps", min_value=5, max_value=200,
                            value=20, step=5,
                            key=f"sim_min_{m_name}_{stock_name}",
                            # [RL] PG Optimizer 최소 탐색 step 수.
                            # n_iters = max(Sim_Min, Auto_Run × Sim_Mult).
                            # 30 step × 3 eval_seeds = 최소 90회 RL 학습+평가.
                            help="시뮬레이션 최소 탐색 step (n_iters 하한). n_iters=max(Min, AutoRun×Mult)."
                        )
                    with hc6:
                        l_sim_mult = st.number_input(
                            "Sim Step Mult.", min_value=1, max_value=30,
                            value=6, step=1,
                            key=f"sim_mult_{m_name}_{stock_name}",
                            # [RL] Auto Run Count 배수로 총 탐색 step 결정.
                            # AutoRun=6, Mult=10 → n_iters=60 step.
                            # n_iters × _n_eval = 총 RL 평가 횟수.
                            help="n_iters = max(Min Steps, Auto Run Count × Mult.). Mult=10, Count=6 → 60 step."
                        )

                # ── Run Evaluation / Simulation 버튼 + 진행률 ──
                _has_confirm = _sim_confirm_key in st.session_state
                if _has_confirm:
                    btn_col, run_prog_col = st.columns([3, 2])
                else:
                    btn_col, run_prog_col = st.columns([2, 3])
                with btn_col:
                    if _has_confirm:
                        b1, b2, b3, b4, b5 = st.columns([5, 4, 5, 4, 1])
                    else:
                        b1, b2, b5 = st.columns([5, 4, 1])
                        b3, b4 = None, None
                    with b1:
                        run_clicked = st.button(
                            "▶ Run Evaluation",
                            key=f"run_btn_{m_name}_{stock_name}",
                            type="primary",
                            use_container_width=True,
                        )
                    with b2:
                        st.markdown('<span class="sim-btn-marker"></span>', unsafe_allow_html=True)
                        sim_clicked = st.button(
                            "Simulation",
                            key=f"sim_btn_{m_name}_{stock_name}",
                            type="primary",
                            use_container_width=True,
                        )
                    if _has_confirm and b3 is not None and b4 is not None:
                        with b3:
                            _save_clicked = st.button(
                                "저장 및 반영",
                                key=f"sim_save_{m_name}_{stock_name}",
                                type="primary",
                                use_container_width=True,
                            )
                        with b4:
                            _cancel_clicked = st.button(
                                "반영 취소",
                                key=f"sim_cancel_{m_name}_{stock_name}",
                                use_container_width=True,
                            )
                        if _save_clicked:
                            _best_params = st.session_state.pop(_sim_confirm_key)
                            _save_sim_params_to_config(m_config, stock_idx, _best_params)
                            st.session_state[f"sim_pending_{hist_key}"] = _best_params
                            st.session_state.stocks_reverted.add(hist_key)
                            st.session_state[f"auto_run_{hist_key}"] = True
                            st.rerun()
                        if _cancel_clicked:
                            st.session_state.pop(_sim_confirm_key, None)
                            st.rerun()
                    with b5:
                        st.markdown('<span class="stop-btn-marker"></span>', unsafe_allow_html=True)
                        _stop_clicked = st.button(
                            "■",
                            key=f"stop_btn_{m_name}_{stock_name}",
                            use_container_width=True,
                            help="진행 중인 Eval. All / Simul. All 큐를 중단합니다",
                        )
                    if _stop_clicked:
                        st.session_state.run_all_queue = []
                        st.session_state.sim_all_queue = []
                        st.session_state.interrupt_requested = True
                run_prog_slot = run_prog_col.empty()
                if _has_confirm:
                    _pending = st.session_state.get(_sim_confirm_key, {})
                    run_prog_col.markdown(
                        f"**저장 예정 파라미터**  \n"
                        f"lr=`{_pending.get('lr', 0):.4f}` · "
                        f"γ=`{_pending.get('gamma', 0):.4f}` · "
                        f"ε=`{_pending.get('epsilon', 0):.4f}` · "
                        f"v_ε=`{_pending.get('v_epsilon', 0):.4f}`  \n"
                        f"STATIC: `{_pending.get('s_final', 0):+.2f}%` · "
                        f"Vanilla: `{_pending.get('v_final', 0):+.2f}%` · "
                        f"Gap: `{_pending.get('gap', 0):+.4f}`"
                    )

                # ── 이전 Simulation 결과 배너 ──
                if hist_key in st.session_state.sim_result:
                    sr = st.session_state.sim_result[hist_key]
                    _gap_val = sr.get("gap", -999.0)
                    _status = ("🏆 +5%p↑ 달성" if _gap_val >= 5.0
                               else "✅ 목표 달성(≥1%p)" if sr.get("found")
                               else "⚠️ 최선값")
                    st.caption(
                        f"🔍 최근 Simulation (PG Actor-Critic) — {_status}  |  "
                        f"LR={sr['lr']:.4f}  γ={sr['gamma']:.4f}  "
                        f"ε(S)={sr['epsilon']:.4f}  ε(V)={sr['v_epsilon']:.4f}  |  "
                        f"STATIC {sr['s_final']:+.2f}%  Market {sr.get('m_final', sr['v_final']):+.2f}%  "
                        f"Alpha {sr['gap']:+.2f}%p"
                    )

                # ── Simulation 후 자동 Run Evaluation 트리거 ──
                _auto_run_key = f"auto_run_{hist_key}"
                if st.session_state.get(_auto_run_key, False):
                    st.session_state[_auto_run_key] = False
                    run_clicked = True

                # ── Run Eval. All / Simul. All 큐 처리 ──
                _rq = st.session_state.run_all_queue
                if _rq and _rq[0] == (m_name, stock_name):
                    run_clicked = True
                _sq = st.session_state.sim_all_queue
                if _sq and _sq[0] == (m_name, stock_name):
                    sim_clicked = True
                    st.session_state.sim_all_queue = _sq[1:]  # 즉시 팝 (sim은 rerun 전 팝)

                # ══════════════════════════════════════════
                # Run Evaluation
                # ══════════════════════════════════════════
                if run_clicked and not st.session_state.get('interrupt_requested', False):
                    if not _gauge_loading_set:
                        update_load_bar(st.session_state.prev_episodes_run, gauge_placeholder, is_loading=True)
                        _gauge_loading_set = True
                    # 종목별 파라미터를 다시 적용: fallback 목록에서 이 종목 제거
                    st.session_state.stocks_reverted.add(hist_key)
                    trials = st.session_state.stock_trial_history.setdefault(hist_key, [])
                    n_runs = int(l_auto_runs)
                    _interrupted = False
                    _run_err_msg = None
                    try:
                        for run_i in range(n_runs):
                            if st.session_state.get('interrupt_requested', False):
                                st.session_state.interrupt_requested = False
                                run_prog_slot.warning(f"⛔ 중단됨 ({run_i}/{n_runs} 완료)")
                                _interrupted = True
                                break
                            trial_seed = int(l_seed) + (len(trials) + run_i) * 37  # improve 4-1: 소수 간격 ×37로 시드 독립성 강화
                            run_prog_slot.progress(
                                run_i / n_runs,
                                text=f"Running trial {run_i + 1} / {n_runs}  (seed={trial_seed})"
                            )
                            try:
                                _, vt, s_tr, mkt, _, _, _, _, _ = get_rl_data(
                                    ticker, l_lr, l_gamma, l_epsilon, l_epi, l_train_epi, trial_seed,
                                    v_epsilon=l_v_epsilon, fee_rate=fee_rate, interval=l_interval,
                                    use_vol=_use_vol_now, roll_period=_roll_period_now,
                                    algorithm=l_algorithm
                                )
                            except Exception as _e:
                                vt, s_tr, mkt = None, None, None
                                _run_err_msg = str(_e)[:80]
                            if vt is not None:
                                trials.append({
                                    "Trial": len(trials) + 1,
                                    "Seed": trial_seed,
                                    "Vanilla Final (%)": float(vt[-1]),
                                    "STATIC Final (%)":  float(s_tr[-1]),
                                    "Market Final (%)":  float(mkt.iloc[-1]),
                                })
                            elif not _interrupted:
                                run_prog_slot.warning(
                                    f"⚠️ Trial {run_i+1}: {ticker} 데이터 로드 실패"
                                    + (f" — {_run_err_msg}" if _run_err_msg else "")
                                )
                        if not _interrupted:
                            run_prog_slot.success(f"완료: {n_runs}회 실행 / 누적 {len(trials)}건")
                    except Exception as _outer_e:
                        run_prog_slot.error(f"Run Evaluation 오류: {str(_outer_e)[:100]}")
                    finally:
                        # 큐 팝은 항상 실행 (예외 발생 시에도 다음 멤버로 진행)
                        _rq = st.session_state.run_all_queue
                        if _rq and _rq[0] == (m_name, stock_name):
                            st.session_state.run_all_queue = _rq[1:]
                        st.rerun()

                # ══════════════════════════════════════════
                # Simulation: PG Actor-Critic 기반 파라미터 탐색
                # Policy Gradient Theorem + REINFORCE with baseline + Actor-Critic
                # ══════════════════════════════════════════
                if sim_clicked:
                    if not _gauge_loading_set:
                        update_load_bar(st.session_state.prev_episodes_run, gauge_placeholder, is_loading=True)
                        _gauge_loading_set = True

                    # fallback override for sim parameters
                    _fb_p0 = st.session_state.get("fallback_params", {})
                    _fb_c0 = _fb_p0.get("checked", {})
                    _ufb0  = (st.session_state.get("stock_use_fallback", "") == "ALL"
                              and hist_key not in st.session_state.get("stocks_reverted", set()))
                    eff_sim_min  = int(_fb_p0.get("sim_min",  l_sim_min))  if _ufb0 and _fb_c0.get("sim_min")  else int(l_sim_min)
                    eff_sim_mult = int(_fb_p0.get("sim_mult", l_sim_mult)) if _ufb0 and _fb_c0.get("sim_mult") else int(l_sim_mult)
                    n_iters = max(eff_sim_min, int(l_auto_runs) * eff_sim_mult)
                    param_bounds = {
                        "lr":        (0.005, 0.1),
                        "gamma":     (0.85,  0.99),
                        "epsilon":   (0.01,  0.25),  # improve 4-3: 상한 0.5→0.25 (경계값 고착 방지)
                        "v_epsilon": (0.01,  0.25),  # improve 4-3: 상한 0.5→0.25
                    }

                    # ── PG Actor-Critic Optimizer 초기화 ──
                    optimizer = PGActorCriticOptimizer(
                        bounds=param_bounds,
                        lr_actor=0.12,
                        sigma_init=0.18,
                        sigma_min=0.02,
                        sigma_max=0.30,
                        value_alpha=0.25,
                        seed=int(l_seed),
                    )

                    best = {
                        "lr": l_lr, "gamma": l_gamma,
                        "epsilon": l_epsilon, "v_epsilon": l_v_epsilon,
                        "gap": -999.0, "s_final": 0.0, "v_final": 0.0, "m_final": 0.0
                    }
                    gap_history      = []          # best gap 추이
                    gap_iter_history = []          # 각 iteration 기대값 (평가 성공 시)
                    mu_hist_norm = {k: [] for k in param_bounds}  # 정책 μ 정규화 값 추이
                    sim_display  = st.empty()

                    # 복수 시드 평균으로 일반화 성능 측정
                    _n_eval     = min(4, max(3, int(l_auto_runs) // 2))  # 최소 3 보장
                    _eval_seeds = [int(l_seed) + _j * 37 for _j in range(_n_eval)]  # improve 4-3: ×37 간격으로 OOS seed 포함

                    # Ghost 미리보기용 best traces 저장
                    _best_v_trace = None
                    _best_s_trace = None
                    _best_mkt_trace = None
                    _best_df_stock = None

                    # 페이즈 경계 (탐험→수렴)
                    _explore_end = max(6, n_iters // 4)

                    for _i in range(n_iters):
                        # ─ 인터럽트 체크 ─
                        if st.session_state.get('interrupt_requested', False):
                            st.session_state.interrupt_requested = False
                            sim_display.empty()
                            st.warning(f"⛔ Simulation 중단됨 ({_i}/{n_iters} 반복 완료)")
                            break

                        # ─ 페이즈 레이블 ─
                        _sigma_now = optimizer.sigma_mean
                        if _i < _explore_end:
                            phase_name = "🔴 PG Exploring"
                        elif _sigma_now > 0.12:
                            phase_name = "🟡 PG Actor-Critic"
                        else:
                            phase_name = "🟢 PG Converging"

                        # ─ Actor: 다음 파라미터 후보 제안 (π_θ 샘플링) ─
                        candidate = optimizer.suggest_next()

                        # ─ 복수 시드로 평가 → 평균 gap (STATIC vs Market) ─
                        _gaps, _s_list, _v_list, _m_list = [], [], [], []
                        _tmp_v_trace, _tmp_s_trace, _tmp_mkt_trace, _tmp_df_cand = None, None, None, None
                        for _eseed in _eval_seeds:
                            try:
                                _df_tmp, _vt, _s_tr, _mkt_tr, _, _, _, _, _ = get_rl_data(
                                    ticker,
                                    candidate["lr"], candidate["gamma"], candidate["epsilon"],
                                    int(l_epi), l_train_epi, _eseed, v_epsilon=candidate["v_epsilon"],
                                    fee_rate=fee_rate, interval=l_interval,
                                    use_vol=_use_vol_now, roll_period=_roll_period_now,
                                    algorithm=l_algorithm
                                )
                            except Exception:
                                _vt, _s_tr, _mkt_tr = None, None, None
                            if _vt is not None and _s_tr is not None and _mkt_tr is not None:
                                # improve 4-2: 복합 Gap — 시장 대비(60%) + Vanilla 대비(40%)
                                # V_floor = Market×0.3 → 역유인 제거 (Vanilla=0% 최대화 방지)
                                _gap_vs_market  = float(_s_tr[-1]) - float(_mkt_tr[-1])
                                _v_floor        = float(_mkt_tr[-1]) * 0.3
                                _v_adj          = max(float(_vt[-1]), _v_floor)
                                _gap_vs_vanilla = float(_s_tr[-1]) - _v_adj
                                _gaps.append(0.6 * _gap_vs_market + 0.4 * _gap_vs_vanilla)
                                _s_list.append(float(_s_tr[-1]))
                                _v_list.append(float(_vt[-1]))
                                _m_list.append(float(_mkt_tr[-1]))
                                if _tmp_v_trace is None:
                                    _tmp_v_trace = _vt
                                    _tmp_s_trace = _s_tr
                                    _tmp_mkt_trace = _mkt_tr
                                    _tmp_df_cand = _df_tmp

                        _iter_gap_val = None
                        if _gaps:
                            _gap = float(np.mean(_gaps))
                            _iter_gap_val = _gap
                            candidate["gap"]     = _gap
                            candidate["s_final"] = float(np.mean(_s_list))
                            candidate["v_final"] = float(np.mean(_v_list))
                            candidate["m_final"] = float(np.mean(_m_list))

                            # ─ Critic + Actor 업데이트 (Policy Gradient) ─
                            optimizer.update(candidate, _gap)

                            if _gap > best["gap"]:
                                best = candidate.copy()
                                _best_v_trace = _tmp_v_trace
                                _best_s_trace = _tmp_s_trace
                                _best_mkt_trace = _tmp_mkt_trace
                                _best_df_stock = _tmp_df_cand

                        gap_history.append(best["gap"])
                        gap_iter_history.append(_iter_gap_val)

                        # μ 정규화 추이 기록 (원본 값 → [0,1] 정규화)
                        for _k, (_lo, _hi) in param_bounds.items():
                            _mu_val = optimizer.mu_history[-1].get(_k, best.get(_k, 0.0)) \
                                      if optimizer.mu_history else best.get(_k, 0.0)
                            mu_hist_norm[_k].append(
                                (_mu_val - _lo) / (_hi - _lo) if _hi != _lo else 0.5
                            )

                        # ─ 실시간 디스플레이 ─
                        with sim_display.container(border=True):
                            _prog = (_i + 1) / n_iters
                            if best["gap"] >= 5.0:
                                _goal_txt = " 🏆 +5%p↑"
                            elif best["gap"] >= 1.0:
                                _goal_txt = " ✅"
                            else:
                                _goal_txt = ""

                            _disp_lr  = candidate.get("lr",        best["lr"])
                            _disp_g   = candidate.get("gamma",     best["gamma"])
                            _disp_e   = candidate.get("epsilon",   best["epsilon"])
                            _disp_ve  = candidate.get("v_epsilon", best["v_epsilon"])
                            _steps = len(mu_hist_norm["lr"])
                            _prev_idx = -2 if _steps > 1 else -1
                            _prev_lr  = mu_hist_norm["lr"][_prev_idx]        * (param_bounds["lr"][1]        - param_bounds["lr"][0])        + param_bounds["lr"][0]
                            _prev_g   = mu_hist_norm["gamma"][_prev_idx]     * (param_bounds["gamma"][1]     - param_bounds["gamma"][0])     + param_bounds["gamma"][0]
                            _prev_e   = mu_hist_norm["epsilon"][_prev_idx]   * (param_bounds["epsilon"][1]   - param_bounds["epsilon"][0])   + param_bounds["epsilon"][0]
                            _prev_ve  = mu_hist_norm["v_epsilon"][_prev_idx] * (param_bounds["v_epsilon"][1] - param_bounds["v_epsilon"][0]) + param_bounds["v_epsilon"][0]

                            # ── 3컬럼: 좌(진행+파라미터 카드) | 중(파라미터 수렴) | 우(기대값 수렴) ──
                            _left_col, _mid_col, _right_col = st.columns([1, 1.1, 1.1])

                            with _left_col:
                                st.progress(_prog,
                                    text=f"{phase_name}  {_i+1}/{n_iters}  |  "
                                         f"Gap {best['gap']:+.1f}%{_goal_txt}  σ={_sigma_now:.3f}")
                                st.markdown(
                                    f"<div style='font-size:11px;color:rgba(180,180,180,0.7);"
                                    f"margin:2px 0 6px 0;'>"
                                    f"STATIC {best['s_final']:+.2f}%  &nbsp;·&nbsp;  "
                                    f"Market {best['m_final']:+.2f}%</div>",
                                    unsafe_allow_html=True
                                )
                                _r1c1, _r1c2 = st.columns(2)
                                _r2c1, _r2c2 = st.columns(2)
                                _r1c1.metric("Learning Rate (α)", f"{_disp_lr:.4f}",
                                             f"{_disp_lr - _prev_lr:+.4f}")
                                _r1c2.metric("Discount Factor (γ)", f"{_disp_g:.4f}",
                                             f"{_disp_g - _prev_g:+.4f}")
                                _r2c1.metric("STATIC ε",  f"{_disp_e:.4f}",
                                             f"{_disp_e - _prev_e:+.4f}")
                                _r2c2.metric("Vanilla ε", f"{_disp_ve:.4f}",
                                             f"{_disp_ve - _prev_ve:+.4f}")

                            with _mid_col:
                                if _steps > 1:
                                    _steps_x = list(range(1, _steps + 1))
                                    _param_colors = {
                                        "lr":        "#4a90d9",
                                        "gamma":     "#e05050",
                                        "epsilon":   "#50c878",
                                        "v_epsilon": "#ff9800",
                                    }
                                    _param_labels = {
                                        "lr":        "α (LR)",
                                        "gamma":     "γ (Discount)",
                                        "epsilon":   "ε STATIC",
                                        "v_epsilon": "ε Vanilla",
                                    }
                                    _fig_sim = go.Figure()
                                    for _pk, _pc in _param_colors.items():
                                        _fig_sim.add_trace(go.Scatter(
                                            x=_steps_x,
                                            y=mu_hist_norm[_pk],
                                            mode="lines",
                                            name=f"<b>{_param_labels[_pk]}</b>",
                                            line=dict(color=_pc, width=2),
                                        ))
                                    _fig_sim.update_layout(
                                        title=dict(
                                            text="<b>Parameter Convergence (Policy μ, Normalized [0-1])</b>",
                                            font=dict(size=12),
                                            x=0.5, xanchor="center",
                                        ),
                                        height=420,
                                        margin=dict(l=10, r=15, t=45, b=40),
                                        xaxis=dict(
                                            title="<b>Step</b>", showgrid=True,
                                            range=[0.5, _steps + 0.5],
                                        ),
                                        yaxis=dict(
                                            title="<b>Normalized Value [0-1]</b>",
                                            showgrid=True,
                                            range=[-0.05, 1.05],
                                        ),
                                        legend=dict(
                                            orientation="v",
                                            x=0.01, y=0.90,
                                            xanchor="left", yanchor="top",
                                            font=dict(size=12, color="white"),
                                            bgcolor="rgba(15,15,28,0.80)",
                                            bordercolor="rgba(160,160,170,0.35)",
                                            borderwidth=1,
                                        ),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                    )
                                    st.plotly_chart(_fig_sim, use_container_width=True,
                                                    key=f"sim_chart_{m_name}_{stock_name}_{_i}")

                            # ── 우: 기대값 → 목표 수렴 차트 ──
                            with _right_col:
                                if _steps > 1:
                                    _gap_steps = list(range(1, len(gap_history) + 1))
                                    _fig_gap = go.Figure()

                                    # 현재 iteration 기대값 (산점)
                                    _vx = [i+1 for i, g in enumerate(gap_iter_history) if g is not None]
                                    _vy = [g   for g in gap_iter_history if g is not None]
                                    if _vx:
                                        _fig_gap.add_trace(go.Scatter(
                                            x=_vx, y=_vy,
                                            mode="lines",
                                            name="<b>Current Expected</b>",
                                            line=dict(color="rgba(150,200,255,0.6)", width=1.2),
                                        ))

                                    # 누적 최적 gap 라인
                                    _fig_gap.add_trace(go.Scatter(
                                        x=_gap_steps, y=gap_history,
                                        mode="lines",
                                        name="<b>Best Expected</b>",
                                        line=dict(color="#ffffff", width=2.2),
                                        fill="tozeroy",
                                        fillcolor="rgba(80,200,120,0.07)",
                                    ))

                                    # 목표선
                                    _fig_gap.add_hline(
                                        y=1.0, line_dash="dash", line_color="#50c878",
                                        annotation_text="Target +1%p (vs Market)",
                                        annotation_position="top right",
                                        annotation_font_size=10,
                                    )
                                    _fig_gap.add_hline(
                                        y=5.0, line_dash="dot", line_color="#ffd700",
                                        annotation_text="Best +5%p (vs Market)",
                                        annotation_position="top right",
                                        annotation_font_size=10,
                                    )

                                    _fig_gap.update_layout(
                                        title=dict(
                                            text="<b>STATIC Alpha vs Market → Target Convergence</b>",
                                            font=dict(size=12),
                                            x=0.5, xanchor="center",
                                        ),
                                        height=420,
                                        margin=dict(l=10, r=10, t=45, b=40),
                                        xaxis=dict(
                                            title="<b>Step</b>", showgrid=True,
                                            range=[0.5, _steps + 0.5],
                                        ),
                                        yaxis=dict(title="<b>Alpha vs Market (%p)</b>", showgrid=True),
                                        legend=dict(
                                            orientation="v",
                                            x=0.01, y=0.90,
                                            xanchor="left", yanchor="top",
                                            font=dict(size=11, color="white"),
                                            bgcolor="rgba(15,15,28,0.80)",
                                            bordercolor="rgba(160,160,170,0.35)",
                                            borderwidth=1,
                                        ),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                    )
                                    st.plotly_chart(_fig_gap, use_container_width=True,
                                                    key=f"gap_chart_{m_name}_{stock_name}_{_i}")

                            # ── 실시간 누적 수익률 비교 차트 ──
                            if (_best_v_trace is not None and _best_s_trace is not None
                                    and _best_mkt_trace is not None and _best_df_stock is not None):
                                _fig_rt = _make_cumulative_fig(
                                    stock_name, _best_df_stock,
                                    _best_v_trace, _best_s_trace, _best_mkt_trace,
                                    algo_name=l_algorithm if l_algorithm != "STATIC" else "STATIC RL",
                                )
                                st.plotly_chart(_fig_rt, use_container_width=True,
                                                key=f"rt_curve_{m_name}_{stock_name}_{_i}")

                    # ─ 완료: Ghost 데이터 저장 ─
                    best["found"] = best["gap"] >= 1.0
                    st.session_state.sim_result[hist_key] = best

                    # Ghost Line 저장 (재실행 후에도 차트에 점선으로 표시됨)
                    if _best_v_trace is not None and _best_s_trace is not None:
                        st.session_state.ghost_data[hist_key] = {
                            'v_trace': _best_v_trace,
                            's_trace': _best_s_trace,
                            'params': {
                                'lr':        best['lr'],
                                'gamma':     best['gamma'],
                                'epsilon':   best['epsilon'],
                                'v_epsilon': best['v_epsilon'],
                            },
                            'gap': best['gap'],
                        }

                    _best_params = {
                        "lr":        best["lr"],
                        "gamma":     best["gamma"],
                        "epsilon":   best["epsilon"],
                        "v_epsilon": best["v_epsilon"],
                        "gap":       best["gap"],
                        "s_final":   best.get("s_final", 0.0),
                        "v_final":   best.get("v_final", 0.0),
                        "algorithm": l_algorithm,
                    }
                    if st.session_state.get('sim_auto_save', False):
                        # Simul. All 모드: 대화상자 없이 자동 저장 및 Run Eval 트리거
                        _save_sim_params_to_config(m_config, stock_idx, _best_params)
                        st.session_state[f"sim_pending_{hist_key}"] = _best_params
                        st.session_state.stocks_reverted.add(hist_key)
                        st.session_state[f"auto_run_{hist_key}"] = True
                        # 큐가 비었으면 자동 저장 모드 해제
                        if not st.session_state.sim_all_queue:
                            st.session_state.sim_auto_save = False
                    else:
                        # 개별 시뮬레이션: 저장 여부 확인 대화상자 표시
                        st.session_state[f"sim_confirm_{hist_key}"] = _best_params
                    sim_display.empty()
                    st.rerun()

                # ── 유효 파라미터 결정: fallback 활성 여부 ──
                _use_fb = (
                    st.session_state.stock_use_fallback == "ALL"
                    and hist_key not in st.session_state.stocks_reverted
                )
                if _use_fb:
                    fp    = st.session_state.fallback_params
                    _fchk = fp.get("checked", {k: True for k in _ALL_CHK_KEYS})
                    eff_lr        = fp["lr"]                                      if _fchk.get("lr")        else l_lr
                    eff_gamma     = fp["gamma"]                                   if _fchk.get("gamma")     else l_gamma
                    eff_eps       = fp["epsilon"]                                 if _fchk.get("eps")       else l_epsilon
                    eff_epi       = fp["episodes"]                                if _fchk.get("episodes")  else l_epi
                    eff_train_epi = fp.get("train_episodes", 100)                 if _fchk.get("train_epi") else l_train_epi
                    eff_seed      = fp["seed"]                                    if _fchk.get("seed")      else l_seed
                    eff_v_eps     = fp.get("v_epsilon", fp["epsilon"])            if _fchk.get("v_eps")     else l_v_epsilon
                    eff_active_agents = fp.get("active_agents", ["Vanilla RL", "STATIC RL"]) if _fchk.get("active") else l_active_agents
                    eff_algorithm = fp.get("algorithm", "STATIC") if _fchk.get("algo") else l_algorithm
                    eff_sim_min  = fp.get("sim_min",  l_sim_min)  if _fchk.get("sim_min")  else l_sim_min
                    eff_sim_mult = fp.get("sim_mult", l_sim_mult) if _fchk.get("sim_mult") else l_sim_mult
                    _fb_parts = []
                    if _fchk.get("lr"):        _fb_parts.append(f"LR={eff_lr:.3f}")
                    if _fchk.get("gamma"):     _fb_parts.append(f"γ={eff_gamma:.2f}")
                    if _fchk.get("eps"):       _fb_parts.append(f"ε(S)={eff_eps:.2f}")
                    if _fchk.get("v_eps"):     _fb_parts.append(f"ε(V)={eff_v_eps:.2f}")
                    if _fchk.get("episodes"):  _fb_parts.append(f"Days={eff_epi}")
                    if _fchk.get("train_epi"): _fb_parts.append(f"Episodes={eff_train_epi}")
                    if _fchk.get("seed"):      _fb_parts.append(f"Seed={eff_seed}")
                    if _fchk.get("active"):    _fb_parts.append(f"Agents={', '.join(eff_active_agents) if eff_active_agents else '없음'}")
                    if _fchk.get("algo"):      _fb_parts.append(f"Algo={eff_algorithm}")
                    if _fchk.get("sim_min"):   _fb_parts.append(f"MinSteps={eff_sim_min}")
                    if _fchk.get("sim_mult"):  _fb_parts.append(f"SimMult={eff_sim_mult}")
                    st.info(
                        "Fallback 파라미터 적용 중 (" + "  ".join(_fb_parts) + ")",
                        icon="ℹ️"
                    )
                else:
                    eff_lr, eff_gamma, eff_eps, eff_epi, eff_train_epi, eff_seed = (
                        l_lr, l_gamma, l_epsilon, l_epi, l_train_epi, l_seed
                    )
                    eff_v_eps         = l_v_epsilon
                    eff_active_agents = l_active_agents
                    eff_algorithm     = l_algorithm
                    eff_sim_min       = l_sim_min
                    eff_sim_mult      = l_sim_mult

                # ── 시뮬레이션 실행 (유효 파라미터 기준) ──
                # [P3/P4] _use_vol_now, _roll_period_now: p_settings 정의 직후 공통 계산 완료
                with st.spinner(f"Processing {stock_name}..."):
                    df_stock, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable = get_rl_data(
                        ticker, eff_lr, eff_gamma, eff_eps, eff_epi, eff_train_epi, eff_seed,
                        v_epsilon=eff_v_eps, fee_rate=fee_rate, interval=l_interval,
                        use_vol=_use_vol_now, roll_period=_roll_period_now,
                        algorithm=eff_algorithm
                    )
                # [P2] 학습된 정책 캐시 저장 (State Analysis Dashboard용)
                # 신경망 알고리즘(A2C/PPO 등)은 s_theta = TinyMLP actor → theta 표시 불가
                _is_neural = eff_algorithm not in ("STATIC",)
                _theta_for_cache = None if _is_neural else s_theta
                if _theta_for_cache is not None or v_qtable is not None:
                    st.session_state.policy_cache[hist_key] = {
                        'theta':     _theta_for_cache,
                        'q_table':   v_qtable,
                        'n_states':  _theta_for_cache.shape[0] if _theta_for_cache is not None else 4,
                        'algorithm': eff_algorithm,
                    }

                if df_stock is None:
                    st.warning(f"데이터를 불러올 수 없습니다: {stock_name}")
                    continue

                # ── Active Agents 적용: 비활성 에이전트 → 0% 수평선 ──
                if "Vanilla RL" not in eff_active_agents:
                    v_trace = np.zeros(len(df_stock))
                    v_log   = []
                if "STATIC RL" not in eff_active_agents:
                    s_trace = np.zeros(len(df_stock))
                    s_log   = []
                    s_mdd   = 0.0

                s_final      = float(s_trace[-1])
                v_final      = float(v_trace[-1])
                market_final = float(real_ret_trace.iloc[-1])
                # 마지막 스텝 일별 수익 (delta용)
                v_last_day = float(v_trace[-1] - v_trace[-2]) if len(v_trace) > 1 else 0.0
                s_last_day = float(s_trace[-1] - s_trace[-2]) if len(s_trace) > 1 else 0.0
                m_last_day = float(real_ret_trace.iloc[-1] - real_ret_trace.iloc[-2]) if len(real_ret_trace) > 1 else 0.0

                # ── Ghost Line 로드 (이전 Simulation 결과) ──
                ghost = st.session_state.ghost_data.get(hist_key)
                opt_v = ghost['v_trace'] if ghost else None
                opt_s = ghost['s_trace'] if ghost else None

                # ── 멤버별 에쿼티 곡선 버퍼에 추가 (팀 펀드용) ──
                if m_name not in _member_trace_buf:
                    _member_trace_buf[m_name] = {
                        'traces': [], 'dates': df_stock.index, 'stock_names': []
                    }
                _member_trace_buf[m_name]['traces'].append(s_trace)
                _member_trace_buf[m_name]['stock_names'].append(stock_name)

                # ── 메인 2컬럼 레이아웃 ──
                col_left, col_right = st.columns([1, 1])

                # ══════════════════════════════════════════
                # 왼쪽: 누적 수익 차트 + 지표 카드 + 의사결정 분석
                # ══════════════════════════════════════════
                with col_left:
                    st.markdown(f"#### {stock_name} Performance")

                    # 누적 수익 차트 (Ghost Line 포함)
                    fig_cum = _make_cumulative_fig(
                        stock_name, df_stock, v_trace, s_trace, real_ret_trace,
                        opt_v_trace=opt_v, opt_s_trace=opt_s,
                        algo_name=eff_algorithm if eff_algorithm != "STATIC" else "STATIC RL",
                    )
                    st.plotly_chart(fig_cum, use_container_width=True,
                                    key=f"chart_cum_{m_name}_{stock_name}")

                    # Ghost 존재 시 최적 파라미터 캡션 표시
                    if ghost:
                        g = ghost['params']
                        st.caption(
                            f"✦ Optimal Ghost: LR={g['lr']:.4f}  γ={g['gamma']:.4f}  "
                            f"ε(S)={g['epsilon']:.4f}  ε(V)={g['v_epsilon']:.4f}  "
                            f"Gap={ghost['gap']:+.2f}%"
                        )

                    # 3 지표 카드 – 색상 커스터마이징 (HTML)
                    st.markdown(
                        "<p style='margin:6px 0 2px 0;font-size:12px;font-weight:700;"
                        "color:rgba(180,180,180,0.8);letter-spacing:0.05em;'>Final Cumulative Return</p>",
                        unsafe_allow_html=True
                    )
                    mc1, mc2, mc3 = st.columns(3)
                    def _metric_html(label, value_pct, delta_pct, color):
                        delta_sign = "▲" if delta_pct >= 0 else "▼"
                        delta_color = "#2ecc71" if delta_pct >= 0 else "#e05050"
                        return (
                            f"<div style='padding:8px 4px 4px 4px;'>"
                            f"<div style='font-size:12px;font-weight:700;color:{color};'>{label}</div>"
                            f"<div style='font-size:28px;font-weight:900;color:{color};line-height:1.2;'>"
                            f"{value_pct:.2f}%</div>"
                            f"<div style='font-size:13px;color:{delta_color};margin-top:2px;'>"
                            f"{delta_sign} {abs(delta_pct):.2f}%</div>"
                            f"</div>"
                        )
                    with mc1:
                        st.markdown(_metric_html("Vanilla RL", v_final, v_last_day, "#e05050"), unsafe_allow_html=True)
                    with mc2:
                        st.markdown(_metric_html("STATIC RL", s_final, s_last_day, "#4a90d9"), unsafe_allow_html=True)
                    with mc3:
                        st.markdown(_metric_html("Market (Buy&Hold)", market_final, m_last_day, "#2ecc71"), unsafe_allow_html=True)

                    # Agent Decision Analysis – Action Frequency(좌) + 테이블(우)
                    if v_log and s_log:
                        st.markdown("---")
                        st.markdown("#### Agent Decision Analysis")

                        df_v = pd.DataFrame(v_log).rename(columns={"Action": "Vanilla Action", "Daily_Return(%)": "Vanilla Return(%)"})
                        df_s = pd.DataFrame(s_log).rename(columns={"Action": "STATIC Action",  "Daily_Return(%)": "STATIC Return(%)"})
                        df_log = df_v.merge(df_s, on="Day").set_index("Day")

                        bar_col, tbl_col = st.columns([1, 1.4])
                        with bar_col:
                            action_counts = df_log["STATIC Action"].value_counts().reset_index()
                            action_counts.columns = ["Action", "Count"]
                            for _act in ["BUY", "CASH"]:
                                if _act not in action_counts["Action"].values:
                                    action_counts = pd.concat([action_counts,
                                        pd.DataFrame({"Action": [_act], "Count": [0]})], ignore_index=True)
                            action_counts = action_counts.sort_values("Action").reset_index(drop=True)
                            _bar_colors = {"BUY": "#4a90d9", "CASH": "#e05050"}
                            fig_bar = go.Figure()
                            for _, row in action_counts.iterrows():
                                fig_bar.add_trace(go.Bar(
                                    x=[row["Action"]], y=[row["Count"]],
                                    orientation='v',
                                    name=row["Action"],
                                    marker_color=_bar_colors.get(row["Action"], "#888"),
                                    width=0.35,
                                    text=[f"<b>{row['Count']}</b>"],
                                    textposition="outside",
                                    textfont=dict(size=14, color=_bar_colors.get(row["Action"], "#888"))
                                ))
                            fig_bar.update_layout(
                                title=dict(text="<b>STATIC: Action Frequency</b>", font=dict(size=13)),
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                height=250, showlegend=False,
                                margin=dict(t=45, b=25, l=30, r=10),
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=True, range=[0, action_counts["Count"].max() * 1.2])
                            )
                            st.plotly_chart(fig_bar, use_container_width=True,
                                            key=f"bar_{m_name}_{stock_name}")
                        with tbl_col:
                            _td = "padding:3px 3px;text-align:right;font-weight:bold;font-size:13px;border-bottom:1px solid rgba(128,128,128,0.1);background:var(--background-color,#0e1117);position:relative;z-index:1;"
                            _th = "padding:5px 3px;text-align:right;font-size:12px;border-bottom:2px solid rgba(128,128,128,0.4);position:sticky;top:0;z-index:10;background:var(--secondary-background-color,#1e1e2e);"
                            _rows = ""
                            for _day, _r in df_log.iterrows():
                                _va = _r["Vanilla Action"]
                                _vr = float(_r["Vanilla Return(%)"])
                                _sa = _r["STATIC Action"]
                                _sr = float(_r["STATIC Return(%)"])
                                _vc = "#4a90d9" if _va == "BUY" else "#e05050"
                                _sc = "#4a90d9" if _sa == "BUY" else "#e05050"
                                _vrc = "color:#e05050;" if _vr < 0 else ""
                                _src = "color:#e05050;" if _sr < 0 else ""
                                _rows += (
                                    f"<tr>"
                                    f"<td style='{_td}'>{_day}</td>"
                                    f"<td style='{_td}color:{_vc};'>{_va}</td>"
                                    f"<td style='{_td}{_vrc}'>{_vr:.2f}</td>"
                                    f"<td style='{_td}color:{_sc};'>{_sa}</td>"
                                    f"<td style='{_td}{_src}'>{_sr:.2f}</td>"
                                    f"</tr>"
                                )
                            st.markdown(
                                f"<div style='max-height:253px;overflow-y:auto;"
                                f"background:var(--background-color);"
                                f"border:1px solid rgba(128,128,128,0.3);border-radius:4px;'>"
                                f"<table style='width:100%;border-collapse:collapse;'>"
                                f"<thead><tr>"
                                f"<th style='{_th}'>Day</th>"
                                f"<th style='{_th}'>Vanilla<br>Action</th>"
                                f"<th style='{_th}'>Vanilla<br>Return(%)</th>"
                                f"<th style='{_th}'>STATIC<br>Action</th>"
                                f"<th style='{_th}'>STATIC<br>Return(%)</th>"
                                f"</tr></thead>"
                                f"<tbody>{_rows}</tbody>"
                                f"</table></div>",
                                unsafe_allow_html=True
                            )

                    # ── [P2] State Policy Analysis (Explainable RL) ──────────
                    _pcache = st.session_state.policy_cache.get(hist_key)
                    if _pcache is not None:
                        with st.expander("State Policy Analysis (Explainable RL)", expanded=False):
                            _theta_c   = _pcache.get('theta')
                            _qtable_c  = _pcache.get('q_table')
                            _ns        = _pcache.get('n_states', 4)

                            _s4_labels = [
                                "S0: 하락+EMA아래", "S1: 상승+EMA아래",
                                "S2: 하락+EMA위",   "S3: 상승+EMA위"
                            ]
                            _s8_labels = _s4_labels + [
                                "S4: 하락+EMA아래+고변동", "S5: 상승+EMA아래+고변동",
                                "S6: 하락+EMA위+고변동",   "S7: 상승+EMA위+고변동"
                            ]
                            _state_labels = _s8_labels if _ns == 8 else _s4_labels

                            pol_c1, pol_c2 = st.columns(2)

                            # 신경망 알고리즘: tabular State Analysis 미지원
                            _algo_label = _pcache.get('algorithm', 'STATIC')
                            if _algo_label not in ('STATIC',) and _theta_c is None:
                                st.info(
                                    f"**{_algo_label}** 알고리즘은 연속 특징 벡터(5차원) 기반 TinyMLP를 사용합니다. "
                                    "이산 상태 P(BUY|state) 시각화는 STATIC 알고리즘에서만 지원됩니다.",
                                    icon="ℹ️"
                                )

                            # STATIC RL: 각 상태별 P(BUY|s)
                            if _theta_c is not None:
                                with pol_c1:
                                    st.markdown("**STATIC RL — P(BUY|state)**")
                                    _buy_probs = []
                                    for _s in range(_ns):
                                        _logits = _theta_c[_s]
                                        _emax   = np.exp(np.clip(_logits - _logits.max(), -30, 30))
                                        _buy_probs.append(float(_emax[1] / (_emax.sum() + 1e-10)))
                                    _fig_pol = go.Figure(go.Bar(
                                        x=_buy_probs,
                                        y=_state_labels[:_ns],
                                        orientation='h',
                                        marker_color=[
                                            f"rgba(74,144,217,{max(0.3, p)})" for p in _buy_probs
                                        ],
                                        text=[f"{p:.1%}" for p in _buy_probs],
                                        textposition='outside'
                                    ))
                                    _fig_pol.update_layout(
                                        height=200 + _ns * 22,
                                        xaxis=dict(range=[0, 1.15], tickformat='.0%',
                                                   showgrid=True, title="P(BUY)"),
                                        yaxis=dict(autorange='reversed'),
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        margin=dict(t=10, b=20, l=5, r=30),
                                        showlegend=False
                                    )
                                    st.plotly_chart(_fig_pol, use_container_width=True,
                                                    key=f"pol_s_{m_name}_{stock_name}")

                            # Vanilla RL: Q[s,BUY] - Q[s,CASH] Advantage
                            if _qtable_c is not None:
                                with pol_c2:
                                    st.markdown("**Vanilla RL — Q Advantage (BUY-CASH)**")
                                    _adv = [float(_qtable_c[s, 1] - _qtable_c[s, 0])
                                            for s in range(2)]
                                    _fig_q = go.Figure(go.Bar(
                                        x=_adv,
                                        y=["S0: 하락", "S1: 상승"],
                                        orientation='h',
                                        marker_color=[
                                            "#4a90d9" if a >= 0 else "#e05050" for a in _adv
                                        ],
                                        text=[f"{a:+.4f}" for a in _adv],
                                        textposition='outside'
                                    ))
                                    _fig_q.update_layout(
                                        height=180,
                                        xaxis=dict(title="Q(BUY) - Q(CASH)", showgrid=True,
                                                   zeroline=True, zerolinecolor='rgba(180,180,180,0.5)'),
                                        yaxis=dict(autorange='reversed'),
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        margin=dict(t=10, b=20, l=5, r=50),
                                        showlegend=False
                                    )
                                    st.plotly_chart(_fig_q, use_container_width=True,
                                                    key=f"pol_v_{m_name}_{stock_name}")

                # ══════════════════════════════════════════
                # 오른쪽: Trial History Statistical Analysis
                # ══════════════════════════════════════════
                with col_right:
                    st.markdown("### Trial History: Statistical Analysis (Alpha Performance)")
                    trials = st.session_state.stock_trial_history.get(hist_key, [])

                    if not trials:
                        st.markdown("""
<div style='background:var(--secondary-background-color);padding:40px;border-radius:10px;
border:1px solid rgba(128,128,128,0.3);text-align:center;margin-top:20px;'>
<h4 style='color:rgba(128,128,128,0.6);margin-bottom:8px;'>No Trial History</h4>
<p style='color:rgba(128,128,128,0.4);font-size:14px;'>Click <b>▶ Run Evaluation</b> above<br>to accumulate trial history</p>
</div>""", unsafe_allow_html=True)
                    else:
                        df_h = pd.DataFrame(trials)
                        # Trial > 0 필터: 이전 세션 잔류 데이터(음수/0) 제거
                        df_h = df_h[df_h['Trial'] > 0].reset_index(drop=True)
                        v_mean = df_h['Vanilla Final (%)'].mean()
                        s_mean = df_h['STATIC Final (%)'].mean()
                        v_std  = df_h['Vanilla Final (%)'].std(ddof=0) if len(df_h) > 1 else 0.0
                        s_std  = df_h['STATIC Final (%)'].std(ddof=0) if len(df_h) > 1 else 0.0
                        v_min, v_max = df_h['Vanilla Final (%)'].min(), df_h['Vanilla Final (%)'].max()
                        s_min, s_max = df_h['STATIC Final (%)'].min(), df_h['STATIC Final (%)'].max()
                        avg_mkt = df_h['Market Final (%)'].mean()

                        # Alpha 배너
                        st.success(
                            f"**Expected Alpha vs. Market Avg.**: "
                            f"STATIC **{s_mean - avg_mkt:.2f}%p** | Vanilla **{v_mean - avg_mkt:.2f}%p**"
                        )

                        # Trial-by-Trial 추이 차트
                        st.plotly_chart(_make_trend_fig(df_h), use_container_width=True,
                                        key=f"trend_{m_name}_{stock_name}")

                        # 박스 플롯 + 통계 카드 2열
                        box_col, stat_col = st.columns([1.6, 1])
                        with box_col:
                            st.plotly_chart(_make_trial_box_fig(df_h), use_container_width=True,
                                            key=f"tribox_{m_name}_{stock_name}")
                        with stat_col:
                            # 스페이서: 통계 요약 섹션을 하단으로 이동
                            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background:var(--secondary-background-color);padding:10px 0 12px 0;border-radius:10px;border:1px solid rgba(128,128,128,0.3);width:100%;box-sizing:border-box;'><h4 style='margin-top:0;margin-bottom:8px;font-weight:900;font-size:15px;line-height:1.25;padding:0 6px;'>Statistics Summary<br>(Expected &amp; Risk)</h4><div style='display:flex;gap:0;padding:0 4px;'><div style='flex:1;border-right:1px solid rgba(128,128,128,0.3);padding-right:4px;'><ul style='font-size:13px;margin:0;padding-left:0;line-height:1.4;list-style:none;text-align:left;'><li style='margin-bottom:8px;'><b style='color:#e05050;'>Vanilla Mean</b><br>{v_mean:.2f}% (σ={v_std:.2f}%)</li><li style='margin-bottom:4px;'><b style='color:#e05050;'>Vanilla Range</b><br>{v_min:.2f}% ~ {v_max:.2f}%</li></ul></div><div style='flex:1;padding-left:4px;'><ul style='font-size:13px;margin:0;padding-left:0;line-height:1.4;list-style:none;text-align:left;'><li style='margin-bottom:8px;'><b style='color:#4a90d9;'>STATIC Mean</b><br>{s_mean:.2f}% (σ={s_std:.2f}%)</li><li style='margin-bottom:4px;'><b style='color:#4a90d9;'>STATIC Range</b><br>{s_min:.2f}% ~ {s_max:.2f}%</li></ul></div></div></div>""", unsafe_allow_html=True)

                            def _color_neg(val):
                                if isinstance(val, (int, float)) and val < 0:
                                    return 'color: #e05050; font-weight: bold;'
                                return 'font-weight: bold;'

                            _disp_df = df_h.rename(columns={
                                "Vanilla Final (%)": "Vanilla<br>Final (%)",
                                "STATIC Final (%)":  "STATIC<br>Final (%)",
                                "Market Final (%)":  "Market<br>Final (%)",
                            }).set_index("Trial")
                            _disp_df.index.name = None
                            _tbl_html = (
                                _disp_df.style
                                .map(_color_neg)
                                .format({
                                    "Vanilla<br>Final (%)": "{:.2f}",
                                    "STATIC<br>Final (%)":  "{:.2f}",
                                    "Market<br>Final (%)":  "{:.2f}",
                                    "Seed": "{:.0f}"
                                })
                                .set_properties(**{"text-align": "right"})
                                .set_table_styles([
                                    {"selector": "th", "props": [
                                        ("text-align", "right"),
                                        ("vertical-align", "middle"),
                                        ("padding", "4px 6px"),
                                        ("font-size", "12px"),
                                        ("line-height", "1.3"),
                                    ]},
                                    {"selector": "td", "props": [("padding", "3px 6px")]},
                                    {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
                                ])
                                .to_html(escape=False)
                            )
                            st.markdown(
                                f'<div style="max-height:300px;overflow-y:auto;font-size:13px;width:100%;box-sizing:border-box;">{_tbl_html}</div>',
                                unsafe_allow_html=True
                            )

                total_episodes_run += l_epi
                rendered_count += 1
                if total_charts > 0:
                    pct = min(rendered_count / total_charts, 1.0)
                    st.session_state.master_pbar_pct = pct
                    _render_master_pbar_html(pct, master_pbar)

                mem_s_rets.append(s_final)
                mem_v_rets.append(v_final)
                mem_mdds.append(s_mdd)

        # ── 멤버 평균 에쿼티 곡선 계산 및 session_state 저장 ──
        if _member_trace_buf.get(m_name, {}).get('traces'):
            _buf = _member_trace_buf[m_name]
            _traces = _buf['traces']
            _min_len = min(len(t) for t in _traces)
            _avg_trace = np.mean([t[:_min_len] for t in _traces], axis=0)
            st.session_state.member_traces[m_name] = {
                's_trace':    _avg_trace,
                'dates':      _buf['dates'][:_min_len],
                'stocks':     _buf['stock_names'],
            }

        if mem_s_rets:
            avg_s, avg_v = np.mean(mem_s_rets), np.mean(mem_v_rets)
            # selected_stock_names는 현재 멤버 루프의 마지막 종목 목록
            _m_stocks = st.session_state.get(f"ms_{m_name}", [
                all_stock_names[i] for i in getattr(m_config, 'TARGET_INDICES', [])
                if i in all_stock_names
            ])
            final_contributions.append({
                "Member": m_name,
                "Stocks": ", ".join(_m_stocks),
                "Final_Capital": 1.0 * (1 + avg_s / 100),
                "Profit_Dollar": (1.0 * (1 + avg_s / 100)) - 1.0,
                "Vanilla_Profit": (1.0 * (1 + avg_v / 100)) - 1.0,
                "Avg_Return": avg_s,
                "Vanilla_Return": avg_v,
                "Avg_MDD": np.mean(mem_mdds),
                "CTPT_Code": ctpt_code
            })
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

if final_contributions:
    st.session_state.master_pbar_pct = 1.0
    _render_master_pbar_html(1.0, master_progress_placeholder)

    # 첫 번째 데이터 도착 시 st.rerun() → 상단 container에 즉시 차트 표시
    _first_data = not bool(st.session_state.prev_final_contributions)

    # 세션 상태 업데이트 (다음 rerun에서 상단 컨테이너가 최신 데이터로 렌더)
    st.session_state.prev_final_contributions = final_contributions
    st.session_state.prev_episodes_run = total_episodes_run

    if _first_data:
        st.rerun()  # 차트가 아직 없을 때 한 번만 강제 갱신

# 스크립트 실행 완료 후 load bar 최종 렌더
update_load_bar(
    total_episodes_run if final_contributions else st.session_state.prev_episodes_run,
    gauge_placeholder
)
