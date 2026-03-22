#!/usr/bin/env python3
"""
update_html_params.py
────────────────────────────────────────────────────────────────
6개 멤버의 config.py 파라미터를 읽어
Presentation_Bellman_to_Static_H.html 의 MEMBER_PARAMS JS 객체를
자동으로 최신값으로 갱신합니다.

사용법:
    python update_html_params.py
────────────────────────────────────────────────────────────────
"""

import importlib.util
import io
import re
import sys
from pathlib import Path

# Windows cp949 터미널에서 한글/특수문자 출력 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent
HTML_PATH = ROOT / "Presentation_Bellman_to_Static_H.html"


# ── 1. config.py 동적 로드 ──────────────────────────────────────
def load_config(member_n: int) -> dict:
    """members/member_N/config.py 를 로드하여 첫 번째 종목 파라미터 반환"""
    path = ROOT / "members" / f"member_{member_n}" / "config.py"
    spec = importlib.util.spec_from_file_location(f"cfg_m{member_n}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    key = mod.TARGET_INDICES[0]          # e.g. 0, 3, 10, 11 …
    p   = mod.RL_PARAMS[key]
    return {
        "state": 8 if p.get("use_vol", False) else 4,
        "roll":  p.get("roll_period"),   # None or int
        "lr":    p["lr"],
        "gamma": p["gamma"],
        "eps_s": p["epsilon"],
        "eps_v": p["v_epsilon"],
    }


# ── 2. JS MEMBER_PARAMS 블록 생성 ──────────────────────────────
def build_js_block(params: dict) -> str:
    lines = ["      const MEMBER_PARAMS = {"]
    keys  = list(params.keys())
    for i, key in enumerate(keys):
        p        = params[key]
        roll_val = p["roll"] if p["roll"] is not None else "null"
        comma    = "," if i < len(keys) - 1 else ""
        lines.append(
            f'        {key}: {{ state: {p["state"]}, roll: {roll_val}, '
            f'lr: {p["lr"]}, gamma: {p["gamma"]}, '
            f'eps_s: {p["eps_s"]}, eps_v: {p["eps_v"]} }}{comma}'
        )
    lines.append("      };")
    return "\n".join(lines)


# ── 3. HTML 내 MEMBER_PARAMS 블록 교체 ─────────────────────────
PATTERN = re.compile(
    r"(      const MEMBER_PARAMS = \{.*?\n      \};)",
    re.DOTALL,
)

def update_html(new_block: str):
    text = HTML_PATH.read_text(encoding="utf-8")
    if not PATTERN.search(text):
        print("ERROR: HTML에서 MEMBER_PARAMS 블록을 찾을 수 없습니다.")
        sys.exit(1)
    new_text = PATTERN.sub(new_block, text, count=1)
    HTML_PATH.write_text(new_text, encoding="utf-8")


# ── 4. 메인 ────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  HTML 파라미터 자동 갱신 - update_html_params.py")
    print("=" * 52)

    params = {}
    for n in range(1, 7):
        try:
            p = load_config(n)
            params[f"m{n}"] = p
            roll_str = f"Roll={p['roll']}" if p["roll"] else "Roll=None"
            print(
                f"  M{n} 로드 완료 │ {p['state']}-State │ {roll_str} │ "
                f"lr={p['lr']:.3f} │ γ={p['gamma']:.3f} │ "
                f"εₛ={p['eps_s']:.3f} │ εᵥ={p['eps_v']:.3f}"
            )
        except Exception as e:
            print(f"  M{n} 로드 실패: {e}")
            sys.exit(1)

    print()
    new_block = build_js_block(params)
    update_html(new_block)
    print(f"  HTML 갱신 완료: {HTML_PATH.name}")
    print("=" * 52)


if __name__ == "__main__":
    main()
