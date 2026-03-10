import numpy as np

class BaseSTATICAgent:
    """모든 팀원이 공통으로 사용할 수 있는 기본 STATIC RL 에이전트 클래스"""
    
    def __init__(self, rl_params, custom_params=None):
        # 기본 강화학습 파라미터 적용
        self.lr = rl_params.get("learning_rate", 0.01)
        self.gamma = rl_params.get("discount_factor", 0.98)
        self.epsilon = rl_params.get("exploration_rate", 0.10)
        
        # 팀원별 커스텀 파라미터 수용
        self.custom_params = custom_params if custom_params is not None else {}
        
    def apply_static_mask(self, logits, valid_mask):
        """
        STATIC 프레임워크의 핵심인 CSR 매트릭스 변환 및 마스킹 로직
        유효하지 않은 행동의 확률을 -무한대로 설정합니다.
        """
        return np.where(valid_mask, logits, -1e9)

    def select_action(self, state_logits, valid_mask):
        """마스킹된 로짓을 바탕으로 최적의 행동을 선택합니다."""
        masked_logits = self.apply_static_mask(state_logits, valid_mask)
        # Softmax 기반 확률적 선택 또는 Argmax 기반 탐욕적 선택 로직 구현
        best_action = np.argmax(masked_logits)
        return best_action