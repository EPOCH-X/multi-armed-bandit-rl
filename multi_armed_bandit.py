from unittest import result

import numpy as np

class BanditEnvironment:
    def __init__(self):
        # 5개 머신의 실제 당첨 확률 (비밀 설정)
        # 예: 2번 인덱스(3번째 머신)가 80% 확률로 가장 높음
        self.arms_probs = [0.1, 0.3, 0.8, 0.2, 0.5]
        self.n_arms = len(self.arms_probs)

    def react(self, arm_index):
        """
        선수가 레버를 당겼을 때(arm_index) 결과를 반환합니다.
        p의 확률로 1, 1-p의 확률로 0이 나옵니다.
        """
        result = np.random.choice([1, 0],
        p=[self.arms_probs[arm_index], 1 - self.arms_probs[arm_index]]
        )
        return result