from unittest import result

import numpy as np

class BanditEnvironment:
    def __init__(self):
        # 5개 머신의 실제 당첨 확률 (비밀 설정)
        # 예: 2번 인덱스(3번째 머신)가 80% 확률로 가장 높음
        self.arms_probs = [0.1, 0.3, 0.8, 0.2, 0.5]
        self.n_arms = len(self.arms_probs)
        # print(self.n_arms)
    def react(self, arm_index):
        # print(arm_index) 1~5 
        """
        선수가 레버를 당겼을 때(arm_index) 결과를 반환합니다.
        p의 확률로 1, 1-p의 확률로 0이 나옵니다.
        """
        result = np.random.choice([1, 0],
        p=[self.arms_probs[arm_index], 1 - self.arms_probs[arm_index]]
        )
        return result
#-----------
# 랜덤 전략
#-----------
class RandomAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms   # 각 머신 선택 횟수 기록 
        # print(self.counts) [0,0,0,0,0] 

    def select_arm(self):
        # 0 1 2 3 4 중 랜덤 선택
        # np.random.randint(N) = 0 ~ N-1 중 랜덤 정수 1개
        return np.random.randint(self.n_arms)

    def update(self, arm, reward):
        # 선택 횟수만 기록
        self.counts[arm] += 1


# 테스트
env = BanditEnvironment()
agent = RandomAgent(env.n_arms)

for _ in range(1000):
    arm = agent.select_arm()
    reward = env.react(arm) # 머신 인덱스 받음 
    agent.update(arm, reward)

print("머신 선택 횟수:", agent.counts)