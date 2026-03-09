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
        # print(arm_index) 0~4
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


#-----------
# greedy 전략 (지금까지 잘 나온 머신만 계속 누르는 전략)
#-----------
"""
1️ 처음에는 정보가 없으니까 아무거나 몇 번 눌러본다
2️ 각 머신의 평균 보상을 계산한다
3️ 평균이 가장 높은 머신만 계속 선택
# """
# counts      # 몇 번 선택했는지
# sum_rewards # 받은 보상 합
class GreedyAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.sum_rewards = [0] * n_arms

    def select_arm(self):

        # 아직 한 번도 안 뽑은 머신이 있으면 먼저 탐색
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        # 평균 보상 계산
        avg_rewards = [
            self.sum_rewards[i] / self.counts[i]
            for i in range(self.n_arms)
        ]

        # 평균이 가장 높은 머신 선택
        return np.argmax(avg_rewards)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward

#-----------
# ε-greedy 전략 (ε 확률로 랜덤 선택, 나머지는 최고 머신 선택)
#-----------
class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms
        self.sum_rewards = [0] * n_arms

    def select_arm(self):

    # ε 확률로 랜덤 선택
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)

    # Greedy 선택
        avg_rewards = [
            self.sum_rewards[i] / self.counts[i]
            if self.counts[i] > 0 else 0
            for i in range(self.n_arms)
        ]

        return np.argmax(avg_rewards)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward


#-----------
# 테스트 함수
#----------
def run_simulation(agent_class, steps=1000, **kwargs):
    env = BanditEnvironment()
    agent = agent_class(env.n_arms, **kwargs)

    for _ in range(steps):
        arm = agent.select_arm()
        reward = env.react(arm)
        agent.update(arm, reward)

    print(f"{agent_class.__name__} 선택 횟수:", agent.counts)
#----------
# 사용 코드
#-------- 

run_simulation(RandomAgent)
run_simulation(GreedyAgent)
run_simulation(EpsilonGreedyAgent, epsilon=0.1)