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
    

def run_random(env: BanditEnvironment, n_steps: int = 1000, seed: int = 0):
    np.random.seed(seed)
    n_arms = env.n_arms

    counts = np.zeros(n_arms, dtype=int)
    rewards_sum = np.zeros(n_arms, dtype=float)

    total_reward = 0
    for _ in range(n_steps):
        arm = np.random.randint(n_arms)  # 무작위 선택
        r = env.react(arm)

        counts[arm] += 1
        rewards_sum[arm] += r
        total_reward += r

    estimates = np.divide(rewards_sum, counts, out=np.zeros_like(rewards_sum), where=counts > 0)
    return total_reward, counts, estimates


def run_greedy(env: BanditEnvironment, n_steps: int = 1000, seed: int = 0):
    np.random.seed(seed)
    n_arms = env.n_arms

    counts = np.zeros(n_arms, dtype=int)
    rewards_sum = np.zeros(n_arms, dtype=float)

    total_reward = 0

    # 초기에는 각 머신을 최소 1번씩 당겨서 평균을 만들고 시작(안 그러면 전부 0이라 선택이 꼬일 수 있음)
    for arm in range(n_arms):
        if arm >= n_steps:
            break
        r = env.react(arm)
        counts[arm] += 1
        rewards_sum[arm] += r
        total_reward += r

    for _ in range(n_steps - min(n_arms, n_steps)):
        estimates = np.divide(rewards_sum, counts, out=np.zeros_like(rewards_sum), where=counts > 0)
        arm = int(np.argmax(estimates))  # 현재 평균이 가장 큰 머신만 선택
        r = env.react(arm)

        counts[arm] += 1
        rewards_sum[arm] += r
        total_reward += r

    estimates = np.divide(rewards_sum, counts, out=np.zeros_like(rewards_sum), where=counts > 0)
    return total_reward, counts, estimates


def run_epsilon_greedy(env: BanditEnvironment, epsilon: float = 0.1, n_steps: int = 1000, seed: int = 0):
    np.random.seed(seed)
    n_arms = env.n_arms

    counts = np.zeros(n_arms, dtype=int)
    rewards_sum = np.zeros(n_arms, dtype=float)

    total_reward = 0

    # Greedy와 동일하게 초기 1회씩은 당겨서 시작(선택 안정화)
    for arm in range(n_arms):
        if arm >= n_steps:
            break
        r = env.react(arm)
        counts[arm] += 1
        rewards_sum[arm] += r
        total_reward += r

    for _ in range(n_steps - min(n_arms, n_steps)):
        # 탐험(Exploration): epsilon 확률로 랜덤 선택
        if np.random.rand() < epsilon:
            arm = np.random.randint(n_arms)
        # 활용(Exploitation): (1-epsilon) 확률로 현재 최고 평균 선택
        else:
            estimates = np.divide(rewards_sum, counts, out=np.zeros_like(rewards_sum), where=counts > 0)
            arm = int(np.argmax(estimates))

        r = env.react(arm)
        counts[arm] += 1
        rewards_sum[arm] += r
        total_reward += r

    estimates = np.divide(rewards_sum, counts, out=np.zeros_like(rewards_sum), where=counts > 0)
    return total_reward, counts, estimates


def print_result(name: str, total_reward: int, counts: np.ndarray, estimates: np.ndarray, env: BanditEnvironment):
    best_arm = int(np.argmax(env.arms_probs))
    print(f"\n=== {name} ===")
    print(f"총 보상(1의 개수): {total_reward} / 평균 보상: {total_reward / counts.sum():.3f}")
    print(f"선택 횟수 counts: {counts.tolist()}")
    print(f"추정 확률 estimates: {[round(x, 3) for x in estimates.tolist()]}")
    print(f"(정답) 실제 확률 probs: {env.arms_probs}  | 최적 머신: {best_arm}번(0-index)")


if __name__ == "__main__":
    env = BanditEnvironment()
    n_steps = 1000

    # Random
    r_total, r_counts, r_est = run_random(env, n_steps=n_steps, seed=42)
    print_result("Random", r_total, r_counts, r_est, env)

    # Greedy
    g_total, g_counts, g_est = run_greedy(env, n_steps=n_steps, seed=42)
    print_result("Greedy", g_total, g_counts, g_est, env)

    # Epsilon-Greedy (예: 10%, 1%)
    e1_total, e1_counts, e1_est = run_epsilon_greedy(env, epsilon=0.1, n_steps=n_steps, seed=42)
    print_result("Epsilon-Greedy (epsilon=0.1)", e1_total, e1_counts, e1_est, env)

    e2_total, e2_counts, e2_est = run_epsilon_greedy(env, epsilon=0.01, n_steps=n_steps, seed=42)
    print_result("Epsilon-Greedy (epsilon=0.01)", e2_total, e2_counts, e2_est, env)