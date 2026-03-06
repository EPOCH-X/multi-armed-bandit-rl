import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    def __init__(self):
        self.arms_probs = [0.1, 0.3, 0.8, 0.2, 0.5]
        self.n_arms = len(self.arms_probs)

    def react(self, arm_index):
        result = np.random.choice([1, 0],
        p=[self.arms_probs[arm_index], 1 - self.arms_probs[arm_index]])
        return result

class Agent:
    def __init__(self, n_arms, strategy='random', epsilon=0.1):
        self.n_arms = n_arms
        self.strategy = strategy
        self.epsilon = epsilon
        
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        
        self.total_reward = 0
        self.reward_history = []

    def select_arm(self):
        if self.strategy == 'random':
            return np.random.randint(self.n_arms)
            
        elif self.strategy == 'greedy':
            unexplored = np.where(self.counts == 0)[0]
            if len(unexplored) > 0:
                return np.random.choice(unexplored)
            return np.argmax(self.values)
            
        elif self.strategy == 'epsilon-greedy':
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)
            else:
                if np.sum(self.counts) == 0:
                    return np.random.randint(self.n_arms)
                return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        
        value = self.values[arm]
        self.values[arm] = value + (1.0 / n) * (reward - value)
        
        self.total_reward += reward
        self.reward_history.append(self.total_reward)

def run_experiment():
    env = BanditEnvironment()
    steps = 1000

    # 실험 반복 횟수
    n_runs = 50
    
    strategies = ['Random', 'Greedy', 'e-Greedy (e=0.1)', 'e-Greedy (e=0.01)']
    
    # 누적 데이터를 저장할 딕셔너리
    avg_rewards = {s: np.zeros(steps) for s in strategies}
    avg_counts = {s: np.zeros(env.n_arms) for s in strategies}
    
    # 3번 그래프용 (50번의 실행 각각에서 나온 최대/평균 예측 가치 저장)
    run_max_values = {s: [] for s in strategies}
    run_avg_values = {s: [] for s in strategies}
    
    print(f"총 {n_runs}번의 실험을 진행 중입니다. 기다려주세요.")
    
    for run in range(n_runs):
        agents = {
            'Random': Agent(env.n_arms, strategy='random'),
            'Greedy': Agent(env.n_arms, strategy='greedy'),
            'e-Greedy (e=0.1)': Agent(env.n_arms, strategy='epsilon-greedy', epsilon=0.1),
            'e-Greedy (e=0.01)': Agent(env.n_arms, strategy='epsilon-greedy', epsilon=0.01)
        }
        
        for name, agent in agents.items():
            for _ in range(steps):
                arm = agent.select_arm()
                reward = env.react(arm)
                agent.update(arm, reward)
                
            # 1, 2번 그래프용 데이터 누적
            avg_rewards[name] += np.array(agent.reward_history)
            avg_counts[name] += agent.counts
            
            # 3번 그래프용: 이번 실행(run)에서 이 에이전트가 생각하는 최대 확률과 평균 확률 수집
            run_max_values[name].append(np.max(agent.values))
            run_avg_values[name].append(np.mean(agent.values))
            
    # 전체 횟수(n_runs)로 나누어 평균값 산출
    for name in strategies:
        avg_rewards[name] /= n_runs
        avg_counts[name] /= n_runs

    # 시각화
    plt.figure(figsize=(18, 5))
    
    # 1. 누적 보상 곡선 (50번 평균)
    plt.subplot(1, 3, 1)
    for name in strategies:
        plt.plot(avg_rewards[name], label=name)
    plt.xlabel('Steps')
    plt.ylabel(f'Average Cumulative Reward ({n_runs} runs)')
    plt.title('1. Cumulative Reward (Averaged)')
    plt.legend()
    
    # 2. 머신별 선택 횟수 (50번 평균)
    plt.subplot(1, 3, 2)
    x = np.arange(env.n_arms)
    width = 0.2
    multiplier = 0
    for name in strategies:
        offset = width * multiplier
        plt.bar(x + offset, avg_counts[name], width, label=name)
        multiplier += 1
    plt.xlabel('Arm Index')
    plt.ylabel(f'Average Selection Count')
    plt.title('2. Arm Selection Count (Averaged)')
    plt.xticks(x + width*1.5, [f'Arm {i}' for i in range(env.n_arms)])
    plt.legend()
    
    # 3. 전략별 예측 가치 분포 (50번 실행 결과)
    plt.subplot(1, 3, 3)
    
    for i, name in enumerate(strategies):
        # 겹치지 않게 Jitter 생성 (50개의 점)
        jitter_max = np.random.uniform(-0.1, 0.1, n_runs)
        jitter_avg = np.random.uniform(-0.1, 0.1, n_runs)
        
        # 왼쪽에 최대 예측 가치 분포 (초록색 계열)
        plt.scatter(i - 0.15 + jitter_max, run_max_values[name], color='limegreen', alpha=0.5, s=40,
                    edgecolors='black', linewidths=0.5, label='Max Predicted Value' if i == 0 else "")
        # 오른쪽에 평균 예측 가치 분포 (주황색 계열)
        plt.scatter(i + 0.15 + jitter_avg, run_avg_values[name], color='orange', alpha=0.5, s=40,
                    edgecolors='black', linewidths=0.5, label='Avg Predicted Value' if i == 0 else "")
        
        # 해당 분포들의 중심점(진짜 평균)을 큰 마커로 표시
        plt.scatter([i - 0.15], [np.mean(run_max_values[name])], color='darkgreen', marker='^', s=120, zorder=3)
        plt.scatter([i + 0.15], [np.mean(run_avg_values[name])], color='red', marker='X', s=120, zorder=3)

    plt.xlabel('Strategy', fontsize=11)
    plt.ylabel('Probability Value across 50 runs', fontsize=11)
    plt.title(f'3. Distribution of Max/Avg Predicted Values ({n_runs} runs)', fontsize=13)
    plt.xticks(np.arange(len(strategies)), strategies, rotation=15)
    
    # 80% (실제 최고 확률)과 38% (실제 평균 확률)에 참고용 가로선 긋기
    plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, zorder=1)
    plt.axhline(y=0.38, color='red', linestyle=':', alpha=0.5, zorder=1)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)
    plt.ylim(0, 1.05) 
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_experiment()