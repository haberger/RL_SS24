import numpy as np
import matplotlib.pyplot as plt

num_bandits = 10
num_steps = 1000
num_episodes = 1000

def choose_action_ucb(q_values, action_counts, t, c):
    ucb_values = q_values + c * np.sqrt(np.log(t) / (action_counts + 1e-7))
    return np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))

def run_bandit_ucb(c):
    total_rewards = 0
    optimal_actions = 0
    for _ in range(num_episodes):
        q_true_values = np.random.normal(loc=0, scale=1, size=num_bandits)
        q_values = np.zeros(num_bandits) 
        action_counts = np.zeros(num_bandits)
        for step in range(num_steps):
            action = choose_action_ucb(q_values, action_counts, step+1, c) 
            reward = np.random.normal(loc=q_true_values[action], scale=1)
            total_rewards += reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
            optimal_action = np.argmax(q_true_values)
            if action == optimal_action:
                optimal_actions += 1
    return total_rewards / (num_episodes*num_steps), optimal_actions / (num_episodes*num_steps) * 100

cs = np.linspace(0, 3, num=20)  # Values of c to test


average_rewards_ucb = []

for c in cs:
    print("Current c:", c)
    avg_reward, _ = run_bandit_ucb(c)
    average_rewards_ucb.append(avg_reward)

plt.figure(figsize=(8, 5))

plt.ylim(bottom=0, top=1.65)
plt.plot(cs, average_rewards_ucb)
plt.xlabel('c')
plt.ylabel('Average Reward')
plt.title('Average Reward vs c (UCB)')
#display the maximum value of the average reward and its corresponding c at the peak
max_reward = max(average_rewards_ucb)
max_reward_index = average_rewards_ucb.index(max_reward)
plt.text(cs[max_reward_index], max_reward, f'Max Reward: {max_reward:.2f} c: {cs[max_reward_index]:.2f}', fontsize=8, color='gray')

plt.show()
