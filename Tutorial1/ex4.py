import numpy as np
import matplotlib.pyplot as plt

num_bandits = 10
num_steps = 1000
num_episodes = 1000

def choose_action(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(num_bandits)
    else:
        return np.random.choice(np.flatnonzero(q_values == q_values.max()))

def run_bandit(epsilon):
    total_rewards = 0
    optimal_actions = 0
    for _ in range(num_episodes):
        q_true_values = np.random.normal(loc=0, scale=1, size=num_bandits)
        q_values = np.zeros(num_bandits) 
        action_counts = np.zeros(num_bandits)
        optimal_action = np.argmax(q_true_values)
        for step in range(num_steps):
            action = choose_action(q_values, epsilon)
            reward = np.random.normal(loc=q_true_values[action], scale=1)
            total_rewards += reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
            if action == optimal_action:
                optimal_actions += 1
    return total_rewards / (num_episodes*num_steps), optimal_actions / (num_episodes*num_steps) * 100


epsilons = np.logspace(-4, 0, num=30)[1:]
#instert 0 to the beginning of the array
epsilons = np.insert(epsilons, 0, 0)

average_rewards = []
percentage_optimal_actions = []

for epsilon in epsilons:
    print("Current epsilon:", epsilon)
    avg_reward, opt_actions = run_bandit(epsilon)
    average_rewards.append(avg_reward)
    percentage_optimal_actions.append(opt_actions)

plt.figure(figsize=(10, 5))
# add main title
plt.suptitle('Epsilon-Greedy Bandit Algorithm Ex4')


plt.subplot(1, 2, 1)
plt.plot(epsilons, average_rewards)
plt.xlabel('Epsilon')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Epsilon')

#display the maximum value of the average reward and its corresponding epsilon
max_reward = max(average_rewards)
max_reward_index = average_rewards.index(max_reward)
plt.text(epsilons[max_reward_index], max_reward, f'Max Reward: {max_reward:.2f} Eps: {epsilons[max_reward_index]:.2f}', fontsize=8, color='gray')

plt.subplot(1, 2, 2)
plt.plot(epsilons, percentage_optimal_actions)
plt.xlabel('Epsilon')
plt.ylabel('Percentage of Optimal Actions Taken')
plt.title('Percentage of Optimal Actions Taken vs Epsilon')

#display the maximum value of the percentage of optimal actions and its corresponding epsilon
max_optimal_actions = max(percentage_optimal_actions)
max_optimal_actions_index = percentage_optimal_actions.index(max_optimal_actions)
plt.text(epsilons[max_optimal_actions_index], max_optimal_actions, f'Max num Optimal Actions: {max_optimal_actions:.2f} Eps: {epsilons[max_optimal_actions_index]:.2f}', fontsize=8, color='gray')
plt.tight_layout()
plt.show()
