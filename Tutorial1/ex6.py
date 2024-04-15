import numpy as np
import matplotlib.pyplot as plt

num_bandits = 10
num_steps = 1000
num_episodes = 1000

def choose_action_softmax(q_values, temperature):
    probabilities = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature))
    return np.random.choice(len(q_values), p=probabilities)

def run_bandit_softmax(temperature):
    total_rewards = 0
    optimal_actions = 0
    for _ in range(num_episodes):
        q_true_values = np.random.normal(loc=0, scale=1, size=num_bandits)
        q_values = np.zeros(num_bandits) 
        action_counts = np.zeros(num_bandits)
        for step in range(num_steps):
            action = choose_action_softmax(q_values, temperature)
            reward = np.random.normal(loc=q_true_values[action], scale=1)
            total_rewards += reward
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action]
            optimal_action = np.argmax(q_true_values)
            if action == optimal_action:
                optimal_actions += 1
    return total_rewards / (num_episodes*num_steps), optimal_actions / (num_episodes*num_steps) * 100

temperatures = np.linspace(0.01, 2, num=20)  

average_rewards_softmax = []

for temperature in temperatures:
    print("Current temperature:", temperature)
    avg_reward, _ = run_bandit_softmax(temperature)
    average_rewards_softmax.append(avg_reward)

plt.figure(figsize=(8, 5))
plt.plot(temperatures, average_rewards_softmax)
plt.xlabel('Temperature (tau)')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Temperature (Softmax)')
#display the maximum value of the average reward and its corresponding temperature at the peak
max_reward = max(average_rewards_softmax)
max_reward_index = average_rewards_softmax.index(max_reward)
plt.text(temperatures[max_reward_index], max_reward, f'Max Reward: {max_reward:.2f} Tau: {temperatures[max_reward_index]:.2f}', fontsize=8, color='gray')
plt.grid(True)
plt.ylim(bottom=0, top=1.5)
plt.show()
