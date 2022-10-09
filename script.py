import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GridWorld import GridWorld


# Maybe talk on why we need exploration?
def e_greedy_policy(q_values, state, epsilon=0.1):
    '''
    Choose an action based on a epsilon greedy policy.
    A random action is selected with epsilon probability, else select the best action.
    '''
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])


# Q_Learning
def q_learning(env, num_episodes=5, render=True, exploration_rate=0.1, learning_rate=0.5, gamma=0.9):
    q_values = np.zeros((num_states, num_actions))
    ep_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0

        while not done:
            # Choose action
            action = e_greedy_policy(q_values, state, exploration_rate)
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward
            # Update q_values
            td_target = reward + gamma * np.max(q_values[next_state])
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            # Update state
            state = next_state

            if render:
                env.render(q_values, action=actions[action], colorize_q=True)

        ep_rewards.append(reward_sum)

    return ep_rewards, q_values


# Sarsa
def sarsa(env, num_episodes=5, render=True, exploration_rate=0.1, learning_rate=0.5, gamma=0.9):
    q_values_sarsa = np.zeros((num_states, num_actions))
    ep_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        # Choose action
        action = e_greedy_policy(q_values_sarsa, state, exploration_rate)

        while not done:
            # Do the action
            next_state, reward, done = env.step(action)
            reward_sum += reward

            # Choose next action
            next_action = e_greedy_policy(q_values_sarsa, next_state, exploration_rate)
            # Next q value is the value of the next action
            td_target = reward + gamma * q_values_sarsa[next_state][next_action]
            td_error = td_target - q_values_sarsa[state][action]
            # Update q value
            q_values_sarsa[state][action] += learning_rate * td_error

            # Update state and action
            state = next_state
            action = next_action

            if render:
                env.render(q_values, action=actions[action], colorize_q=True)

        ep_rewards.append(reward_sum)

    return ep_rewards, q_values_sarsa


# Visualization
def play(q_values):
    env = GridWorld()
    state = env.reset()
    done = False

    while not done:
        # Select action
        action = e_greedy_policy(q_values, state, 0.0)
        # Do the action
        next_state, reward, done = env.step(action)

        # Update state and action
        state = next_state

        env.render(q_values=q_values, action=actions[action], colorize_q=True)


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

# Environment Creation
env = GridWorld()

# We need a table of values that maps each state-action pair to a value, we'll create such table and
# initialize all values to zero (or to a random value)

# The number of states in simply the number of "squares" in our grid world, in this case 4 * 12
num_states = 4 * 12

# We have 4 possible actions, up, down, right and left
num_actions = 4

# Initial Q Values
q_values = np.zeros((num_states, num_actions))

df = pd.DataFrame(q_values, columns=[' up ', 'down', 'right', 'left'])
df.index.name = 'States'
df.head()

q_learning_rewards, q_values = q_learning(env, gamma=0.9, learning_rate=1, render=True)
env.render(q_values, colorize_q=True)
np.mean(q_learning_rewards)

q_learning_rewards, _ = zip(*[q_learning(env, render=False, exploration_rate=0.1, learning_rate=1) for _ in range(10)])
avg_rewards = np.mean(q_learning_rewards, axis=0)
mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)

fig, ax = plt.subplots()
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
ax.plot(avg_rewards)
ax.plot(mean_reward, 'g--')

print('Mean Reward: {}'.format(mean_reward[0]))

sarsa_rewards, q_values_sarsa = sarsa(env, render=True, learning_rate=0.5, gamma=0.99)
np.mean(sarsa_rewards)

sarsa_rewards, _ = zip(*[sarsa(env, render=False, exploration_rate=0.2) for _ in range(100)])

avg_rewards = np.mean(sarsa_rewards, axis=0)
mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)

fig, ax = plt.subplots()
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
ax.plot(avg_rewards)
ax.plot(mean_reward, 'g--')

print('Mean Reward: {}'.format(mean_reward[0]))

play(q_values)
play(q_values_sarsa)
