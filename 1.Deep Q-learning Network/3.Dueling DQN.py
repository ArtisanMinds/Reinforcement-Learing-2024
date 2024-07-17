import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, expansion=2):
        super(MLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.ReLU(),
            nn.Linear(hidden_dim * expansion, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.ReLU(),
            nn.Linear(hidden_dim * expansion, 1)
        )

    def forward(self, state):
        backbone = self.backbone(state)
        backbone = backbone.view(backbone.size(0), -1)
        advantage = self.advantage(backbone)
        value = self.value(backbone)
        return value + advantage - advantage.mean()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DuelingDQN:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, buffer_size=10000, target_update_interval=10):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = MLP(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_count = 0

    def update_target_model(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            state = torch.FloatTensor(np.array([state])).to(self.device)
            return self.model(state).argmax().item()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].view(-1, 1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        return nn.MSELoss()(current_q_values, expected_q_values)

    def train_step(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.update_target_model()


env = gym.make('CartPole-v1', render_mode='rgb_array')
max_episodes = 100
max_steps = 500
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 50
agent = DuelingDQN(env)
episode_rewards = []

for episode in tqdm(range(max_episodes), file=sys.stdout):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay))
        action = agent.choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.train_step(batch_size)

        state = next_state
        if done:
            break

    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        tqdm.write(f"Episode {episode}, Reward: {episode_reward}")

    if episode % 10 == 0:
        torch.save(agent.model.state_dict(), 'duelingdqn.pth')

plt.plot(episode_rewards)
plt.title('DQN Training')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

for i in range(500):
    env.render()
    action = agent.choose_action(state, epsilon=0.0)
    state, reward, terminated, truncated, _ = env.step(action)
    print(f'Step: {i}, Action: {action}')
    if terminated or truncated:
        break

env.render()
env.close()
