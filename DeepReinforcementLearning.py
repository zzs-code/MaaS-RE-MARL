#
# (2025/09) author: Zhengshu Zhou (shu@ertl.jp)
#
# Method: System resilience strategy screening using multi-agent reinforcement learning
# Input:  Resilience strategy pool (a table in PostgreSQL DB)
# Output: Optimal resilience strategies for the user agent
#

import numpy as np
import psycopg2
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim

def get_database_data():
    conn = psycopg2.connect(
        dbname="postgres",
        user="xxx",      # preserve anonymity
        password="xxx",  # preserve anonymity
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT strategy_id, stakeholder_value_user, uncertainty, recovery_time_performance, service_recovery_level, budget 
        FROM public.resilience_strategy
        ORDER BY strategy_id ASC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return np.array(rows)

data = get_database_data()

print("Database Data (Sorted by strategy_id):")
print("strategy_id | stakeholder_value_user | uncertainty | recovery_time_performance | service_recovery_level | budget")
print(data)

strategy_ids = data[:, 0]
stakeholder_value_user = data[:, 1].astype(float)
uncertainty = data[:, 2].astype(float)
recovery_time_performance = data[:, 3].astype(float)
service_recovery_level = data[:, 4].astype(float)
budgets = data[:, 5].astype(float)

values = stakeholder_value_user * uncertainty * recovery_time_performance * service_recovery_level

total_budget = 27

class resilienceEngineeringEnv:
    def __init__(self, capacity, weights, values):
        self.capacity = int(capacity)
        self.weights = list(map(int, weights))
        self.values = values
        self.n_items = len(self.weights)
        self.reset()

    def reset(self):
        self.remaining_capacity = self.capacity
        self.current_item = 0
        return (self.remaining_capacity, self.current_item)

    def step(self, action):
        done = False
        reward = 0

        if action == 1:
            if self.weights[self.current_item] <= self.remaining_capacity:
                self.remaining_capacity -= self.weights[self.current_item]
                reward = float(self.values[self.current_item])
            else:
                reward = -10.0

        self.current_item += 1
        if self.current_item >= self.n_items:
            done = True

        return (self.remaining_capacity, self.current_item), reward, done

    def get_state_vector(self, state):
        remaining_capacity, current_item = state
        if self.capacity > 0:
            r = float(remaining_capacity) / float(self.capacity)
        else:
            r = 0.0
        denom = max(1, self.n_items - 1)
        c = float(current_item) / float(denom)
        return np.array([r, c], dtype=np.float32)

# ---------- DQN：Replay Buffer, Network ----------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(env,
              num_episodes=10000,
              batch_size=64,
              gamma=0.99,
              lr=1e-3,
              buffer_capacity=10000,
              epsilon_start=1.0,
              epsilon_end=0.05,
              epsilon_decay=3000.0,
              target_update=500,
              hidden_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 2
    n_actions = 2

    policy_net = DQN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=n_actions).to(device)
    target_net = DQN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    rewards_per_100 = []
    total_rewards = 0.0
    step_count = 0
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state_vec = env.get_state_vector(state)
        done = False
        episode_reward = 0.0

        while not done:
            step_count += 1
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                s_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    qvals = policy_net(s_tensor)
                    action = int(qvals.argmax(dim=1).item())

            next_state, reward, done = env.step(action)
            next_state_vec = env.get_state_vector(next_state)

            replay_buffer.push(state_vec, action, reward, next_state_vec, done)
            state_vec = next_state_vec
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                states_b = torch.tensor(states_b, dtype=torch.float32, device=device)
                actions_b = torch.tensor(actions_b, dtype=torch.long, device=device).unsqueeze(1)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_b = torch.tensor(next_states_b, dtype=torch.float32, device=device)
                dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy_net(states_b).gather(1, actions_b)
                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(dim=1)[0].unsqueeze(1)
                    target_q = rewards_b + (gamma * next_q_values * (1.0 - dones_b))

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        total_rewards += episode_reward

        if (episode + 1) % 100 == 0:
            avg_reward = total_rewards / 100.0
            rewards_per_100.append(avg_reward)
            total_rewards = 0.0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * (episode / epsilon_decay))

    return policy_net, rewards_per_100

def extract_policy(policy_net, env):
    policy = []
    state = env.reset()
    done = False
    device = next(policy_net.parameters()).device

    while not done:
        state_vec = env.get_state_vector(state)
        s_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qvals = policy_net(s_tensor)
            action = int(qvals.argmax(dim=1).item())
        policy.append(int(action))
        state, reward, done = env.step(action)

    return policy

def evaluate_policy(policy, weights, values, capacity):
    total_weight = 0
    total_value = 0
    for i, action in enumerate(policy):
        if action == 1:
            total_weight += weights[i]
            total_value += values[i]
    if total_weight > capacity:
        return -1
    return total_value

def future_best_strategy(budgets, values, selected_items, weights):
    available_strategies = []
    remaining_budget = sum(budgets)
    for i in range(len(budgets)):
        if selected_items[i] == 0 and weights[i] <= remaining_budget:
            value_per_budget = values[i] / 1.0
            available_strategies.append((i, value_per_budget))
    available_strategies.sort(key=lambda x: x[1], reverse=True)
    if available_strategies:
        best_strategy_id = available_strategies[0][0]
        print(f"\nThe strategy that should be chosen in the future: Strategy ID = S{best_strategy_id + 1}.")
    else:
        print("\nThere are no available strategies to choose from, either because the budget has been exhausted or because there are no cost-effective strategies.")

int_budgets = budgets.astype(int)
env = resilienceEngineeringEnv(total_budget, int_budgets, values)

policy_net, rewards = train_dqn(
    env,
    num_episodes=10000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=10000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=3000.0,
    target_update=500,
    hidden_dim=128
)

print("DQN training finished.")

optimal_policy = extract_policy(policy_net, env)
print("\nOptimal Policy (1-Implement, 0-Drop):")
print(optimal_policy)

optimal_policy_int = [int(a) for a in optimal_policy]

strategy_value = evaluate_policy(optimal_policy_int, int_budgets, values, total_budget)
print(f"\nEvaluated Strategy Value: {strategy_value}")

print("\nAverage Reward per 100 Episodes:")
print(rewards)

future_best_strategy(int_budgets, values, optimal_policy_int, int_budgets)
