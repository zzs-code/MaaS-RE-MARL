#
# (2024/09) author: Zhengshu Zhou (shu@ertl.jp)
#
# Method: System resilience strategy screening using multi-agent reinforcement learning
# Input:  Resilience strategy pool (a table in PostgreSQL DB)
# Output: Optimal resilience strategies for the star agent
#

import numpy as np
import psycopg2

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
        SELECT strategy_id, stakeholder_value_total, uncertainty, recovery_time_performance, service_recovery_level, budget 
        FROM public.resilience_strategy
        ORDER BY strategy_id ASC
    """)
    
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return np.array(rows)

data = get_database_data()

print("Database Data (Sorted by strategy_id):")
print("strategy_id | stakeholder_value_total | uncertainty | recovery_time_performance | service_recovery_level | budget")
print(data)

strategy_ids = data[:, 0]
stakeholder_value_total = data[:, 1].astype(float)
uncertainty = data[:, 2].astype(float)
recovery_time_performance = data[:, 3].astype(float)
service_recovery_level = data[:, 4].astype(float)
budgets = data[:, 5].astype(float)

values = stakeholder_value_total * uncertainty * recovery_time_performance * service_recovery_level

total_budget = 27

class resilienceEngineeringEnv:
    def __init__(self, capacity, weights, values):
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.n_items = len(weights)
        self.reset()

    def reset(self):
        self.remaining_capacity = self.capacity
        self.current_item = 0
        return self.remaining_capacity, self.current_item

    def step(self, action):
        done = False
        reward = 0

        if action == 1:
            if self.weights[self.current_item] <= self.remaining_capacity:
                self.remaining_capacity -= self.weights[self.current_item]
                reward = self.values[self.current_item]
            else:
                reward = -10

        self.current_item += 1

        if self.current_item >= self.n_items:
            done = True

        return (self.remaining_capacity, self.current_item), reward, done

def q_learning(env, num_episodes=20000, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
    Q = np.zeros((env.capacity + 1, env.n_items, 2))
    rewards_per_100_episodes = []

    total_rewards = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            remaining_capacity, current_item = state

            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1])
            else:
                action = np.argmax(Q[remaining_capacity, current_item])

            next_state, reward, done = env.step(action)
            next_remaining_capacity, next_item = next_state
            total_reward += reward

            if not done:
                best_next_action = np.argmax(Q[next_remaining_capacity, next_item])
                Q[remaining_capacity, current_item, action] += learning_rate * (
                    reward + discount_factor * Q[next_remaining_capacity, next_item, best_next_action] - Q[remaining_capacity, current_item, action]
                )
            else:
                Q[remaining_capacity, current_item, action] += learning_rate * (reward - Q[remaining_capacity, current_item, action])

            state = next_state

        total_rewards += total_reward

        if (episode + 1) % 100 == 0:
            avg_reward = total_rewards / 100
            rewards_per_100_episodes.append(avg_reward)
            total_rewards = 0

    return Q, rewards_per_100_episodes

def extract_policy(Q, env):
    policy = []
    state = (env.capacity, 0)
    
    while state[1] < env.n_items:
        remaining_capacity, current_item = state
        action = np.argmax(Q[remaining_capacity, current_item])
        policy.append(int(action))
        
        if action == 1 and remaining_capacity >= env.weights[current_item]:
            remaining_capacity -= env.weights[current_item]

        current_item += 1
        state = (remaining_capacity, current_item)
    
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

Q, rewards = q_learning(env, num_episodes=20000)

print("Q-table:")
print(Q)

optimal_policy = extract_policy(Q, env)
print("\nOptimal Policy (1-Implement, 0-Drop):")
print(optimal_policy)

optimal_policy_int = [int(action) for action in optimal_policy]

strategy_value = evaluate_policy(optimal_policy_int, int_budgets, values, total_budget)
print(f"\nEvaluated Strategy Value: {strategy_value}")

print("\nAverage Reward per 100 Episodes:")
print(rewards)

future_best_strategy(int_budgets, values, optimal_policy_int, int_budgets)
