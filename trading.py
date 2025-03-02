import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# Define stock symbol and time period
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-02-14"

# Download historical data
data = yf.download(symbol, start=start_date, end=end_date)

# Feature Engineering
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['Returns'] = data['Close'].pct_change()

# Adding RSI (Relative Strength Index)
window = 14
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Adding MACD Indicator
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Drop NaN values and reset index
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Define action space
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Get state function
def get_state(data, index):
    return np.array([
        data.loc[index, 'Close'].item(),
        data.loc[index, 'EMA_5'].item(),
        data.loc[index, 'EMA_20'].item(),
        data.loc[index, 'RSI'].item(),
        data.loc[index, 'MACD'].item(),
        data.loc[index, 'Signal_Line'].item()
    ])

# Trading Environment
class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        self.trading_fee = 0.001  

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return get_state(self.data, self.index)

    def step(self, action):
        price = self.data.loc[self.index, 'Close'].item()
        reward = 0

        # SMART BUY CONDITION (More aggressive, EMA crossover + RSI confirmation)
        if (
            action == 1 
            and self.balance >= price 
            and self.data.loc[self.index, 'EMA_5'].item() > self.data.loc[self.index, 'EMA_20'].item()
            and self.data.loc[self.index, 'RSI'].item() < 50  
        ):
            num_shares = (self.balance * 0.3) // price  
            self.holdings += num_shares
            self.balance -= num_shares * price

        # SMART SELL CONDITION (Sell if price falls below EMA_5)
        elif action == 2 and self.holdings > 0 and price < self.data.loc[self.index, 'EMA_5'].item():
            self.balance += self.holdings * price
            self.holdings = 0  

        # IMPROVED REWARD FUNCTION
        next_price = self.data.loc[self.index + 1, 'Close'].item() if self.index + 1 < len(self.data) else price
        portfolio_value = self.balance + (self.holdings * next_price)
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        reward = np.clip(reward, -1, 1)  

        self.index += 1
        done = self.index >= len(self.data) - 1
        next_state = get_state(self.data, self.index) if not done else None

        return next_state, reward, done, {}

# Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)  
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  
        self.gamma = 0.99  
        self.epsilon = 1.0  
        self.epsilon_min = 0.05  
        self.epsilon_decay = 0.995  
        self.learning_rate = 0.001  
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_tensor = self.model(state_tensor).clone().detach()
            target_tensor[0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize Trading Environment and agent before training
env = TradingEnvironment(data)  
agent = DQNAgent(state_size=6, action_size=3)

# Train the agent
portfolio_values = []
rewards_per_episode = []

for episode in range(500):
    state = env.reset()
    done = False
    total_reward = 0  

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state if next_state is not None else state
        total_reward += reward  

    agent.replay(32)
    portfolio_values.append(env.balance)  
    rewards_per_episode.append(total_reward)  
    print(f"Episode {episode+1}/500, Total Reward: {total_reward:.2f}")

print("Training Complete!")

final_balance = env.balance  # Get final balance after training
profit = final_balance - env.initial_balance  # Calculate total profit

print(f"💰 Total Profit: ${profit:.2f}")
print(f"📈 Final Balance: ${final_balance:.2f}")

# Plot Portfolio Growth
plt.figure(figsize=(10,5))
plt.plot(portfolio_values, label="Portfolio Value ($)")
plt.xlabel("Episodes")
plt.ylabel("Final Portfolio Balance ($)")
plt.title("Portfolio Growth Over Time")
plt.legend()
plt.show()

# Plot Reward Progression
plt.figure(figsize=(10,5))
plt.plot(rewards_per_episode, label="Reward per Episode", color='orange')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward Progression Over Episodes")
plt.legend()
plt.show()
