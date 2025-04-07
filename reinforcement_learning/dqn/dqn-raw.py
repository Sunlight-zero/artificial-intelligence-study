"""
模型的这一训练方法无法完全收敛
"""

import gymnasium as gym
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


num_epochs = 10000
update_period = 50
buffer_min = (1 << 14)
max_explore = 10000
epsilon = 0.1
gamma = 0.99
lr = 5e-3
device = "cuda:1"
batch_size = 256
save_path = "./ckpt/rl/dqn-first.ckpt"


class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

q_net = DeepQNetwork()
target_q_net = DeepQNetwork()
target_q_net.eval()

class ReplayBuffer(Dataset):
    def __init__(self):
        super().__init__()
        self.buffer = []
    
    def clear(self):
        self.buffer.clear()
    
    def append(self, item):
        self.buffer.append(item)
    
    def __getitem__(self, index):
        return self.buffer[index]
    
    def __len__(self):
        return len(self.buffer)

q_net.to(device)
target_q_net.to(device)
env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")
buffer = ReplayBuffer()
train_dataloader: DataLoader
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

to_float = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
total_return = 0

with tqdm(range(num_epochs)) as pbar:
    for epoch in pbar:
        loss_metric = 0.
        metric_count = 0
        if epoch % update_period == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            # Sample
            buffer.clear()
            while len(buffer) <= buffer_min:
                state, info = env.reset()
                for _ in range(max_explore):
                    if np.random.rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        qnet_output = target_q_net(torch.tensor(state, device=device, dtype=torch.float32))
                        action = int(torch.argmax(qnet_output))
                    next_state, reward, terminated, truncated, info = env.step(action)
                    # if terminated:
                    #     reward = -1
                    done = terminated or truncated
                    buffer.append((to_float(state), action, 
                                   to_float(reward), to_float(next_state),
                                   to_float(done)))
                    if done:
                        break
                    state = next_state
            
            train_dataloader = DataLoader(buffer, batch_size=64, shuffle=True)
        
        for state, action, reward, next_state, done in train_dataloader:
            optimizer.zero_grad()
            this_batch_size = state.shape[0]
            with torch.no_grad():
                target = reward + (1 - done) * gamma * torch.max(target_q_net(next_state), dim=-1).values
            loss = torch.nn.functional.mse_loss(q_net(state)[torch.arange(this_batch_size), action], target)
            loss.backward()
            loss_metric += float(loss)
            metric_count += 1
            optimizer.step()

        if (epoch + 1) % update_period == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            state, info = test_env.reset(seed=42)

            total_return = 0
            done = False
            while not done:
                action = np.argmax(target_q_net(to_float(state)).cpu().detach().numpy())
                state, reward, terminated, truncated, info = test_env.step(action)
                total_return += reward
                # 若终止或截断，重置环境
                done = terminated or truncated
            test_env.close()

        pbar.set_postfix({"loss": loss_metric / metric_count, "return": total_return})

torch.save(q_net.state_dict(), save_path)

input("Press ENTER to continue")
# 初始化环境
target_q_net.to("cpu")
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset(seed=42)

# 运行 1000 个时间步
ret = 0

for _ in range(1000):
    action = np.argmax(target_q_net(torch.tensor(state)).detach().numpy())
    state, reward, terminated, truncated, info = env.step(action)
    ret += 1
    
    # 若终止或截断，重置环境
    if terminated or truncated:
        print(f"Return: {ret}")
        ret = 0
        observation, info = env.reset()
