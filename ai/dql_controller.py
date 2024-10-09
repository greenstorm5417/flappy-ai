import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import logging

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class DQNController:
    def __init__(
        self, 
        input_size=6,  
        action_size=2,
        hidden_size=64,
        gamma=0.95,
        learning_rate=1e-4,
        replay_memory_size=20000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=2000,
        is_training=True
    ):
        self.is_training = is_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(input_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if self.is_training:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            if random.random() < epsilon:
                return random.choice([0, 1])  # Random action
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state)
                    return q_values.argmax().item()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to learn from

        # Sample a random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        state_batch = np.array([s for (s, a, r, s_next, d) in batch])
        next_state_batch = np.array([s_next for (s, a, r, s_next, d) in batch])

        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)

        action_batch = torch.LongTensor([a for (s, a, r, s_next, d) in batch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([r for (s, a, r, s_next, d) in batch]).to(self.device)
        done_batch = torch.FloatTensor([d for (s, a, r, s_next, d) in batch]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_action = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(1, next_action).squeeze(1)

        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5) 
        
        loss = nn.functional.mse_loss(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network(tau=0.05)  # Soft update

        # Log or print Q-values and loss periodically
        if self.steps_done % 100 == 0:
            avg_q_value = q_values.mean().item()
            logging.info(f"Step {self.steps_done}: Avg Q-value: {avg_q_value}, Loss: {loss.item()}")
            print(f"Step {self.steps_done}: Avg Q-value: {avg_q_value}, Loss: {loss.item()}")

    def update_target_network(self, tau=0.01):
        """Soft update target network."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {file_path}")

    def reset(self):
        pass  # No action needed for reset in this implementation
