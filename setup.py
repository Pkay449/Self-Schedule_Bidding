import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

#------------------------
# Network Definitions
#------------------------

class QNetworkDA(nn.Module):
    def __init__(self, state_dim=840, action_dim=24, hidden_dim=256):
        super(QNetworkDA, self).__init__()
        # Q takes state and action as input
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # state: (batch, state_dim)
        # action: (batch, action_dim)
        x = torch.cat([state, action.float()], dim=-1)
        q = self.fc(x)
        return q


class QNetworkID(nn.Module):
    def __init__(self, state_dim=864, action_dim=96, hidden_dim=256):
        super(QNetworkID, self).__init__()
        # Q takes state and action as input
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action.float()], dim=-1)
        q = self.fc(x)
        return q


class PolicyDA(nn.Module):
    def __init__(self, state_dim=840, action_dim=24, num_action_values=10):
        """
        Policy outputs logits for each action dimension.
        num_action_values: number of discrete choices for each action dimension.
        """
        super(PolicyDA, self).__init__()
        self.num_action_values = num_action_values
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * num_action_values)
        )

    def forward(self, state):
        # state: (batch, state_dim)
        logits = self.fc(state)
        # Reshape to (batch, action_dim, num_action_values)
        logits = logits.view(-1, 24, self.num_action_values)
        return logits

    def sample_action(self, state):
        # state: (1, state_dim)
        with torch.no_grad():
            logits = self.forward(state) # (1, 24, num_action_values)
            # Sample or argmax per action dimension
            # For simplicity, we take argmax:
            chosen = torch.argmax(logits, dim=-1) # (1, 24)
        return chosen.squeeze(0)


class PolicyID(nn.Module):
    def __init__(self, state_dim=864, action_dim=96, num_action_values=10):
        super(PolicyID, self).__init__()
        self.num_action_values = num_action_values
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * num_action_values)
        )

    def forward(self, state):
        logits = self.fc(state)
        logits = logits.view(-1, 96, self.num_action_values)
        return logits

    def sample_action(self, state):
        with torch.no_grad():
            logits = self.forward(state) # (1, 96, num_action_values)
            chosen = torch.argmax(logits, dim=-1) # (1, 96)
        return chosen.squeeze(0)


#------------------------
# Offline Dataset Loaders
#------------------------

class DayAheadDataset(Dataset):
    def __init__(self, path):
        df = torch.load(path) if path.endswith('.pt') else torch.load(path)
        self.states = torch.tensor(np.stack(df['state'].values), dtype=torch.float32)
        self.actions = torch.tensor(np.stack(df['action'].values), dtype=torch.int64)
        self.rewards = torch.tensor(df['reward'].values, dtype=torch.float32).unsqueeze(-1)
        self.next_states = torch.tensor(np.stack(df['next_state'].values), dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx])

class IntraDayDataset(Dataset):
    def __init__(self, path):
        df = torch.load(path) if path.endswith('.pt') else torch.load(path)
        self.states = torch.tensor(np.stack(df['state'].values), dtype=torch.float32)
        self.actions = torch.tensor(np.stack(df['action'].values), dtype=torch.int64)
        self.rewards = torch.tensor(df['reward'].values, dtype=torch.float32).unsqueeze(-1)
        self.next_states = torch.tensor(np.stack(df['next_state'].values), dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx])

#------------------------
# Training Setup
#------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_da = QNetworkDA().to(device)
q_id = QNetworkID().to(device)

# Target networks if you want stable training (optional)
q_da_target = QNetworkDA().to(device)
q_id_target = QNetworkID().to(device)
q_da_target.load_state_dict(q_da.state_dict())
q_id_target.load_state_dict(q_id.state_dict())

policy_da = PolicyDA().to(device)
policy_id = PolicyID().to(device)

optimizer_q_da = optim.Adam(q_da.parameters(), lr=1e-3)
optimizer_q_id = optim.Adam(q_id.parameters(), lr=1e-3)

# Hyperparameters
gamma = 0.99
batch_size = 64
num_epochs = 10

da_dataset = DayAheadDataset("Results/offline_dataset_day_ahead.pkl")
id_dataset = IntraDayDataset("Results/offline_dataset_intraday.pkl")

da_loader = DataLoader(da_dataset, batch_size=batch_size, shuffle=True)
id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=True)

#------------------------
# Training Loops
#------------------------

for epoch in range(num_epochs):
    # Train Q_ID first (intraday)
    q_id.train()
    for s_id, a_id, r_id, s_id_next in id_loader:
        s_id = s_id.to(device)
        a_id = a_id.to(device)
        r_id = r_id.to(device)
        s_id_next = s_id_next.to(device)

        # Compute target Q-values
        # For intraday, the episode presumably ends after this stage or you can consider no future beyond ID.
        # If you do have another stage, you must incorporate it. Otherwise:
        with torch.no_grad():
            # If no next stage, target = r_id
            q_target = r_id  # Adjust if there's a next stage or terminal conditions

        q_estimate = q_id(s_id, a_id)
        
        loss_id = nn.MSELoss()(q_estimate, q_target)
        optimizer_q_id.zero_grad()
        loss_id.backward()
        optimizer_q_id.step()

    # Now train Q_DA (day-ahead)
    q_da.train()
    for s_da, a_da, r_da, s_da_next in da_loader:
        s_da = s_da.to(device)
        a_da = a_da.to(device)
        r_da = r_da.to(device)
        s_da_next = s_da_next.to(device)

        # Q_DA target should include the intraday optimal action return
        # We must get the expected intraday Q-value from the next state.
        # The next state for DA is s_da_next which presumably includes the updated info for intraday step.
        # For that next state, we use the ID policy to find the best a_id.
        
        with torch.no_grad():
            # Sample or choose best a_id from policy_id
            # This is a heuristic: we pick the greedy action for simplicity:
            logits_id = policy_id(s_da_next)
            best_id_actions = torch.argmax(logits_id, dim=-1)  # shape: (batch, 96)
            
            q_id_values = q_id_target(s_da_next, best_id_actions)
            q_target_da = r_da + gamma * q_id_values

        q_da_values = q_da(s_da, a_da)

        loss_da = nn.MSELoss()(q_da_values, q_target_da)
        optimizer_q_da.zero_grad()
        loss_da.backward()
        optimizer_q_da.step()

    # Optional: Update target networks
    with torch.no_grad():
        for param, target_param in zip(q_da.parameters(), q_da_target.parameters()):
            target_param.data.mul_(0.995)
            target_param.data.add_(0.005 * param.data)

        for param, target_param in zip(q_id.parameters(), q_id_target.parameters()):
            target_param.data.mul_(0.995)
            target_param.data.add_(0.005 * param.data)

    print(f"Epoch {epoch+1}/{num_epochs}, DA Loss: {loss_da.item():.4f}, ID Loss: {loss_id.item():.4f}")

#------------------------
# Using the Policy:
#------------------------
# Given a new day-ahead state s_da (1D tensor of shape (840,)), we can select a DA action:
s_da_example = torch.randn(1, 840).to(device) # dummy example
da_action = policy_da.sample_action(s_da_example)

# After the environment reveals intraday info, we form the intraday state s_id (1D tensor of shape (864,))
s_id_example = torch.randn(1, 864).to(device)
id_action = policy_id.sample_action(s_id_example)

# The chosen actions are integer tensors corresponding to the chosen discrete action indices for each dimension.
