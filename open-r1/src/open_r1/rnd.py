from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True).to(device)
# model.eval()

# Define the Target and Predictor Networks
STATE_DIM = 1536
HIDDEN_DIM = 512

class TargetNet(nn.Module):
    def __init__(self, input_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super(TargetNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class PredictorNet(nn.Module):
    def __init__(self, input_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super(PredictorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class RNDTrainer:
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, lr=1e-4):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.last_hidden = []

        self.target_net = TargetNet(state_dim, hidden_dim).to(device)
        # print(self.target_net)
        self.predictor_net = PredictorNet(state_dim, hidden_dim).to(device)

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update(self, states):
        self.target_net.eval()
        self.predictor_net.train()

        self.optimizer.zero_grad()
        with torch.no_grad():
            target_features = self.target_net(states)
        predictor_features = self.predictor_net(states)

        loss = self.loss_fn(predictor_features, target_features)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_intrinsic_reward(self, states):
        with torch.no_grad():
            target_features = self.target_net(states)
            predictor_features = self.predictor_net(states)
        mse = (predictor_features - target_features).pow(2).mean(dim=1)  # (batch_size,)

        # Normalize to [0, 1] based on current batch
        min_val = mse.min()
        max_val = mse.max()
        normalized = (mse - min_val) / (max_val - min_val + 1e-8)  # avoid division by 0
        return normalized.mean().item()

