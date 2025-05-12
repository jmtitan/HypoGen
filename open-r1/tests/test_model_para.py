from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True).to(device)
model.eval()

# Define the Target and Predictor Networks
state_dim = 3584
hidden_dim = 512

class TargetNet(nn.Module):
    def __init__(self, input_dim=state_dim, hidden_dim=hidden_dim):
        super(TargetNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class PredictorNet(nn.Module):
    def __init__(self, input_dim=state_dim, hidden_dim=hidden_dim):
        super(PredictorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class RNDTrainer:
    def __init__(self, model, layer_choice="layer26_post_ln", state_dim=3584, hidden_dim=512, lr=1e-4):
        self.model = model
        self.layer_choice = layer_choice
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.last_hidden = []

        self.target_net = TargetNet(state_dim, hidden_dim).to(device)
        self.predictor_net = PredictorNet(state_dim, hidden_dim).to(device)

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.hook_handle = self._register_hook()

    def _hook_fn(self, module, input, output):
        with torch.no_grad():
            mean_hidden = output.mean(dim=1)  # [batch_size, hidden_dim]
            self.last_hidden.append(mean_hidden)

    def _register_hook(self):
        if self.layer_choice == "layer25_post_ln":
            return self.model.model.layers[25].post_attention_layernorm.register_forward_hook(self._hook_fn)
        elif self.layer_choice == "layer26_post_ln":
            return self.model.model.layers[26].post_attention_layernorm.register_forward_hook(self._hook_fn)
        elif self.layer_choice == "layer26_o_proj":
            return self.model.model.layers[26].self_attn.o_proj.register_forward_hook(self._hook_fn)
        else:
            raise ValueError("Unsupported layer choice.")

    def get_states(self, prompt_texts, tokenizer):
        self.last_hidden.clear()
        for text in prompt_texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            _ = self.model(**inputs)
        if self.last_hidden:
            return torch.cat(self.last_hidden, dim=0)
        else:
            raise ValueError("Hook did not collect any output.")

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
        mse = (predictor_features - target_features).pow(2).mean(dim=1)
        return mse
