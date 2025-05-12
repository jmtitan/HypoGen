from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained language model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# Inspect model parameter names and shapes
for name, param in model.named_parameters():
    print(name, param.shape)
