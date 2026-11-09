import random, yaml, numpy as np, torch

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')