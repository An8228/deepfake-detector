import time
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from src.utils.common import load_config, seed_everything, get_device
from src.data.dataset import FaceDataset
from src.models.model import build_model


def main():
    cfg = load_config()
    seed_everything(cfg['seed'])
    dev = get_device()

    model = build_model(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print('device:', dev, '| params (M):', round(n_params, 2))

    ds = FaceDataset('train', cfg)
    g = torch.Generator().manual_seed(cfg['seed'])
    idx = torch.randperm(len(ds), generator=g)[:128].tolist()
    print('fake fraction in subset:', round(float(ds.df.loc[idx, 'label'].mean()), 2))
    loader = DataLoader(Subset(ds, idx), batch_size=32, shuffle=True,
                        num_workers=cfg['num_workers'])

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])

    for epoch in range(5):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += ((logits > 0).float() == y).sum().item()
            total += len(y)
        print(f'epoch {epoch + 1}: loss {loss_sum / total:.3f} '
              f'acc {correct / total:.3f} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()