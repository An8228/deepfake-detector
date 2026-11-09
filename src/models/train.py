import argparse
import time
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.utils.common import load_config, seed_everything, get_device
from src.data.dataset import make_loader
from src.models.model import build_model


def train_one_epoch(model, loader, crit, opt, dev, epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc=f'epoch {epoch} train', leave=False):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        logits = model(x).squeeze(1)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit, dev, desc='val'):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    probs, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(dev), y.to(dev)
        logits = model(x).squeeze(1)
        loss_sum += crit(logits, y).item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        total += len(y)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    auc = roc_auc_score(torch.cat(labels).numpy(), torch.cat(probs).numpy())
    return loss_sum / total, correct / total, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    epochs = args.epochs or cfg['epochs']
    seed_everything(cfg['seed'])
    dev = get_device()

    out = Path(cfg['paths']['outputs'])
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path, hist_path = out / 'last.pt', out / 'best.pt', out / 'history.csv'

    train_loader = make_loader('train', cfg)
    val_loader = make_loader('val', cfg)

    model = build_model(cfg).to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch, best_auc = 0, 0.0
    if last_path.exists():
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch, best_auc = ck['epoch'], ck['best_auc']
        print(f'Resuming after epoch {start_epoch} (best val AUC so far {best_auc:.4f})')

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, crit, opt, dev, epoch)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, crit, dev)
        sched.step()

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': epoch,
                    'best_auc': best_auc}, last_path)
        torch.save(model.state_dict(), out / 'final.pt')

        row = pd.DataFrame([{'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                             'val_loss': va_loss, 'val_acc': va_acc, 'val_auc': va_auc}])
        row.to_csv(hist_path, mode='a', header=not hist_path.exists(), index=False)

        print(f'epoch {epoch}/{epochs}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | '
              f'val loss {va_loss:.3f} acc {va_acc:.3f} auc {va_auc:.4f} | '
              f'{(time.time() - t0) / 60:.1f} min')

    print('Done. Best val AUC:', round(best_auc, 4))


if __name__ == '__main__':
    main()