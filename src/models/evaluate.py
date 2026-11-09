import argparse
import io
import json
import torch
from pathlib import Path
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.utils.common import load_config, seed_everything, get_device
from src.data.dataset import FaceDataset
from src.models.model import build_model


def jpeg(img, q=50):
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def downscale(img, s=128):
    return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)


def blur(img, r=1.5):
    return img.filter(ImageFilter.GaussianBlur(r))


PERTURBATIONS = {'clean': None, 'jpeg_q50': jpeg,
                 'downscale_128': downscale, 'blur_1.5': blur}


class PerturbedDataset(FaceDataset):
    def __init__(self, split, cfg, fn):
        super().__init__(split, cfg)
        self.fn = fn

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row['path']).convert('RGB')
        if self.fn:
            img = self.fn(img)
        return self.tf(img), torch.tensor(row['label'], dtype=torch.float32)


@torch.no_grad()
def run(model, ds, cfg, dev):
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'])
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu())
        labels.append(y)
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    pred = (p > 0.5).astype(int)
    return {'acc': round(float((pred == y).mean()), 4),
            'auc': round(float(roc_auc_score(y, p)), 4),
            'confusion [[TN, FP], [FN, TP]]': confusion_matrix(y, pred).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test')
    parser.add_argument('--ckpt', default='best.pt')
    args = parser.parse_args()

    cfg = load_config()
    seed_everything(cfg['seed'])
    dev = get_device()
    out = Path(cfg['paths']['outputs'])

    model = build_model(cfg, pretrained=False).to(dev)
    model.load_state_dict(torch.load(out / args.ckpt, map_location=dev))
    model.eval()

    results = {}
    for name, fn in PERTURBATIONS.items():
        results[name] = run(model, PerturbedDataset(args.split, cfg, fn), cfg, dev)
        print(name, results[name])

    with open(out / f'{args.split}_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()