import io
import random
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from src.utils.common import load_config

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class RandomDegrade:
    """With probability p, apply one random degradation (same for real and fake)."""
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        op = random.choice(['blur', 'down', 'jpeg'])
        if op == 'blur':
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
        if op == 'down':
            s = random.choice([96, 128, 160, 192])
            return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=random.randint(30, 90))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


def build_transforms(img_size, train):
    ops = []
    if train:
        ops.append(RandomDegrade(0.7))
    ops.append(T.Resize((img_size, img_size)))
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class FaceDataset(Dataset):
    def __init__(self, split, cfg):
        df = pd.read_csv(Path(cfg['paths']['processed']) / 'splits.csv')
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.root = Path(cfg['paths']['data_root'])
        self.tf = build_transforms(cfg['img_size'], train=(split == 'train'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row['path']).convert('RGB')
        return self.tf(img), torch.tensor(row['label'], dtype=torch.float32)


def make_loader(split, cfg):
    ds = FaceDataset(split, cfg)
    return DataLoader(ds, batch_size=cfg['batch_size'],
                      shuffle=(split == 'train'),
                      num_workers=cfg['num_workers'],
                      pin_memory=torch.cuda.is_available())


if __name__ == '__main__':
    cfg = load_config()
    for s in ('train', 'val', 'test'):
        x, y = next(iter(make_loader(s, cfg)))
        print(s, len(FaceDataset(s, cfg)), tuple(x.shape), y[:8].tolist())