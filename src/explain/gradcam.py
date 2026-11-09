import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from src.utils.common import load_config, seed_everything, get_device
from src.data.dataset import build_transforms
from src.models.model import build_model


class GradCAM:
    """Heatmap of the regions that push the model toward 'fake'."""
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        model.features[-1].register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, 'grads', g))

    def __call__(self, x):
        self.model.zero_grad()
        logit = self.model(x).squeeze(1)
        logit.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).detach().cpu().numpy()
        maps = []
        for c in cam:
            c = c - c.min()
            c = c / (c.max() + 1e-8)
            maps.append(cv2.resize(c, (224, 224)))
        return torch.sigmoid(logit).detach().cpu().numpy(), np.stack(maps)


def load_batch(paths, root, tf):
    return torch.stack([tf(Image.open(root / p).convert('RGB')) for p in paths])


def main():
    cfg = load_config()
    seed_everything(cfg['seed'])
    dev = get_device()
    out = Path(cfg['paths']['outputs'])
    root = Path(cfg['paths']['data_root'])

    model = build_model(cfg, pretrained=False).to(dev)
    model.load_state_dict(torch.load(out / 'final.pt', map_location=dev))
    model.eval()
    gc = GradCAM(model)
    tf = build_transforms(cfg['img_size'], train=False)

    df = pd.read_csv(Path(cfg['paths']['processed']) / 'splits.csv')
    test = df[df['split'] == 'test']

    inner = np.zeros((224, 224), bool)
    inner[28:196, 28:196] = True  # central 75% x 75%

    n = 100
    shown = {}
    for label, name in ((1, 'fake'), (0, 'real')):
        paths = test[test['label'] == label].sample(n, random_state=cfg['seed']).path.tolist()
        fracs, probs = [], []
        for i in range(0, n, 32):
            x = load_batch(paths[i:i + 32], root, tf).to(dev)
            p, maps = gc(x)
            probs += p.tolist()
            fracs += [float(m[~inner].sum() / (m.sum() + 1e-8)) for m in maps]
            if i == 0:
                shown[name] = (paths[:4], maps[:4])
        print(f'{name}: mean P(fake) {np.mean(probs):.3f} | '
              f'share of heatmap in outer border {np.mean(fracs):.3f}')
    print('(uniform attention would put about 0.44 in the border)')

    rows = []
    for name in ('fake', 'real'):
        ps, maps = shown[name]
        originals, overlays = [], []
        for pth, m in zip(ps, maps):
            img = np.array(Image.open(root / pth).convert('RGB').resize((224, 224)))
            heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            overlays.append((0.55 * img + 0.45 * heat).astype(np.uint8))
            originals.append(img)
        rows += [np.hstack(originals), np.hstack(overlays)]
    Image.fromarray(np.vstack(rows)).save(out / 'gradcam.png')
    print('Saved', out / 'gradcam.png')


if __name__ == '__main__':
    main()