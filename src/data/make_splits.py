import numpy as np, pandas as pd
from pathlib import Path
from src.utils.common import load_config, seed_everything


def main():
    cfg = load_config()
    seed_everything(cfg['seed'])
    root = Path(cfg['paths']['data_root'])
    out_dir = Path(cfg['paths']['processed'])
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(root / 'metadata.csv')

    # Union-find: pairs that share any face image end up in one group
    parent = list(range(len(d)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen = {}
    for i, row in enumerate(d.itertuples()):
        for face in (row.source_identity, row.target_face):
            if face in seen:
                union(i, seen[face])
            else:
                seen[face] = i
    d['group'] = [find(i) for i in range(len(d))]

    # Split whole groups 70/15/15
    groups = d['group'].unique()
    np.random.default_rng(cfg['seed']).shuffle(groups)
    n = len(groups)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    split_of = {}
    for j, g in enumerate(groups):
        split_of[g] = 'train' if j < n_train else ('val' if j < n_train + n_val else 'test')
    d['split'] = d['group'].map(split_of)

    # One row per image (label: 1 = fake, 0 = real)
    rows = []
    for r in d.itertuples():
        rows.append((r.fake_file, 1, r.id, r.group, r.split))
        rows.append((r.real_file, 0, r.id, r.group, r.split))
    out = pd.DataFrame(rows, columns=['path', 'label', 'pair_id', 'group', 'split'])
    out.to_csv(out_dir / 'splits.csv', index=False)

    print(out.groupby(['split', 'label']).size().unstack())
    print('groups:', n, '| largest group:', d['group'].value_counts().max())

    # Leak check: every face should live in exactly one split
    faces = pd.concat([
        d[['source_identity', 'split']].rename(columns={'source_identity': 'face'}),
        d[['target_face', 'split']].rename(columns={'target_face': 'face'}),
    ])
    print('max splits per face (should be 1):', faces.groupby('face')['split'].nunique().max())


if __name__ == '__main__':
    main()