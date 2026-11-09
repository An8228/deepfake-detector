import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.utils.common import load_config, get_device
from src.data.dataset import build_transforms
from src.models.model import build_model
from src.explain.gradcam import GradCAM

st.set_page_config(page_title='Deepfake Detector', layout='wide')
cfg = load_config()
OUT = Path(cfg['paths']['outputs'])


@st.cache_resource
def load_model():
    dev = get_device()
    model = build_model(cfg, pretrained=False).to(dev)
    model.load_state_dict(torch.load(OUT / 'final.pt', map_location=dev))
    model.eval()
    return model, GradCAM(model), dev


def predict(img):
    model, gc, dev = load_model()
    tf = build_transforms(cfg['img_size'], train=False)
    x = tf(img).unsqueeze(0).to(dev)
    prob, maps = gc(x)
    return float(prob[0]), maps[0]


def overlay(img, m):
    base = np.array(img.resize((224, 224)))
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (0.55 * base + 0.45 * heat).astype(np.uint8)


LIMITS = ('Limits: trained on face crops made with a single face-swap tool '
          '(inswapper_128). A high score here does not mean it detects fakes '
          'from other methods. It is still weakest on blurry images.')

st.title('Explainable Deepfake Detector')
tab1, tab2 = st.tabs(['Detect', 'Results'])

with tab1:
    st.write('Upload a close-up face image (like a passport-style crop).')
    up = st.file_uploader('Image', type=['jpg', 'jpeg', 'png'])
    if up:
        img = Image.open(up).convert('RGB')
        prob, m = predict(img)
        fake = prob > 0.5
        c1, c2 = st.columns(2)
        c1.image(img.resize((224, 224)), caption='Input')
        if fake:
            c2.image(overlay(img, m), caption='Grad-CAM: regions pushing toward "fake"')
            st.error(f'Likely FAKE ({prob:.1%} confidence)')
        else:
            c2.info('No heatmap shown: nothing in this image pushed the model strongly toward "fake".')
            st.success(f'Likely REAL ({1 - prob:.1%} confidence)')
    st.caption(LIMITS)

with tab2:
    hist = OUT / 'history.csv'
    if hist.exists():
        h = pd.read_csv(hist).set_index('epoch')
        st.subheader('Training curves')
        st.line_chart(h[['train_loss', 'val_loss']])
        st.line_chart(h[['train_acc', 'val_acc']])
    res = OUT / 'test_results.json'
    if res.exists():
        st.subheader('Test set, clean and degraded images')
        r = json.load(open(res))
        st.table(pd.DataFrame([{'condition': k, 'accuracy': v['acc'], 'AUC': v['auc']}
                               for k, v in r.items()]))
    png = OUT / 'gradcam.png'
    if png.exists():
        st.subheader('Grad-CAM examples')
        st.image(str(png), caption='Rows: fake, fake heatmap, real, real heatmap')
    st.caption(LIMITS)