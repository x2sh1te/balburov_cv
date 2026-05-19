import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path
import collections

def get_holes(prop):
    h, w = prop.image.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = prop.image
    return np.max(label(~padded)) - 1

def calc_sym(prop):
    img = prop.image.astype(float)
    h, w = img.shape
    if w < 6:
        return 0.0
    return 1.0 - np.mean(np.abs(img[:, :w//2] - np.fliplr(img[:, w - w//2:])))

def build_vector(prop):
    img = prop.image
    h, w = img.shape
    cy, cx = prop.centroid_local
    v_lines = np.sum(img.sum(axis=0) == h) / w
    h_lines = np.sum(img.sum(axis=1) == w) / h
    return np.array([
        prop.area / img.size, cx / w, cy / h, prop.perimeter / img.size,
        get_holes(prop), v_lines, h_lines, prop.eccentricity,
        min(h, w) / max(h, w), calc_sym(prop)
    ])

def find_match(prop, refs):
    vec = build_vector(prop)
    return min(refs.keys(), key=lambda k: np.linalg.norm(refs[k] - vec))

out_p = Path(__file__).parent / "out"
out_p.mkdir(exist_ok=True)

ref_img = imread("alphabet_ext.png")[:, :, :-1].sum(axis=2)
ref_regions = regionprops(label(ref_img < 500))

labels = ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-", "P", "D"]
features_db = {sym: build_vector(r) for r, sym in zip(ref_regions, labels)}

tgt_img = imread("symbols.png")[:, :, :-1].mean(axis=2) > 0
tgt_labeled = label(tgt_img)
print(np.max(tgt_labeled))

tgt_regions = regionprops(tgt_labeled)
stats = collections.Counter()

plt.figure(figsize=(5, 7))

for r in tgt_regions:
    match = find_match(r, features_db)
    stats[match] += 1
    plt.clf()
    plt.title(f"Class - '{match}'")
    plt.imshow(r.image)
    plt.savefig(out_p / f"image_{r.label}.png")

for k, v in stats.most_common():
    print(f"'{k}': {v}")