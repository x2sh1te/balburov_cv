import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops, label
from pathlib import Path

def get_holes(prop):
    img = prop.image
    h, w = img.shape
    padded = np.zeros((h + 2, w + 2))
    padded[1:-1, 1:-1] = img
    inv = np.logical_not(padded)
    return np.max(label(inv)) - 1

def check_symm(prop):
    img = prop.image.astype(float)
    h, w = img.shape
    if w < 6: return 0.0
    mid = w // 2
    l_part = img[:, :mid]
    r_part = np.fliplr(img[:, w - mid:])
    diff = np.mean(np.abs(l_part - r_part))
    return 1.0 - diff

def get_features(prop):
    h, w = prop.image.shape
    cy, cx = prop.centroid_local
    area_ratio = prop.area / prop.image.size
    y_rel = cy / h
    x_rel = cx / w
    perim_ratio = prop.perimeter / prop.image.size
    hl = get_holes(prop)
    v_l = (np.sum(prop.image, axis=0) == h).sum() / w
    h_l = (np.sum(prop.image, axis=1) == w).sum() / h
    ecc = prop.eccentricity
    asp = min(h, w) / max(h, w)
    symm = check_symm(prop)
    return np.array([area_ratio, x_rel, y_rel, perim_ratio, hl, v_l, h_l, ecc, asp, symm])

def recognize(prop, refs):
    vec = get_features(prop)
    best_sym = ""
    min_dist = float('inf')
    for sym, ref_vec in refs.items():
        dist = np.sqrt(np.sum((ref_vec - vec) ** 2))
        if dist < min_dist:
            min_dist = dist
            best_sym = sym
    return best_sym

def main():
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)

    ref_img = imread("alphabet-small.png")[
              :, :, :-1]
    ref_gray = ref_img.sum(axis=2)
    ref_bin = ref_gray != 765.0
    ref_props = regionprops(label(ref_bin))

    symbols = ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]
    reference_data = {sym: get_features(p) for p, sym in zip(ref_props, symbols)}

    target_img = imread("alphabet.png")[:,
                 :, :-1]
    target_bin = target_img.mean(axis=2) > 0
    target_props = regionprops(label(target_bin))

    counts = {}
    fig = plt.figure(figsize=(5, 7))

    for p in target_props:
        sym = recognize(p, reference_data)
        counts[sym] = counts.get(sym, 0) + 1
        plt.clf()
        plt.title(f"Class - '{sym}'")
        plt.imshow(p.image)
        plt.savefig(out_dir / f"image_{p.label}.png")
    print(counts)

if __name__ == "__main__":
    main()