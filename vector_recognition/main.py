import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

BASE_PATH = Path(r"C:\Users\nikit\OneDrive\Desktop\compvision\vector_recognition")
image_path = BASE_PATH / "out_tree"
image_path.mkdir(exist_ok=True)


def count_holes(region):
# сортировка символов по "дыркам"
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    labeled = label(np.logical_not(new_image))
    return max(0, np.max(labeled) - 1)


def is_symmetric_lr(img, threshold=0.72):
# проверяем "8" и "B" путем проверки на симметрию
    h, w = img.shape
    if w < 4: return False
    mid = w // 2
    left = img[:, :mid]
    right = img[:, w - mid:][:, ::-1]  
    min_w = min(left.shape[1], right.shape[1])
    similarity = np.mean(left[:, :min_w] == right[:, :min_w])
    return similarity > threshold  # ← если похоже → "8"


def narrows_at_bottom(img, ratio_thresh=0.75):
#   различаем "0" и "А". "0" сужается к низу, а "А" нет
    h, w = img.shape
    if h < 6: return False
    middle = img[h // 3: 2 * h // 3, :]
    bottom = img[2 * h // 3:, :]

    mid_width = np.sum(np.any(middle, axis=0))
    bot_width = np.sum(np.any(bottom, axis=0))
    if mid_width == 0: return False
    return (bot_width / mid_width) < ratio_thresh  # ← если дно уже → "0"


def classificator(region):
    holes = count_holes(region)
    img = region.image.astype(bool)
#"8" (симметричный) vs "B" (несимметричный)
    if holes == 2:
        if is_symmetric_lr(img):
            return "8"
        else:
            return "B"

# ️"0" (сужается книзу) vs "A" (широкий снизу)
    elif holes == 1:
        if narrows_at_bottom(img):
            return "0"  # ← сужается → "0"
        else:
            return "A"  # ← широкий низ → "A"

# остальные символы
    else:
        h, w = img.shape
        if img.size > 0 and np.all(img):
            return "-"
        if np.min(img.shape) / np.max(img.shape) > 0.9:
            return "*"
        vlines = (np.sum(img, axis=0) == h).sum()
        hlines = (np.sum(img, axis=1) == w).sum()
        if vlines > 0 and hlines > 0:
            return "1"
        bays = sum(r.area > 3 for r in regionprops(label(np.logical_not(img))))
        if bays == 2:
            return "/"
        elif bays == 4:
            return "X"
        elif bays == 5:
            return "W"

    return "?"

image = imread(BASE_PATH / "alphabet.png")
if image.ndim == 3 and image.shape[2] == 4:
    image = image[:, :, :-1]

abinary = image.mean(2) > 0
aprops = regionprops(label(abinary))
print(f"Символов найдено: {len(aprops)}")

result = {}
plt.figure(figsize=(4, 5))

for region in aprops:
    symbol = classificator(region)  
    result[symbol] = result.get(symbol, 0) + 1
    plt.cla()
    plt.title(f"'{symbol}'")  # ← подпись класса
    plt.imshow(region.image, cmap='gray')
    plt.axis('off')
    plt.savefig(image_path / f"image_{region.label}.png", bbox_inches='tight')

print("\n Результат:", result)
unknown = result.get("?", 0)
acc = 1 - unknown / len(aprops) if aprops else 0
print(f" Точность: {acc:.2%}")
