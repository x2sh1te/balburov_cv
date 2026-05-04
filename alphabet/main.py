import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
import collections

def extractor(region):

    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum() / region.image.shape[1]
    hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum() / region.image.shape[0]
    eccentricity = region.eccentricity
    h, w = region.image.shape
    aspect = min(h, w) / max(h, w)
    return np.array([region.area / region.image.size, cx, cy, perimeter,
                     vlines, hlines, eccentricity, aspect])

def classificator(region, templates):

    features = extractor(region)
    min_d = float('inf')
    result = ""
    for symbol, t in templates.items():
        d = ((t - features) ** 2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("alphabet_ext.png")[:, :, :-1].sum(2)
binary = template < 500
labeled = label(binary)
props = regionprops(labeled)

templates = {}
for region, symbol in zip(props, ["8", "O", "A", "B", "1", "W",
                                   "X", "*", "/", "-", "P", "D"]):
    templates[symbol] = extractor(region)

image = imread("symbols.png")[:, :, :-1].mean(2)
abinary = image > 0
alabeled = label(abinary)
aprops = regionprops(alabeled)

result = collections.Counter()
for region in aprops:
    symbol = classificator(region, templates)
    result[symbol] += 1

for symbol, count in result.most_common():
    print(f"'{symbol}': {count}")