import numpy as np
from scipy import ndimage

img = np.load('coins.npy')

labeled, n = ndimage.label(img)

radii = []
for i in range(1, n + 1):
    area = np.sum(labeled == i)
    radii.append(np.sqrt(area / np.pi))
radii = np.array(radii)

q = np.percentile(radii, [25, 50, 75])
nominals = [1, 2, 5, 10]

def get_nominal(r):
    if   r < q[0]: return nominals[0]
    elif r < q[1]: return nominals[1]
    elif r < q[2]: return nominals[2]
    else:          return nominals[3]

coins = [get_nominal(r) for r in radii]

print(f"Монет найдено: {n}")
print(f"Номиналы: {sorted(coins)}")
print(f"Сумма: {sum(coins)} руб.")