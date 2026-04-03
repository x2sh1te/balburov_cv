import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_data(f):
    image = np.load(f'h_{f}.npy')
    label, n = ndimage.label(image)
    return np.array(ndimage.center_of_mass(image, label, range(1, n + 1)))

centers = get_data(0)
trajectories = [[c] for c in centers[np.argsort(centers[:, 1])]]
num_frames = 100

for f in range(1, num_frames):
    curr_centers = get_data(f)
    for t in trajectories:
        prev = t[-1]
        index = np.argmin(np.sum((curr_centers - prev)**2, axis=1))
        t.append(curr_centers[index])

plt.figure(figsize=(10, 8))
for i, t in enumerate(trajectories):
    t = np.array(t)
    plt.plot(t[:, 1], t[:, 0], label=f'Объект {i+1}')
    plt.scatter(t[0, 1], t[0, 0], s=100)
    plt.scatter(t[-1, 1], t[-1, 0], s=100)

plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()