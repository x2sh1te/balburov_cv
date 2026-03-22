import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage



def solve_trajectories():
    trajectories = {}
    for i in range(100):
        file_path = f'h_{i}.npy'
        img = np.load(file_path)
        labeled_array, num_features = ndimage.label(img > 0)
        centers = ndimage.center_of_mass(img, labeled_array, range(1, num_features + 1))

        for obj_id, center in enumerate(centers):
            y, x = center
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append((x, y))

    plt.figure(figsize=(12, 10))

    for obj_id, coords in trajectories.items():
        if len(coords) > 1:
            x_vals, y_vals = zip(*coords)

            plt.plot(x_vals, y_vals, marker='o', markersize=3, linewidth=1)

    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    solve_trajectories()