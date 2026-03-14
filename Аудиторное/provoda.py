import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

struct = np.ones((3, 1))

images = []
processes = []

for i in range(1, 7):
    image = np.load(f"wires{i}.npy")
    process = binary_opening(image, struct)
    images.append(image)
    processes.append(process)


for i in range(6):
    labeled_image = label(images[i])
    labeled_process = label(processes[i])

    print(f"File wires{i + 1}.npy:")
    print(f"Original {np.max(labeled_image)}")
    print(f"Processed {np.max(labeled_process)}")
    print()

plt.subplot(121)
plt.imshow(labeled_image == 2)
plt.subplot(122)
plt.imshow(processes[-1])
plt.show()