import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening

struct = np.ones((3, 1))

images = []
processes = []

for i in range(1, 7):
    image = np.load(f"wires{i}.npy")
    labeled_image = label(image)
    for wires_id in range (1, np.max(labeled_image) + 1 ):
        wire = labeled_image == wires_id
        parts = opening(wire, struct)
        labeled_parts = label(parts)
        parts_count = np.max(labeled_parts)
        print("")
        print(f"Картинка {i}:")
        print(f"Провод {wires_id} имеет {parts_count} частей")

# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.imshow(process)
# plt.show()

