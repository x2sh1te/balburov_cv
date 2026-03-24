import numpy as np
import matplotlib.pyplot as plt


img = np.zeros((16, 16), dtype=int)
# Квадрат
img[2:6, 10:14] = 1
img[6:14, 2:5] = 1
img[8:12, 6:14] = 1

plt.imshow(img, cmap='gray')
plt.show()