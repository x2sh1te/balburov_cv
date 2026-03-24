import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


def main():
    img = np.load('stars.npy')
    binary = img > img.max() * 0.5

    # Шаблоны 5x5
    plus = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    cross = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])


    plus_sum = np.sum(plus)
    cross_sum = np.sum(cross)

    plus_corr = correlate2d(binary, plus, mode='same')
    cross_corr = correlate2d(binary, cross, mode='same')

    plus_mask = plus_corr == plus_sum
    cross_mask = cross_corr == cross_sum
    cross_mask = cross_mask & ~plus_mask  # Убираем перекрытия

    plus_count = np.sum(plus_mask)
    cross_count = np.sum(cross_mask)

    # Визуализация
    vis = np.dstack([img] * 3) / img.max()
    vis[plus_mask] = [0, 1, 0]
    vis[cross_mask] = [1, 0, 0]

    print(f"Плюсов: {plus_count}, Крестов: {cross_count}, Всего: {plus_count + cross_count}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(vis)
    plt.show()


if __name__ == '__main__':
    main()
