import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import os


def solve_trajectories():
    # Словарь для хранения путей объектов: {id: [[x1, x2...], [y1, y2...]]}
    trajectories = {}
    # Храним последние координаты объектов для сопоставления
    last_points = {}
    next_id = 0

    threshold = 77  # Порог из твоего примера

    for i in range(100):
        file_path = f"{i}.npy"
        if not os.path.exists(file_path):
            continue

        # 1. Загрузка и бинаризация (как в твоем коде)
        image = np.load(file_path)
        binary = image > threshold

        # 2. Поиск объектов (связных областей)
        labeled = label(binary)
        regions = regionprops(labeled)

        current_centers = []
        for region in regions:
            # centroid возвращает (row, col) -> (y, x)
            y, x = region.centroid
            current_centers.append((x, y))

        # 3. Логика трекинга (сопоставление с прошлым кадром)
        if i == 0 or not last_points:
            # На первом кадре просто регистрируем все объекты
            for pt in current_centers:
                trajectories[next_id] = [[pt[0]], [pt[1]]]
                last_points[next_id] = pt
                next_id += 1
        else:
            new_last_points = {}
            used_current_indices = set()

            # Пытаемся найти продолжение для каждой уже известной траектории
            for obj_id, last_pt in last_points.items():
                distances = []
                for idx, curr_pt in enumerate(current_centers):
                    if idx in used_current_indices:
                        distances.append(float('inf'))
                        continue
                    # Считаем Евклидово расстояние
                    dist = np.sqrt((last_pt[0] - curr_pt[0]) ** 2 + (last_pt[1] - curr_pt[1]) ** 2)
                    distances.append(dist)

                if distances and min(distances) < 50:  # 50 — макс. смещение объекта
                    best_idx = np.argmin(distances)
                    best_pt = current_centers[best_idx]

                    trajectories[obj_id][0].append(best_pt[0])
                    trajectories[obj_id][1].append(best_pt[1])
                    new_last_points[obj_id] = best_pt
                    used_current_indices.add(best_idx)

            # Если появились новые объекты, которых не было раньше
            for idx, pt in enumerate(current_centers):
                if idx not in used_current_indices:
                    trajectories[next_id] = [[pt[0]], [pt[1]]]
                    new_last_points[next_id] = pt
                    next_id += 1

            last_points = new_last_points

    # 4. Визуализация (Линейный график)
    plt.figure(figsize=(10, 7))
    for obj_id, coords in trajectories.items():
        if len(coords[0]) > 2:  # Рисуем только те, что двигались
            plt.plot(coords[0], coords[1], '-o', markersize=3, label=f"Объект {obj_id}")

    plt.title("Траектории движения объектов")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Чтобы (0,0) был вверху, как в массиве
    plt.grid(True, ls=':')
    if len(trajectories) < 15:  # Показываем легенду, если объектов мало
        plt.legend()

    # Важно: это заставляет окно появиться
    plt.show()


# Обязательно вызываем функцию!
if __name__ == "__main__":
    solve_trajectories()