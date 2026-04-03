import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
import glob
import os

# Параметры
folder = "C:\Users\nikit\OneDrive\Desktop\compvision\trajectory"  # Укажите путь к папке с файлами
threshold = 0.5  # Порог для бинаризации (настройте под ваши данные)
max_objects = 10  # Максимальное число объектов для отслеживания

# Словарь для хранения траекторий: {object_id: [(x1,y1), (x2,y2), ...]}
trajectories = {}

# Получаем отсортированный список файлов
files = sorted(glob.glob(os.path.join(folder, "h_*.npy")),
               key=lambda x: int(x.split('_')[-1].split('.')[0]))

for frame_idx, filepath in enumerate(files):
    data = np.load(filepath)

    # Бинаризация и поиск объектов
    binary = data > threshold
    labeled, num = label(binary)

    # Получаем центры масс объектов
    centers = center_of_mass(binary, labeled, range(1, num + 1))

    # Простая ассоциация объектов по ближайшему соседу (для демо)
    # В реальном проекте используйте Hungarian algorithm или SORT/DeepSORT
    current_positions = [(int(c[1]), int(c[0])) for c in centers]  # (x, y)

    # Здесь должна быть логика трекинга (ID assignment)
    # Для простоты: считаем, что объекты не пересекаются и идут по порядку
    for obj_id, (x, y) in enumerate(current_positions[:max_objects]):
        if obj_id not in trajectories:
            trajectories[obj_id] = []
        trajectories[obj_id].append((frame_idx, x, y))

# Визуализация траекторий
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, max_objects))

for obj_id, traj in trajectories.items():
    if len(traj) > 1:
        frames, xs, ys = zip(*traj)
        plt.plot(xs, ys, '-o', label=f'Object {obj_id}',
                 color=colors[obj_id % len(colors)], markersize=3)

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Trajectories of Objects (frames h_0 to h_99)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()