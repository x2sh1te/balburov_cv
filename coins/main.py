import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path

BASE_PATH = Path(__file__).parent
coins = np.load(BASE_PATH / "coins.npy")

labeled = label(coins)
props = regionprops(labeled)

# измеряем площадь каждой монеты
areas = [p.area for p in props]

# кластеризация по размеру: 4 номинала → 4 группы
unique_areas = sorted(set(areas))
if len(unique_areas) > 4:
    # Если много вариаций — группируем через квантование
    unique_areas = sorted(np.quantile(areas, [0.1, 0.35, 0.65, 0.95]))

# сопоставление: наименьшая → 1, далее → 2, 5, 10
size_to_value = {s: v for s, v in zip(sorted(unique_areas)[:4], [1, 2, 5, 10])}

# подсчёт суммы
total = 0
for p in props:
    closest = min(size_to_value.keys(), key=lambda s: abs(s - p.area))
    value = size_to_value[closest]
    total += value

print(f"💰 Общая сумма: {total}")
print(f"🪙 Найдено монет: {len(props)}")