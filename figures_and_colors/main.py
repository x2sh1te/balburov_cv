import cv2
import numpy as np
from skimage.io import imread
from collections import Counter
import os


def get_color_name(hue):
    if (hue <= 10) or (hue >= 160):
        return "Красный"
    elif 11 <= hue <= 25:
        return "Оранжевый"
    elif 26 <= hue <= 35:
        return "Желтый"
    elif 36 <= hue <= 80:
        return "Зеленый"
    elif 81 <= hue <= 130:
        return "Синий/Голубой"
    elif 131 <= hue <= 155:
        return "Фиолетовый/Пурпурный"
    else:
        return "Неопределенный"


def process_colored_shapes(image_path):
    # Чтение изображения через skimage
    img_rgb = imread(image_path)

    # Конвертация из float [0,1] в uint8 [0,255]
    img = (img_rgb * 255).astype(np.uint8)

    # Конвертация в HSV (skimage читает в RGB формате)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects_colors = []
    circles_colors = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        mean_hsv = cv2.mean(hsv, mask=mask)
        hue = mean_hsv[0]
        color_name = get_color_name(hue)

        if len(approx) == 4:
            rects_colors.append(color_name)
        elif len(approx) > 5:
            circles_colors.append(color_name)

    total = len(rects_colors) + len(circles_colors)
    rect_stats = Counter(rects_colors)
    circle_stats = Counter(circles_colors)

    print("=" * 30)
    print(f"ИТОГО ФИГУР: {total}")
    print("=" * 30)

    print("\nПРЯМОУГОЛЬНИКИ:")
    for color, count in rect_stats.items():
        print(f"  - {color}: {count}")

    print("\nКРУГИ:")
    for color, count in circle_stats.items():
        print(f"  - {color}: {count}")


# Получаем путь к папке со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__))
# Создаем полный путь к изображению
path = os.path.join(script_dir, 'balls_and_rects.png')

# Проверяем существует ли файл
if os.path.exists(path):
    print(f"Загрузка файла: {path}")
    process_colored_shapes(path)
else:
    print(f"Ошибка: Файл не найден по пути: {path}")
    print("Убедитесь, что balls_and_rects.png находится в той же папке что и main.py")