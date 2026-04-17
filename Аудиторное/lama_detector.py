import cv2
import numpy as np
import matplotlib.pyplot as plt


def count_and_sort_balls(C:\Users\nikit\OneDrive\Desktop\compvision\Аудиторное\balls.png):
    # 1. Чтение изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    # Преобразуем в HSV (Hue, Saturation, Value) - это удобнее для работы с цветом
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Создание маски для отделения от черного фона
    # Мы берем все пиксели, у которых насыщенность (S) и яркость (V) больше 50.
    # Черный фон имеет низкие значения S и V.
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 3. Поиск контуров (шариков)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_balls = len(contours)
    print(f"Общее количество шариков: {total_balls}")

    # Словарь для подсчета цветов
    # Диапазоны Hue (оттенок) в OpenCV: 0-180
    color_counts = {
        'Красный': 0,
        'Оранжевый': 0,
        'Желтый': 0,
        'Зеленый': 0,
        'Голубой (Cyan)': 0,
        'Синий': 0,
        'Фиолетовый': 0,
        'Розовый': 0,
        'Неопределенный': 0
    }

    # Копия изображения для отрисовки результатов
    img_result = img.copy()

    # 4. Перебор каждого найденного шарика
    for i, cnt in enumerate(contours):
        # Вычисляем центр масс контура
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Берем цвет пикселя в центре шарика
            # Используем [cY, cX], так как numpy использует [строка, столбец]
            h, s, v = hsv[cY, cX]

            # 5. Классификация цвета по Hue (H)
            # Примечание: Красный находится на стыке (0 и 180)
            if (0 <= h < 12) or (160 <= h <= 180):
                color_counts['Красный'] += 1
                color_bgr = (0, 0, 255)  # BGR для отрисовки
            elif 12 <= h < 22:
                color_counts['Оранжевый'] += 1
                color_bgr = (0, 165, 255)
            elif 22 <= h < 33:
                color_counts['Желтый'] += 1
                color_bgr = (0, 255, 255)
            elif 33 <= h < 70:
                color_counts['Зеленый'] += 1
                color_bgr = (0, 255, 0)
            elif 70 <= h < 100:
                color_counts['Голубой (Cyan)'] += 1
                color_bgr = (255, 255, 0)
            elif 100 <= h < 125:
                color_counts['Синий'] += 1
                color_bgr = (255, 0, 0)
            elif 125 <= h < 145:
                color_counts['Фиолетовый'] += 1
                color_bgr = (255, 0, 255)
            elif 145 <= h < 160:
                color_counts['Розовый'] += 1
                color_bgr = (192, 192, 255)  # Light Pink-ish in BGR
            else:
                color_counts['Неопределенный'] += 1
                color_bgr = (255, 255, 255)

            # Рисуем кружок вокруг шарика и номер
            cv2.circle(img_result, (cX, cY), 7, color_bgr, -1)
            # Если нужно видеть номера (раскомментируйте строку ниже, но их будет много)
            # cv2.putText(img_result, str(i), (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 6. Вывод статистики
    print("\n--- Сортировка по цветам ---")
    for color, count in color_counts.items():
        if count > 0:
            print(f"{color}: {count}")

    # Показываем результат
    # Преобразуем BGR в RGB для отображения через matplotlib
    img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 15))
    plt.imshow(img_rgb)
    plt.title(f"Total: {total_balls}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Запуск функции (замените 'image.jpg' на имя вашего файла)
count_and_sort_balls('image.jpg')