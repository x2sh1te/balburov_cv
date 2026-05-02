import cv2
import numpy as np

# === Загрузка изображения ===
image = cv2.imread("cubes_2.png")
if image is None:
    raise FileNotFoundError("❌ Не удалось загрузить cubes_2.png. Проверь путь и наличие файла.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
height, width = image.shape[:2]

# === Настройка окон ===
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = [0, 0]
clicked = False


def on_click(event, x, y, flags, params):
    """Обработчик клика мыши по окну 'Image'"""
    global position, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Проверка: клик внутри изображения
        if 0 <= x < width and 0 <= y < height:
            print(f"✓ Clicked at ({x}, {y})")
            position = [x, y]
            clicked = True
        else:
            print(f"✗ Click outside image bounds: ({x}, {y})")


cv2.setMouseCallback("Image", on_click)

# === Инициализация накопленной маски ===
mask = np.zeros(image.shape[:2], dtype=np.uint8)


# === Функция классификации фигур ===
def classify(contour):
    """
    Анализирует контур и возвращает:
    (количество вершин, аппроксимация, название фигуры, метрика 'круглости')
    """
    verts = -1
    solidity = -1.0
    approx = []
    figure = "Unknown"

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return verts, approx, figure, solidity

    # Аппроксимация контура полигоном
    eps = 0.04 * perimeter  # 4% от периметра — хороший баланс точности
    approx = cv2.approxPolyDP(contour, eps, True)
    verts = len(approx)

    # Метрика "похожести на круг": отношение площади к площади описанной окружности
    _, radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * (radius ** 2) if radius > 0 else 1
    circularity = area / circle_area if circle_area > 0 else 0

    # Логика классификации (порядок важен!)
    if circularity > 0.75:  # Круг/сфера — самый "плотный" контур
        figure = "Sphere"
    elif verts == 3:
        figure = "Triangle"
    elif verts == 4:
        figure = "Cube/Rect"
    elif verts == 5:
        figure = "Pentagon"
    elif verts == 6:
        figure = "Hexagon"
    else:
        figure = f"Poly{verts}"

    return verts, approx, figure, circularity


# === Настройки отображения текста ===
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

# === Главный цикл ===
while True:
    display_image = image.copy()
    key = cv2.waitKey(50) & 0xFF  # Маска для кроссплатформенной совместимости

    # Выход по 'q'
    if key == ord("q"):
        break

    # Очистка маски по 'c'
    if key == ord("c"):
        mask[:] = 0
        print("✓ Mask cleared")

    # === Обработка клика: выделение цвета ===
    if clicked:
        clicked = False
        x, y = position

        # Получаем цвет в точке клика (помни: [y, x] для numpy!)
        color = hsv[y, x]

        # Диапазон ±10% с защитой от выхода за границы [0, 255]
        lower = np.clip(color * 0.9, 0, 255).astype(np.uint8)
        upper = np.clip(color * 1.1, 0, 255).astype(np.uint8)

        # Создаём маску по диапазону
        inr = cv2.inRange(hsv, lower, upper)

        # Морфологическое "закрытие": убираем шум и дыры
        kernel = np.ones((5, 5), np.uint8)
        inr = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, kernel)

        # Накладываем на основную маску (накопление)
        mask = cv2.bitwise_or(mask, inr)

        # Показываем накопленную маску
        cv2.imshow("Mask", mask)

    # === Поиск и обработка контуров ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем пустые контуры
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # порог 50 пикселей

    for i, cnt in enumerate(contours):
        # 1. Классифицируем фигуру
        verts, approx, figure, circularity = classify(cnt)

        # 2. Считаем площадь и центр
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        # 3. Находим верхнюю точку для привязки текста
        top_point = tuple(cnt[cnt[:, :, 1].argmin()][0])

        # 4. Формируем текст: тип фигуры + круглость в %
        text = f"{figure} ({circularity * 100:.0f}%)"

        # 5. Вычисляем размер текста
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # 6. Позиция: над объектом, центрировано
        text_x = max(5, min(width - text_w - 5, top_point[0] - text_w // 2))
        text_y = max(text_h + 5, top_point[1] - 10)

        # 7. Рисуем фон под текст (тёмный полупрозрачный прямоугольник)
        cv2.rectangle(
            display_image,
            (text_x, text_y - text_h - 4),
            (text_x + text_w, text_y + 4),
            (30, 30, 30),  # почти чёрный
            -1  # заливка
        )

        # 8. Рисуем текст
        cv2.putText(
            display_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # белый
            thickness,
            cv2.LINE_AA
        )

        # 9. Рисуем контур и аппроксимацию
        cv2.drawContours(display_image, [cnt], -1, (0, 255, 0), 2)  # зелёный — основной контур
        if len(approx) > 0:
            cv2.drawContours(display_image, [approx], -1, (255, 0, 255), 1)  # малиновый — аппроксимация

        # 10. Рисуем точку центра
        cv2.circle(display_image, (cx, cy), 3, (255, 255, 0), -1)  # циановая точка

    # Показываем результат
    cv2.imshow("Image", display_image)

# === Завершение ===
cv2.destroyAllWindows()
print("✓ Program finished")