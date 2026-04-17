import cv2
import numpy as np


def get_feature_vector(img):

    resized = cv2.resize(img, (20, 20))

    return resized.flatten().astype(float)


def find_characters(image_path, is_template=False):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return []


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_template:
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if is_template:
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    char_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 2 and h > 5:
            roi = thresh[y:y + h, x:x + w]
            char_imgs.append(roi)

    return char_imgs


alphabet_list = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/']

small_chars_images = find_characters('alphabet-small.png', is_template=True)

reference_vectors = []
for i in range(min(len(alphabet_list), len(small_chars_images))):
    vec = get_feature_vector(small_chars_images[i])
    reference_vectors.append((alphabet_list[i], vec))

test_chars_images = find_characters('alphabet.png', is_template=False)
counts = {char: 0 for char in alphabet_list}
counts['Unknown'] = 0

for char_img in test_chars_images:
    test_vec = get_feature_vector(char_img)
    best_char = None
    min_dist = float('inf')

    for char_name, ref_vec in reference_vectors:
        dist = np.linalg.norm(test_vec - ref_vec)
        if dist < min_dist:
            min_dist = dist
            best_char = char_name
    if min_dist < 4000.0:
        counts[best_char] += 1
    else:
        counts['Unknown'] += 1

print("\nРезультаты распознавания:")
total_found = 0
for char, count in counts.items():
    if count > 0:
        print(f"'{char}': {count}")
        if char != 'Unknown':
            total_found += count
print(f"Всего распознано символов: {total_found}")
