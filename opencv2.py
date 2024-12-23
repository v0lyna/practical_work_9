import cv2
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def preprocess_image(image):
    """
    Попередня обробка зображення для покращення розпізнавання.
    """
    # Конвертація в відтінки сірого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Білатеральна фільтрація для зменшення шуму зі збереженням країв
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # Виявлення країв using Canny
    edges = cv2.Canny(bilateral, 30, 200)

    return gray, edges


def find_plate_contour(edges, original_image):
    """
    Пошук контуру номерного знаку на зображенні.
    """
    # Пошук контурів
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Перевірка чи контур має 4 вершини (прямокутник)
        if len(approx) == 4:
            plate_contour = approx
            break

    return plate_contour


def extract_plate(image, plate_contour):
    """
    Вирізання області номерного знаку та його обробка.
    """
    if plate_contour is None:
        return None

    # Створення маски для номерного знаку
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, (255, 255, 255), -1)

    # Застосування маски
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Отримання обмежувального прямокутника
    (x, y, w, h) = cv2.boundingRect(plate_contour)
    plate = masked[y:y + h, x:x + w]

    return plate


def recognize_text(plate_image):
    """
    Розпізнавання тексту з зображення номерного знаку.
    """
    # Конвертація в відтінки сірого
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Бінаризація зображення
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Конвертація в формат PIL Image
    pil_image = Image.fromarray(binary)

    # Розпізнавання тексту за допомогою Tesseract
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(pil_image, config=custom_config)

    return text.strip()


def process_license_plate(image_path):
    """
    Головна функція обробки зображення та розпізнавання номерного знаку.
    """
    # Завантаження зображення
    image = cv2.imread(image_path)
    if image is None:
        return "Помилка: Неможливо завантажити зображення"

    # Попередня обробка
    gray, edges = preprocess_image(image)

    # Пошук контуру номерного знаку
    plate_contour = find_plate_contour(edges, image)
    if plate_contour is None:
        return "Номерний знак не знайдено"

    # Вирізання області номерного знаку
    plate = extract_plate(image, plate_contour)
    if plate is None:
        return "Помилка при вирізанні номерного знаку"

    # Розпізнавання тексту
    text = recognize_text(plate)

    # Візуалізація результату
    cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
    cv2.imshow('Знайдений номерний знак', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text


# Приклад використання
if __name__ == "__main__":
    image_path = "./images/cars.jpg"  # Вкажіть шлях до вашого зображення
    result = process_license_plate(image_path)
    print(f"Розпізнаний номер: {result}")
