import cv2
import numpy as np
import os


def load_images(folder_path):
    """
    Завантажує зображення з вказаної папки.
    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images


def resize_images(images, target_size):
    """
    Змінює розмір усіх зображень до вказаного розміру.
    """
    return [cv2.resize(img, target_size) for img in images]


def add_border(image, border_size, border_color=(255, 255, 255)):
    """
    Додає рамку навколо зображення.
    """
    return cv2.copyMakeBorder(
        image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=border_color
    )


def create_collage(images, grid_size=(2, 2), target_size=(300, 300),
                   border_size=10, border_color=(255, 255, 255)):
    """
    Створює колаж із зображень у вигляді сітки.

    Параметри:
    - images: список зображень
    - grid_size: кортеж (rows, cols) для розміру сітки
    - target_size: розмір кожного зображення (width, height)
    - border_size: товщина рамки між зображеннями
    - border_color: колір рамки (B, G, R)
    """
    rows, cols = grid_size
    n_images = len(images)

    # Перевірка чи достатньо зображень
    if n_images < rows * cols:
        raise ValueError(f"Недостатньо зображень. Потрібно {rows * cols}, наявно {n_images}")

    # Зміна розміру зображень
    images = resize_images(images[:rows * cols], target_size)

    # Додавання рамок
    images_with_borders = [add_border(img, border_size, border_color) for img in images]

    # Створення рядків колажу
    rows_list = []
    for i in range(0, rows * cols, cols):
        row_images = images_with_borders[i:i + cols]
        row = np.hstack(row_images)
        rows_list.append(row)

    # Об'єднання рядків у фінальний колаж
    collage = np.vstack(rows_list)

    return collage


def main():
    # Параметри колажу
    folder_path = "./images"  # Замініть на шлях до ваших зображень
    grid_size = (2, 2)  # Розмір сітки (рядки, стовпці)
    target_size = (300, 300)  # Розмір кожного зображення
    border_size = 10  # Товщина рамки
    border_color = (255, 255, 255)  # Білий колір рамки

    try:
        # Завантаження зображень
        images = load_images(folder_path)

        if not images:
            print("Не знайдено зображень у вказаній папці")
            return

        # Створення колажу
        collage = create_collage(
            images,
            grid_size=grid_size,
            target_size=target_size,
            border_size=border_size,
            border_color=border_color
        )

        # Відображення результату
        cv2.imshow('Collage', collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Збереження колажу
        cv2.imwrite('collage.jpg', collage)
        print("Колаж успішно створено та збережено як 'collage.jpg'")

    except Exception as e:
        print(f"Помилка: {str(e)}")


if __name__ == "__main__":
    main()
