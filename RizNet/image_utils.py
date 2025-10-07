import base64
from io import BytesIO
from PIL import Image


def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """
    Центрована обрізка зображення до вказаного розміру.

    Args:
        img (Image.Image): Початкове зображення.
        width (int): Потрібна ширина.
        height (int): Потрібна висота.

    Returns:
        Image.Image: Обрізане зображення.
    """
    w, h = img.size  # Поточні розміри зображення
    left = (w - width) / 2
    top = (h - height) / 2
    right = (w + width) / 2
    bottom = (h + height) / 2

    return img.crop((left, top, right, bottom))


def decode_img(img_base64: str) -> Image.Image:
    """
    Декодує base64-рядок в об'єкт зображення PIL.

    Args:
        img_base64 (str): Base64-закодований рядок зображення.

    Returns:
        Image.Image: Об'єкт зображення PIL.
    """
    # Видаляємо префікс data URL, якщо він є
    if img_base64.startswith("data:image/"):
        header, encoded = img_base64.split(",", 1)
    else:
        encoded = img_base64

    # Декодуємо base64 в байти
    img_bytes = base64.b64decode(encoded)

    # Створюємо об'єкт BytesIO для читання байтів
    img_buffer = BytesIO(img_bytes)

    # Відкриваємо зображення за допомогою PIL
    return Image.open(img_buffer).convert("RGB")