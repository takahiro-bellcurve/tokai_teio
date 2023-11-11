from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
from rembg import remove
from PIL import Image

from src.lib.image_preprocessor import ImagePreprocessor


def main():
    original_images = Path('data/original_images')
    print(original_images)
    images = list(original_images.glob('*.jpg'))

    i = 0
    for image in images:
        if i > 5000:
            break
        base_image = Image.open(image)
        fill_white_bg_image = ImagePreprocessor.fill_white_background(
            base_image)
        resized_image = ImagePreprocessor.resize(fill_white_bg_image, 64, 64)
        gray_scaled_image = ImagePreprocessor.convert_to_grayscale(
            resized_image)
        gray_scaled_image.save(
            f'data/preprocessed_images_64_with_gray_scale/{image.name}')
        print(f"Saved {image.name}")
        i += 1


if __name__ == '__main__':
    print("Start creating preprocessed images...")
    main()
