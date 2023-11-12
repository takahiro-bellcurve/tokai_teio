import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from src.lib.image_preprocessor import ImagePreprocessor

DOWNLOAD_DIR = 'preprocessed_images_512'


def process_image(image_path, output_dir):
    image_name = image_path.name
    output_path = output_dir / image_name

    if output_path.exists():
        return f"Skipped {image_name} (already exists)"

    try:
        image = Image.open(image_path)
        image = ImagePreprocessor.remove_background(image)
        image = ImagePreprocessor.fill_white_background(image)
        image = ImagePreprocessor.resize(image, 512, 512)
        image.save(output_path)
        return f"Saved {image_name}"
    except Exception as e:
        return f"Error processing {image_name}: {e}"


async def main():
    original_images_dir = Path('data/original_images')
    processed_images_dir = Path(f'data/{DOWNLOAD_DIR}')
    os.makedirs(processed_images_dir, exist_ok=True)

    original_images = [img for img in original_images_dir.glob(
        '*.jpg') if not (processed_images_dir / img.name).exists()]
    executor = ProcessPoolExecutor(max_workers=4)

    for i in range(0, len(original_images), 4):
        batch = original_images[i:i + 4]
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(
            executor, process_image, img, processed_images_dir) for img in batch]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)

if __name__ == '__main__':
    asyncio.run(main())
