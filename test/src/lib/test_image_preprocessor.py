import pytest
import os
from PIL import Image

from src.lib.image_preprocessor import ImagePreprocessor


class TestImagePreprocessor:

    def test_open_image(self):
        # パスから画像を開くテスト
        image_path = './test/data/sample.jpg'
        image = ImagePreprocessor.open_image(image_path)
        assert isinstance(image, Image.Image)

    def test_remove_background(self):
        # 背景を削除するテスト
        image = Image.open('./test/data/sample.jpg')
        removed_image = ImagePreprocessor.remove_background(image)
        assert removed_image is not None

    def test_trim(self):
        # 余白をトリムするテスト
        image = Image.open('./test/data/sample.jpg')
        trimmed_image = ImagePreprocessor.trim(image)
        assert trimmed_image is not None

    def test_resize(self):
        # サイズを調整するテスト
        image = Image.open('./test/data/sample.jpg')
        resized_image = ImagePreprocessor.resize(image, 128, 128)
        assert resized_image is not None
        assert resized_image.size == (128, 128)

    def test_fill_white_background(self):
        # 背景を白で塗りつぶすテスト
        image = Image.open('./test/data/sample.jpg')
        filled_image = ImagePreprocessor.fill_white_background(image)
        assert filled_image is not None
        assert filled_image.mode == 'RGB'

    def test_convert_to_grayscale(self):
        # グレースケールに変換するテスト
        image = Image.open('./test/data/sample.jpg')
        grayscale_image = ImagePreprocessor.convert_to_grayscale(image)
        assert grayscale_image is not None
        assert grayscale_image.mode == 'L'

    def test_save_image(self):
        # 画像を保存するテスト
        image = Image.open('./test/data/sample.jpg')
        path = './output.jpg'
        ImagePreprocessor.save_image(image, path)
        assert os.path.exists(path)
