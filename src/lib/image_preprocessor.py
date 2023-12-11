import requests
from torchvision import transforms
from PIL import Image, ImageOps, ImageChops
from io import BytesIO
from rembg import remove


class ImagePreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def download_image(image_url):
        """
        指定されたURLから画像をダウンロードし、PIL.Imageオブジェクトとして返す。

        :param image_url: ダウンロードする画像のURL
        :return: PIL.Imageオブジェクト
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading the image: {e}")
            return None

    @staticmethod
    def remove_background(image):
        """
        PIL.Imageオブジェクトから背景を削除する関数。

        :param image: 背景を削除するPIL.Imageオブジェクト
        :return: 背景が削除されたPIL.Imageオブジェクト
        """
        try:
            # PIL.Imageオブジェクトをバイト列に変換
            img_bytes = BytesIO()
            image.save(img_bytes, format=image.format)

            # 背景を削除
            output_bytes = remove(img_bytes.getvalue())

            # バイト列からPIL.Imageオブジェクトを再生成
            result_image = Image.open(BytesIO(output_bytes))
            return result_image

        except Exception as e:
            print(f"An error occurred while removing the background: {e}")
            return None

    @staticmethod
    def trim(image, border_color=(255, 255, 255, 0)):
        """
        画像の余白をトリムする。

        :param image: PIL.Imageオブジェクト
        :param border_color: トリムする背景色。デフォルトは透明または白。
        :return: トリムされたPIL.Imageオブジェクト
        """
        bg = Image.new(image.mode, image.size, border_color)
        diff = ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image  # トリムする必要がない場合

    @staticmethod
    def resize(image, width, height, to_tensor=False):
        """
        画像のサイズを調整するメソッド。

        :param image: PIL.Imageオブジェクト
        :param width: 変更後の幅
        :param height: 変更後の高さ
        :return: サイズ変更されたPIL.Imageオブジェクト
        """
        # 余白をトリムし、その後10pxの余白を追加
        image = ImagePreprocessor.trim(image)
        image = ImageOps.expand(image, border=10, fill=image.getpixel((0, 0)))

        # 横幅と縦幅を比較し、大きい方を基準にリサイズする
        aspect_ratio = image.width / image.height
        if image.width > image.height:
            new_height = int(width / aspect_ratio)
            resized_image = image.resize(
                (width, new_height), Image.Resampling.LANCZOS)
        else:
            new_width = int(height * aspect_ratio)
            resized_image = image.resize(
                (new_width, height), Image.Resampling.LANCZOS)

        # 背景の色を決定（透明か白か）
        background_color = (
            255, 255, 255, 0) if image.mode == 'RGBA' else (255, 255, 255)

        # 最終的な画像のサイズを作成し、中央にリサイズした画像を配置
        final_image = Image.new(image.mode, (width, height), background_color)
        final_image.paste(resized_image, ((
            width - resized_image.width) // 2, (height - resized_image.height) // 2))

        if to_tensor:
            final_image = transforms.ToTensor()(final_image)

        return final_image

    @staticmethod
    def fill_white_background(image):
        """
        透明な背景を持つPIL.Imageオブジェクトの背景を白で塗りつぶす。

        :param image: 背景を白で塗りつぶすPIL.Imageオブジェクト
        :return: 背景が白で塗りつぶされたPIL.Imageオブジェクト
        """
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            with Image.new("RGB", image.size, (255, 255, 255)) as background:
                background.paste(image, mask=image.split()[3])
                return background
        else:
            return image

    @staticmethod
    def convert_to_grayscale(image):
        """
        PIL.Imageオブジェクトをグレースケールに変換する。

        :param image: PIL.Imageオブジェクト
        :return: グレースケール化されたPIL.Imageオブジェクト
        """
        return image.convert('L')

    @staticmethod
    def save_image(image, path):
        """
        PIL.Imageオブジェクトを指定されたパスに保存する。

        :param image: PIL.Imageオブジェクト
        :param path: 保存先のパス
        """
        image.save(path)
