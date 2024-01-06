import logging
import sys


def setup_logging():
    """
    共通のログ設定を行う関数。
    ログレベルはINFOで、出力先は標準出力。
    """
    # ロガーの設定
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 標準出力へのハンドラー設定
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # ログのフォーマット設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # ロガーにハンドラーを追加
    logger.addHandler(handler)
