# wsgi.py
import torch
from smart_getenv import getenv
from app import create_app
from app.classifier import Classifier
# パラメータファイルのパスを環境変数から取得
prm_file = getenv("PRM_FILE", default="/data/taco_burrito.prm")
# パラメータファイルを読み込む
params = torch.load(prm_file,
                    map_location=lambda storage, loc: storage)
# ClassifierとFlaskアプリケーションを作成
classifier = Classifier(params)
app = create_app(classifier)
