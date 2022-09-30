# classifier.py
from torch import nn
from torchvision import transforms, models
def create_network():
    # resnet18を読み込む。
    # パラメータは後でセットするのでpretrained=Trueは必要ない
    net = models.resnet18()
    
    # 最後の層を2出力の線形層に付け替え
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 2)
    return net
class Classifier(object):
    def __init__(self, params):
        # 識別のネットワークを作成
        self.net = create__network()
        # 学習済みパラメータのセット
        self.net.load_state_dict(params)
        # 評価モードにする
        self.net.eval()
        # 画像を整形してTensorにする関数
        self.transformer = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        # クラスのIDと名前の対応
        self.classes = ["burrito", "taco"]
    def predict(self, img):
        # 画像を整形してTensorに変換
        x = self.transformer(img)
        # PyTorchは常にバッチで処理するので
        # batchの次元を先頭に追加
        x = x.unsqueeze(0)
        # ネットワークの出力を計算
        out = self.net(x)
        out = out.max(1)[1].item()
        # 予測されたクラス名を返す
        return self.classes[out]
