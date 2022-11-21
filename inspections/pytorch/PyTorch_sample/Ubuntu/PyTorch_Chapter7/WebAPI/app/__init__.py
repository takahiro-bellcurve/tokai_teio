# __init__.py
from flask import Flask, request, jsonify
from PIL import Image

def create_app(classifier):
    # Flaskアプリケーションを生成
    app = Flask(__name__)
    # POST / に対応する関数を定義
    @app.route("/", methods=["POST"])
    def predict():
        # 受け取ったファイルのハンドラーを取得
        img_file = request.files["img"]
        
        # ファイルが空かどうかチェック
        if img_file.filename == "":
            return "Bad Request", 400
   
        # PILを使用して画像ファイルを読み込む
        img = Image.open(img_file)
        
        # 識別モデルを使用してタコスかブリトーかを予測
        result = classifier.predict(img)
        
        # 結果をJSON形式で返す
        return jsonify({
            "result": result
        })
    return app
