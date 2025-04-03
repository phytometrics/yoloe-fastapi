# ベースイメージとしてUltralytics GPUイメージを使用
FROM ultralytics/ultralytics:latest

# FastAPIと関連パッケージをインストール
RUN pip install fastapi uvicorn[standard] python-multipart supervision huggingface-hub
RUN pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/CLIP"
RUN pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip"
RUN pip install "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/lvis-api"
#RUN pip install "git+https://github.com/THU-MIG/yoloe.git"



# 作業ディレクトリ作成
WORKDIR /app/yoloe

# 先に download.py と必要ファイルをコピーして実行
COPY download.py .
RUN wget -q https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
RUN python download.py

# 最後にコード本体をコピー（ここを最後にすることでキャッシュ活用）
COPY ./yoloe /app/yoloe
RUN pip install -e .


# FastAPIサーバーを起動するコマンド
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
