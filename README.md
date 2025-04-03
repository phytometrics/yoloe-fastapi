# 🧠 YOLOE API - Real-Time Seeing Anything

このプロジェクトは、[THU-MIG/YOLOE](https://github.com/THU-MIG/yoloe) に基づいた、リアルタイム物体セグメンテーションAPIです。  
FastAPI により構築され、テキスト・ビジュアル・プロンプトフリーの3つのモードでのセグメンテーションが可能です。

---

## 🚀 起動方法

```bash
docker build -t yoloe-api .
docker run -p 8000:8000 yoloe-api
docker run --gpus all -p 8000:8000 yoloe-api
```

- 起動後、APIは `http://localhost:8000` で利用可能。
- API仕様は Swagger UI（`/docs`）から確認できます。

---

## 📦 モデルと依存の取得

初回起動時、必要なモデルは [Hugging Face Hub](https://huggingface.co/jameslahm/yoloe) から自動的にダウンロードされます。  
`Prompt-free` モードでは `tools/ram_tag_list.txt` のラベルを使用します。

---

## 📡 API エンドポイント概要

各エンドポイントには `return_image` フラグがあります：  
- `true`: PNG画像を返す（デフォルト）  
- `false`: アノテーションデータ（JSON）を返す

---

### `POST /api/predict/text`
**テキストプロンプトによるセグメンテーション**

- **Input**
  - `image`: アップロード画像
  - `texts`: `"cat, dog"` のようなカンマ区切り文字列
- **Output**
  - アノテーション画像 or JSON

**cURL例：**
```bash
curl -X POST http://localhost:8000/api/predict/text \
  -F "image=@image.jpg" \
  -F "texts=cat, dog" \
  -F 'model_id=yoloe-11m' \
  -F "return_image=false"
```

**Python例：**
```python
import requests

files = {'image': open('image.jpg', 'rb')}
data = {'texts': 'cat, dog', 'return_image': 'false', 'model_id': 'yoloe-11m'}
res = requests.post('http://localhost:8000/api/predict/text', files=files, data=data)
print(res.json())
```

---

### `GET /api/models`
使用可能なモデル一覧を返す

---

### `GET /`
API の状態を返す

---

## 💡 補足

- `return_image=false` で JSON データ（class_name / bbox / confidence）を取得できます。
- 画像を含めた出力がほしい場合は `return_image=true` にしてください。
- GPUがあれば自動的に利用されます。

# 以下のエンドポイントは未検証。

---

### `POST /api/predict/visual`
**ビジュアルプロンプト（バウンディングボックス or マスク）によるセグメンテーション**

- **Input**
  - `image`, `visual_prompt_type`, `bboxes` or `mask_image`
- **Output**
  - アノテーション画像 or JSON

**cURL例（bbox）:**
```bash
curl -X POST http://localhost:8000/api/predict/visual \
  -F "image=@image.jpg" \
  -F "visual_prompt_type=bboxes" \
  -F "bboxes=[[100,100,200,200]]" \
  -F "return_image=false"
```

---

### `POST /api/predict/prompt-free`
**プロンプト無しの自動検出モード（RAMタグ使用）**

**cURL例：**
```bash
curl -X POST http://localhost:8000/api/predict/prompt-free \
  -F "image=@image.jpg" \
  -F "return_image=false"
```

**Python例：**
```python
files = {'image': open('image.jpg', 'rb')}
data = {'return_image': 'false'}
res = requests.post('http://localhost:8000/api/predict/prompt-free', files=files, data=data)
print(res.json())
```
