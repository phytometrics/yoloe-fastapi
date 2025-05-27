import torch
import numpy as np
import supervision as sv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.utils.torch_utils import smart_inference_mode
from huggingface_hub import hf_hub_download
import traceback

from typing import List, Dict, Any

app = FastAPI(
    title="YOLOE API",
    description="YOLOE: Real-Time Seeing Anything API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cache = {}

def init_model(model_id, is_pf=False):
    cache_key = f"{model_id}_{is_pf}"
    if cache_key in model_cache:
        return model_cache[cache_key]
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model_cache[cache_key] = model
    return model

def extract_detections(results, model):
    detections = sv.Detections.from_ultralytics(results[0])
    output = []
    for class_id, confidence, bbox in zip(detections.class_id, detections.confidence, detections.xyxy):
        print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {bbox}")
        output.append({
            "class_name": model.names[int(class_id)],
            "confidence": float(confidence),
            "bbox": bbox.tolist()
        })
    return output, detections

def detections_to_dict(detections: sv.Detections) -> List[Dict[str, Any]]:
    results = []
    for i in range(len(detections)):
        detection = {
            "box": detections.xyxy[i].tolist(),  # Convert NumPy array to list
            "confidence": float(detections.confidence[i]) if detections.confidence is not None else None,
            "class_id": int(detections.class_id[i]) if detections.class_id is not None else None,
            "tracker_id": int(detections.tracker_id[i]) if detections.tracker_id is not None else None,
        }
        # Include additional data if present
        for key, value in detections.data.items():
            detection[key] = value[i].tolist() if isinstance(value[i], np.ndarray) else value[i]
        results.append(detection)
    return results

def create_image_response(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    # include output and image
    # response_content = {
    #     "detections": output,
    #     "image": base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    # }
    # return JSONResponse(content=response_content)
    return StreamingResponse(content=img_byte_arr, media_type="image/png")

def annotate_image(image, detections):
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    # labels = [
    #     f"{class_name} {confidence:.2f}"
    #     for class_name, confidence
    #     in zip(detections['class_name'], detections.confidence)
    # ]
    annotated = image.copy()
    annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(annotated, detections)
    annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(annotated, detections)
    # annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(annotated, detections, labels)
    return annotated

@app.post("/api/predict/text")
async def predict_text_multi(
    images: List[UploadFile] = File(...),
    texts: str = Form(...),
    model_id: str = Form("yoloe-11l"),
    image_size: int = Form(640),
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.70),
    return_image: bool = Form(True)
) -> JSONResponse:
    try:
        text_list = [t.strip() for t in texts.split(",")]
        model = init_model(model_id)
        model.set_classes(text_list, model.get_text_pe(text_list))

        results_list = []

        for image in images:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            results = model.predict(source=pil_image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh)
            detections = sv.Detections.from_ultralytics(results[0])

            item = {
                "filename": image.filename,
                "detections": detections_to_dict(detections),
            }

            if return_image:
                annotated = annotate_image(pil_image, detections)
                img_byte_arr = io.BytesIO()
                annotated.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                item["image_base64"] = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            results_list.append(item)

        return JSONResponse(content={"results": results_list})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/predict/visual")
# async def predict_visual(
#     image: UploadFile = File(...),
#     visual_prompt_type: str = Form(...),
#     visual_usage_type: str = Form("Intra-Image"),
#     model_id: str = Form("yoloe-v8l"),
#     image_size: int = Form(640),
#     conf_thresh: float = Form(0.25),
#     iou_thresh: float = Form(0.70),
#     bboxes: str = Form(None),
#     target_image_base64: str = Form(None),
#     mask_image: UploadFile = File(None),
#     return_image: bool = Form(True)
# ):
#     try:
#         contents = await image.read()
#         pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
#         target_image = None
#         if visual_usage_type == "Cross-Image" and target_image_base64:
#             target_image_data = base64.b64decode(target_image_base64)
#             target_image = Image.open(io.BytesIO(target_image_data)).convert("RGB")
#         prompts = {}
#         if visual_prompt_type == "bboxes" and bboxes:
#             bbox_list = np.array(eval(bboxes))
#             prompts = {"bboxes": bbox_list, "cls": np.zeros(len(bbox_list), dtype=int)}
#         elif visual_prompt_type == "masks" and mask_image:
#             mask_contents = await mask_image.read()
#             mask = Image.open(io.BytesIO(mask_contents)).convert("L")
#             mask_array = binary_fill_holes(np.array(mask)).astype(np.uint8)
#             mask_array[mask_array > 0] = 1
#             if mask_array.sum() == 0:
#                 raise HTTPException(status_code=400, detail="Empty mask image.")
#             prompts = {"masks": mask_array[None], "cls": np.array([0])}
#         else:
#             raise HTTPException(status_code=400, detail="Invalid prompt setup.")
#         model = init_model(model_id)
#         kwargs = dict(prompts=prompts, predictor=YOLOEVPSegPredictor)
#         if target_image:
#             model.predict(source=pil_image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
#             model.set_classes(["object0"], model.predictor.vpe)
#             model.predictor = None
#             pil_image = target_image
#             kwargs = {}
#         results = model.predict(source=pil_image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
#         output, detections = extract_detections(results, model)
#         if return_image:
#             annotated = annotate_image(pil_image, detections)
#             return create_image_response(annotated)
#         return JSONResponse(content={"detections": output})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/predict/prompt-free")
# async def predict_prompt_free(
#     image: UploadFile = File(...),
#     model_id: str = Form("yoloe-v8l"),
#     image_size: int = Form(640),
#     conf_thresh: float = Form(0.25),
#     iou_thresh: float = Form(0.70),
#     return_image: bool = Form(True)
# ):
#     try:
#         contents = await image.read()
#         pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
#         with open('tools/ram_tag_list.txt', 'r') as f:
#             texts = [x.strip() for x in f.readlines()]
#         prompts = { "texts": texts }
#         model = init_model(model_id, is_pf=True)
#         vocab = model.get_vocab(texts)
#         model.set_vocab(vocab, names=texts)

#         model.model.model[-1].conf = 0.001
#         model.model.model[-1].max_det = 1000
#         model.model.model[-1].is_fused = True
#         results = model.predict(source=pil_image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh)
#         output, detections = extract_detections(results, model)
#         if return_image:
#             annotated = annotate_image(pil_image, detections)
#             return create_image_response(annotated)
#         return JSONResponse(content={"detections": output})
#     except Exception as e:
#         traceback.print_exc()  # ← これでコンソールにスタックトレースが出る
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    return {
        "models": [
            "yoloe-v8s",
            "yoloe-v8m",
            "yoloe-v8l",
            "yoloe-11s",
            "yoloe-11m",
            "yoloe-11l",
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "YOLOE API is running. Visit /docs for API documentation.",
        "documentation_url": "/docs",
        "github": "https://github.com/THU-MIG/yoloe",
        "paper": "https://arxiv.org/abs/2503.07465"
    }
