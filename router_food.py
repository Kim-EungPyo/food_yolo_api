# router_food.py
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import yaml
from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO

router = APIRouter()

# ── 경로/하이퍼파라미터 ──────────────────────────────────────
ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "model"                     # ← 폴더명 'model'
WEIGHTS   = Path(os.getenv("YOLO_WEIGHTS", MODEL_DIR / "single_model_final.pt"))
DATA_YAML = MODEL_DIR / "data_100.yaml"

IMG_SZ  = int(os.getenv("YOLO_IMG",  "640"))
CONF_TH = float(os.getenv("YOLO_CONF", "0.25"))
IOU_TH  = float(os.getenv("YOLO_IOU",  "0.45"))
DEVICE  = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ── 클래스 이름 로드(없으면 모델 내 names 사용) ──────────────
NAMES = None
if DATA_YAML.exists():
    with open(DATA_YAML, encoding="utf-8") as f:
        NAMES = yaml.safe_load(f).get("names", None)

# ── 단일 모델 싱글톤 로드 ────────────────────────────────────
_MODEL = None
def get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        assert WEIGHTS.exists(), f"가중치를 찾을 수 없습니다: {WEIGHTS}"
        _MODEL = YOLO(str(WEIGHTS))
    return _MODEL

# ── API: reps 리스트만 반환 ──────────────────────────────────
@router.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 지원합니다.")

    raw = await file.read()
    img_np = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩하지 못했습니다.")

    model = get_model()
    results = model.predict(
        source=img_np,        # numpy 배열 직접 입력
        imgsz=IMG_SZ,
        conf=CONF_TH,
        iou=IOU_TH,
        device=DEVICE,
        verbose=False,
        save=False,           # 파일 저장 없음
    )
    r = results[0]
    names_from_model = getattr(model.model, "names", getattr(model, "names", None)) or {}

    reps = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            xyxy   = b.xyxy[0].tolist()        # [x1,y1,x2,y2]
            cls_id = int(b.cls[0].item())
            conf_v = float(b.conf[0].item())

            # 라벨 이름: data_100.yaml → 모델 내 names → cls_id
            if isinstance(NAMES, list) and 0 <= cls_id < len(NAMES):
                label = NAMES[cls_id]
            elif isinstance(names_from_model, dict):
                label = names_from_model.get(cls_id, str(cls_id))
            elif isinstance(names_from_model, list) and 0 <= cls_id < len(names_from_model):
                label = names_from_model[cls_id]
            else:
                label = str(cls_id)

            reps.append({
                "bbox":   xyxy,
                "conf":   conf_v,
                "cls_id": cls_id,
                "label":  label,
                "model":  0,   # 단일 모델이므로 0 고정
            })

    # 요청하신 대로 reps만 반환
    return reps
