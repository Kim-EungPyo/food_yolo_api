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
MODEL_DIR = ROOT / "model"                      # 폴더명 'model'
WEIGHTS   = MODEL_DIR / "best.pt"
DATA_YAML = MODEL_DIR / "data_100.yaml"

IMG_SZ  = int(os.getenv("YOLO_IMG",  "640"))
CONF_TH = float(os.getenv("YOLO_CONF", "0.5"))  # ← 요청: conf 0.5
IOU_TH  = float(os.getenv("YOLO_IOU",  "0.5"))  # ← 요청: IoU 0.25 (기본값 변경 없음)
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

def _decode_image(raw_bytes: bytes, filename: str | None, content_type: str | None):
    """
    jpg/png 등은 OpenCV로, HEIC/HEIF는 pillow-heif + Pillow로 디코딩하여 BGR(np.ndarray) 반환.
    """
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    is_heic = name.endswith((".heic", ".heif")) or ("image/heic" in ctype) or ("image/heif" in ctype)
    if is_heic:
        try:
            # 지연 임포트: pillow-heif가 없을 경우 명확한 에러 메시지
            from pillow_heif import read_heif
            from PIL import Image

            heif = read_heif(raw_bytes)
            pil_img = Image.frombytes(heif.mode, heif.size, heif.data)

            # OpenCV로 넘기기 위해 RGB -> BGR, 알파 채널 제거
            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
            if pil_img.mode == "RGBA":
                pil_img = pil_img.convert("RGB")

            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img_np
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="HEIC 지원이 설정되지 않았습니다. 서버에 'pillow-heif' 패키지를 설치하세요 (pip install pillow-heif)."
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"HEIC 디코딩 실패: {e}")
    else:
        # 일반 포맷은 OpenCV로
        img_np = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img_np

# ── API: reps 리스트만 반환 ──────────────────────────────────
@router.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 지원합니다.")

    raw = await file.read()
    img_np = _decode_image(raw, file.filename, file.content_type)

    if img_np is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩하지 못했습니다.")

    model = get_model()
    results = model.predict(
        source=img_np,        # numpy 배열 직접 입력
        imgsz=IMG_SZ,
        conf=CONF_TH,         # conf ≥ 0.5만 남도록 모델 내부 필터
        iou=IOU_TH,           # NMS IoU
        device=DEVICE,
        verbose=False,
        save=False,           # 파일 저장 없음
    )
    r = results[0]

    # 모델 내 names 우선 사용 (data.yaml 없는 경우 대비)
    names_from_model = getattr(model.model, "names", getattr(model, "names", None)) or {}

    # --- 클래스별 최고 confidence 1개만 유지 ---
    best_by_label = {}  # label -> dict
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            xyxy   = b.xyxy[0].tolist()        # [x1,y1,x2,y2]
            cls_id = int(b.cls[0].item())
            conf_v = float(b.conf[0].item())

            # 라벨 이름: data_100.yaml → 모델 내 names → cls_id(문자열)
            if isinstance(NAMES, list) and 0 <= cls_id < len(NAMES):
                label = NAMES[cls_id]
            elif isinstance(names_from_model, dict):
                label = names_from_model.get(cls_id, str(cls_id))
            elif isinstance(names_from_model, list) and 0 <= cls_id < len(names_from_model):
                label = names_from_model[cls_id]
            else:
                label = str(cls_id)

            cand = {
                "bbox":   xyxy,
                "conf":   conf_v,
                "cls_id": cls_id,
                "label":  label,
                "model":  0,   # 단일 모델이므로 0 고정
            }

            # 같은 label 중 confidence가 더 높으면 교체
            if (label not in best_by_label) or (conf_v > best_by_label[label]["conf"]):
                best_by_label[label] = cand

    # 요청하신 대로 reps만 반환 (클래스당 1개)
    reps = list(best_by_label.values())
    return reps
