"""PP-DocLayout region detection and box overlay."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import cv2
import numpy as np
from classic_enhance import ImageInput, _to_rgb

DEFAULT_LAYOUT_MODEL = "PP-DocLayoutV3"
LAYOUT_MODELS: tuple[str, ...] = (
    "PP-DocLayoutV3",
    "PP-DocLayoutV2",
    "PP-DocLayout_plus-L",
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout-S",
)
DEFAULT_DET_MODEL = "PP-OCRv6_medium_det"
DET_MODELS: tuple[str, ...] = (
    "PP-OCRv6_medium_det",
    "PP-OCRv5_server_det",
)
DEFAULT_REC_MODEL = "korean_PP-OCRv5_mobile_rec"
REC_MODELS: tuple[str, ...] = (
    "korean_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_rec",
)
DetectMode = Literal["layout", "text"]
DETECT_MODES: tuple[DetectMode, ...] = ("layout", "text")

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "text": (46, 125, 50),
    "paragraph_title": (25, 118, 210),
    "doc_title": (123, 31, 162),
    "title": (123, 31, 162),
    "table": (230, 81, 0),
    "table_title": (239, 108, 0),
    "table_caption": (251, 140, 0),
    "figure": (0, 151, 167),
    "image": (0, 151, 167),
    "chart": (0, 121, 107),
    "figure_title": (0, 137, 123),
    "figure_caption": (38, 166, 154),
    "formula": (194, 24, 91),
    "display_formula": (194, 24, 91),
    "formula_number": (233, 30, 99),
    "header": (97, 97, 97),
    "footer": (97, 97, 97),
    "header_image": (66, 66, 66),
    "footer_image": (66, 66, 66),
    "page_number": (117, 117, 117),
    "footnote": (141, 110, 99),
    "abstract": (81, 45, 168),
    "content": (56, 142, 60),
    "reference": (69, 90, 100),
    "references": (69, 90, 100),
    "algorithm": (21, 101, 192),
    "seal": (183, 28, 28),
    "aside_text": (121, 85, 72),
    "sidebar_text": (121, 85, 72),
    "list": (67, 160, 71),
    "number": (117, 117, 117),
    "vertical_text": (46, 125, 50),
    "inline_formula": (194, 24, 91),
}
_FALLBACK_COLOR = (33, 33, 33)

Predictor = Callable[[np.ndarray], Any]


def layout_ready() -> bool:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class LayoutBox:
    label: str
    score: float
    coordinate: tuple[float, float, float, float]
    cls_id: int = 0
    text: str = ""
    rec_score: float = 0.0


@dataclass
class LayoutResult:
    boxes: list[LayoutBox]
    annotated: np.ndarray
    elapsed_sec: float
    model_name: str
    rec_model_name: str | None = None
    rec_elapsed_sec: float = 0.0


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    json_attr = getattr(raw, "json", None)
    if isinstance(json_attr, dict):
        return json_attr
    if hasattr(raw, "keys"):
        try:
            return dict(raw)
        except Exception:  # noqa: BLE001
            pass
    raise TypeError(f"Unsupported layout result type: {type(raw)!r}")


def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value)


def parse_layout_boxes(raw: Any) -> list[LayoutBox]:
    data = _as_mapping(raw)
    if "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    items = _seq(data.get("boxes"))
    boxes: list[LayoutBox] = []
    for item in items:
        coord = item["coordinate"]
        boxes.append(
            LayoutBox(
                label=str(item.get("label", "")),
                score=float(item.get("score", 0.0)),
                coordinate=(
                    float(coord[0]),
                    float(coord[1]),
                    float(coord[2]),
                    float(coord[3]),
                ),
                cls_id=int(item.get("cls_id", 0)),
            )
        )
    return boxes


def parse_text_boxes(raw: Any) -> list[LayoutBox]:
    data = _as_mapping(raw)
    if "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    polys = _seq(data.get("dt_polys"))
    scores = _seq(data.get("dt_scores"))
    boxes: list[LayoutBox] = []
    for i, poly in enumerate(polys):
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        score = float(scores[i]) if i < len(scores) else 0.0
        boxes.append(
            LayoutBox(
                label="text",
                score=score,
                coordinate=(min(xs), min(ys), max(xs), max(ys)),
                cls_id=0,
            )
        )
    return boxes


def parse_rec_result(raw: Any) -> tuple[str, float]:
    data = _as_mapping(raw)
    if "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    return str(data.get("rec_text", "")), float(data.get("rec_score", 0.0))


def crop_box(
    image: np.ndarray,
    box: LayoutBox,
    *,
    pad: int = 2,
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box.coordinate)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _color_for(label: str) -> tuple[int, int, int]:
    return _LABEL_COLORS.get(label, _FALLBACK_COLOR)


def draw_layout_boxes(
    image: np.ndarray,
    boxes: Sequence[LayoutBox],
    *,
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of an RGB image with labeled layout rectangles."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    h, w = out.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.coordinate)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        color = _color_for(box.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if not show_labels:
            continue
        if box.text:
            caption = f"{box.text} {box.rec_score:.2f}"
        else:
            caption = f"{box.label} {box.score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y1 + th + 4
        cv2.rectangle(
            out,
            (x1, ty - th - 2),
            (x1 + tw + 4, ty + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            out,
            caption,
            (x1 + 2, ty),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


_cached_models: dict[str, Any] = {}


def get_layout_model(model_name: str = DEFAULT_LAYOUT_MODEL) -> Any:
    if model_name not in _cached_models:
        from paddleocr import LayoutDetection

        _cached_models[model_name] = LayoutDetection(
            model_name=model_name,
            device="cpu",
        )
    return _cached_models[model_name]


def _first_output(outputs: Any) -> Any:
    try:
        return next(iter(outputs))
    except StopIteration as exc:
        raise RuntimeError("detection returned no results") from exc


def detect_layout(
    image: ImageInput,
    *,
    model_name: str = DEFAULT_LAYOUT_MODEL,
    predictor: Predictor | None = None,
    threshold: float | None = None,
) -> LayoutResult:
    """Detect layout regions and return boxes + annotated RGB image."""
    rgb = _to_rgb(image)
    t0 = time.perf_counter()
    if predictor is None:
        model = get_layout_model(model_name)

        def predictor(img: np.ndarray) -> Any:
            outputs = model.predict(
                img,
                batch_size=1,
                layout_nms=True,
                threshold=threshold,
            )
            return _first_output(outputs)

    raw = predictor(rgb)
    boxes = parse_layout_boxes(raw)
    annotated = draw_layout_boxes(rgb, boxes)
    return LayoutResult(
        boxes=boxes,
        annotated=annotated,
        elapsed_sec=time.perf_counter() - t0,
        model_name=model_name,
    )


def get_text_det_model(model_name: str = DEFAULT_DET_MODEL) -> Any:
    if model_name not in _cached_models:
        from paddleocr import TextDetection

        _cached_models[model_name] = TextDetection(
            model_name=model_name,
            device="cpu",
        )
    return _cached_models[model_name]


def detect_text_lines(
    image: ImageInput,
    *,
    model_name: str = DEFAULT_DET_MODEL,
    predictor: Predictor | None = None,
) -> LayoutResult:
    """Detect text lines and return boxes + annotated RGB image."""
    rgb = _to_rgb(image)
    t0 = time.perf_counter()
    if predictor is None:
        model = get_text_det_model(model_name)

        def predictor(img: np.ndarray) -> Any:
            outputs = model.predict(img, batch_size=1)
            return _first_output(outputs)

    raw = predictor(rgb)
    boxes = parse_text_boxes(raw)
    annotated = draw_layout_boxes(rgb, boxes, show_labels=False)
    return LayoutResult(
        boxes=boxes,
        annotated=annotated,
        elapsed_sec=time.perf_counter() - t0,
        model_name=model_name,
    )


RecPredictor = Callable[[list[np.ndarray]], Sequence[Any]]


def get_text_rec_model(model_name: str = DEFAULT_REC_MODEL) -> Any:
    key = f"rec:{model_name}"
    if key not in _cached_models:
        from paddleocr import TextRecognition

        _cached_models[key] = TextRecognition(
            model_name=model_name,
            device="cpu",
        )
    return _cached_models[key]


def recognize_boxes(
    image: ImageInput,
    boxes: Sequence[LayoutBox],
    *,
    model_name: str = DEFAULT_REC_MODEL,
    predictor: RecPredictor | None = None,
    pad: int = 2,
    batch_size: int = 16,
) -> list[LayoutBox]:
    """Run text recognition on each box crop and return boxes with text filled."""
    rgb = _to_rgb(image)
    if not boxes:
        return []
    crops = [crop_box(rgb, box, pad=pad) for box in boxes]
    if predictor is None:
        model = get_text_rec_model(model_name)

        def predictor(batch: list[np.ndarray]) -> Sequence[Any]:
            return list(model.predict(batch, batch_size=len(batch)))

    texts: list[tuple[str, float]] = []
    for i in range(0, len(crops), batch_size):
        chunk = crops[i : i + batch_size]
        outputs = list(predictor(chunk))
        if len(outputs) != len(chunk):
            outputs = []
            for crop in chunk:
                outputs.extend(list(predictor([crop])))
        for raw in outputs:
            texts.append(parse_rec_result(raw))

    return [
        LayoutBox(
            label=box.label,
            score=box.score,
            coordinate=box.coordinate,
            cls_id=box.cls_id,
            text=text,
            rec_score=score,
        )
        for box, (text, score) in zip(boxes, texts, strict=True)
    ]


def detect_and_recognize(
    image: ImageInput,
    *,
    det_model: str = DEFAULT_DET_MODEL,
    rec_model: str = DEFAULT_REC_MODEL,
    det_predictor: Predictor | None = None,
    rec_predictor: RecPredictor | None = None,
) -> LayoutResult:
    """Text-line detection followed by recognition."""
    det = detect_text_lines(image, model_name=det_model, predictor=det_predictor)
    t0 = time.perf_counter()
    boxes = recognize_boxes(
        image,
        det.boxes,
        model_name=rec_model,
        predictor=rec_predictor,
    )
    rec_elapsed = time.perf_counter() - t0
    rgb = _to_rgb(image)
    annotated = draw_layout_boxes(rgb, boxes, show_labels=True)
    return LayoutResult(
        boxes=boxes,
        annotated=annotated,
        elapsed_sec=det.elapsed_sec,
        model_name=det.model_name,
        rec_model_name=rec_model,
        rec_elapsed_sec=rec_elapsed,
    )
