"""Streamlit demo: optional classic preprocess + optional PP-Layout overlay."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classic_enhance import CLASSIC_METHODS, ClassicMethod, working_image
from layout_detect import (
    DEFAULT_LAYOUT_MODEL,
    LAYOUT_MODELS,
    detect_layout,
    layout_ready,
)

st.set_page_config(
    page_title="Classic Preprocess",
    page_icon="🧾",
    layout="wide",
)

st.title("Classic Preprocess")
st.caption("전처리·PP-Layout을 각각 켜고 끌 수 있습니다")

with st.sidebar:
    st.header("전처리")
    apply_preprocess = st.checkbox(
        "전처리 적용",
        value=True,
        help="끄면 원본에 바로 레이아웃 검출을 돌립니다.",
    )
    pp_method: ClassicMethod = st.selectbox(  # type: ignore[assignment]
        "method",
        options=list(CLASSIC_METHODS),
        index=list(CLASSIC_METHODS).index("receipt"),
        disabled=not apply_preprocess,
        help="receipt=×3→CLAHE→Unsharp→mild denoise (기본)",
    )
    st.divider()
    st.header("레이아웃")
    if not layout_ready():
        st.warning("paddleocr 미설치 — `pip install paddlepaddle paddleocr`")
    draw_layout = st.checkbox(
        "PP-Layout으로 영역 검출",
        value=False,
        disabled=not layout_ready(),
        help="전처리가 켜져 있으면 전처리 결과, 꺼져 있으면 원본에서 영역을 찾습니다.",
    )
    layout_model = st.selectbox(
        "model",
        options=list(LAYOUT_MODELS),
        index=list(LAYOUT_MODELS).index(DEFAULT_LAYOUT_MODEL),
        disabled=not draw_layout,
        help="기본은 PP-DocLayoutV3. S/M은 가볍고, L/V2도 선택 가능합니다.",
    )
    layout_threshold = st.slider(
        "threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        disabled=not draw_layout,
    )
    st.divider()
    st.markdown("```bash\npip install -r requirements.txt\nstreamlit run app.py\n```")

uploaded = st.file_uploader(
    "이미지 업로드",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("이미지를 업로드한 뒤 실행하세요.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
st.write(f"원본 크기: **{pil.size[0]} × {pil.size[1]}** · `{uploaded.name}`")

if not apply_preprocess and not draw_layout:
    st.warning("전처리 또는 레이아웃 검출 중 하나 이상을 켜세요.")

run = st.button("실행", type="primary", use_container_width=True)
if not run:
    st.subheader("원본")
    st.image(pil, use_container_width=True)
    st.stop()

if not apply_preprocess and not draw_layout:
    st.stop()

pp_result = None
with st.spinner("전처리 중…" if apply_preprocess else "이미지 준비 중…"):
    try:
        working, pp_result = working_image(
            pil,
            apply_preprocess=apply_preprocess,
            method=pp_method,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"전처리 실패: {exc}")
        st.exception(exc)
        st.stop()

layout_result = None
if draw_layout:
    with st.spinner(f"레이아웃 검출 중… ({layout_model})"):
        try:
            layout_result = detect_layout(
                working,
                model_name=layout_model,
                threshold=layout_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"레이아웃 검출 실패: {exc}")
            st.exception(exc)

status = []
if pp_result is not None:
    status.append(
        f"preprocess {pp_result.elapsed_sec * 1000:.0f} ms · "
        f"{pp_result.original.shape[1]}×{pp_result.original.shape[0]} → "
        f"{pp_result.enhanced.shape[1]}×{pp_result.enhanced.shape[0]}"
        + (f" (×{pp_result.scale:g})" if pp_result.scale != 1.0 else "")
    )
else:
    status.append(f"원본 {working.shape[1]}×{working.shape[0]}")
if layout_result is not None:
    status.append(
        f"layout {len(layout_result.boxes)} boxes / "
        f"{layout_result.elapsed_sec * 1000:.0f} ms"
    )
st.success("완료 — " + " · ".join(status))

panels: list[tuple[str, object, str]] = [("원본", pil, f"{pil.size[0]}×{pil.size[1]}")]
if pp_result is not None:
    panels.append(
        (
            f"After ({pp_result.method})",
            pp_result.enhanced,
            f"{pp_result.enhanced.shape[1]}×{pp_result.enhanced.shape[0]}",
        )
    )
if layout_result is not None:
    panels.append(
        (
            f"Layout ({layout_result.model_name})",
            layout_result.annotated,
            f"{len(layout_result.boxes)} regions",
        )
    )

cols = st.columns(len(panels))
for col, (title, image, caption) in zip(cols, panels, strict=True):
    with col:
        st.subheader(title)
        st.image(image, use_container_width=True)
        st.caption(caption)

if layout_result is not None:
    st.subheader("검출 영역")
    st.dataframe(
        [
            {
                "label": box.label,
                "score": round(box.score, 3),
                "x1": round(box.coordinate[0], 1),
                "y1": round(box.coordinate[1], 1),
                "x2": round(box.coordinate[2], 1),
                "y2": round(box.coordinate[3], 1),
            }
            for box in layout_result.boxes
        ],
        use_container_width=True,
        hide_index=True,
    )

dl_cols = st.columns(2)
if pp_result is not None:
    with dl_cols[0]:
        buf = BytesIO()
        Image.fromarray(pp_result.enhanced).save(buf, format="PNG")
        st.download_button(
            "전처리 결과 PNG 다운로드",
            data=buf.getvalue(),
            file_name=f"pp_{pp_method}_{Path(uploaded.name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )
if layout_result is not None:
    with dl_cols[1 if pp_result is not None else 0]:
        layout_buf = BytesIO()
        Image.fromarray(layout_result.annotated).save(layout_buf, format="PNG")
        st.download_button(
            "레이아웃 박스 PNG 다운로드",
            data=layout_buf.getvalue(),
            file_name=f"layout_{layout_model}_{Path(uploaded.name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )
