"""Streamlit demo: classic image preprocess + optional PP-Layout overlay."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classic_enhance import CLASSIC_METHODS, ClassicMethod, get_classic
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
st.caption("원본 → ×3 upscale → CLAHE → Unsharp → mild denoise · 선택: PP-Layout 영역 박스")

with st.sidebar:
    st.header("전처리")
    pp_method: ClassicMethod = st.selectbox(  # type: ignore[assignment]
        "method",
        options=list(CLASSIC_METHODS),
        index=list(CLASSIC_METHODS).index("receipt"),
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
        help="전처리 결과에서 PP-DocLayout으로 영역을 찾고 박스를 그립니다.",
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
    st.info("이미지를 업로드한 뒤 전처리를 실행하세요.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
st.write(f"원본 크기: **{pil.size[0]} × {pil.size[1]}** · `{uploaded.name}`")

run = st.button("전처리 실행", type="primary", use_container_width=True)
if not run:
    st.subheader("원본")
    st.image(pil, use_container_width=True)
    st.stop()

with st.spinner("전처리 중…"):
    try:
        result = get_classic(method=pp_method).predict(pil)
    except Exception as exc:  # noqa: BLE001
        st.error(f"전처리 실패: {exc}")
        st.exception(exc)
        st.stop()

layout_result = None
if draw_layout:
    with st.spinner(f"레이아웃 검출 중… ({layout_model})"):
        try:
            layout_result = detect_layout(
                result.enhanced,
                model_name=layout_model,
                threshold=layout_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"레이아웃 검출 실패: {exc}")
            st.exception(exc)

st.success(
    f"완료 — {result.elapsed_sec * 1000:.0f} ms · "
    f"{result.original.shape[1]}×{result.original.shape[0]} → "
    f"{result.enhanced.shape[1]}×{result.enhanced.shape[0]}"
    + (f" (×{result.scale:g})" if result.scale != 1.0 else "")
    + (
        f" · layout {len(layout_result.boxes)} boxes / "
        f"{layout_result.elapsed_sec * 1000:.0f} ms"
        if layout_result is not None
        else ""
    )
)

cols = st.columns(3 if layout_result is not None else 2)
with cols[0]:
    st.subheader("Before")
    st.image(result.original, use_container_width=True)
    st.caption(f"{result.original.shape[1]}×{result.original.shape[0]}")
with cols[1]:
    st.subheader(f"After ({result.method})")
    st.image(result.enhanced, use_container_width=True)
    st.caption(f"{result.enhanced.shape[1]}×{result.enhanced.shape[0]}")
if layout_result is not None:
    with cols[2]:
        st.subheader(f"Layout ({layout_result.model_name})")
        st.image(layout_result.annotated, use_container_width=True)
        st.caption(f"{len(layout_result.boxes)} regions")

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

dl1, dl2 = st.columns(2)
with dl1:
    buf = BytesIO()
    Image.fromarray(result.enhanced).save(buf, format="PNG")
    st.download_button(
        "전처리 결과 PNG 다운로드",
        data=buf.getvalue(),
        file_name=f"pp_{pp_method}_{Path(uploaded.name).stem}.png",
        mime="image/png",
        use_container_width=True,
    )
if layout_result is not None:
    with dl2:
        layout_buf = BytesIO()
        Image.fromarray(layout_result.annotated).save(layout_buf, format="PNG")
        st.download_button(
            "레이아웃 박스 PNG 다운로드",
            data=layout_buf.getvalue(),
            file_name=f"layout_{layout_model}_{Path(uploaded.name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )
