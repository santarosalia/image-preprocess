# Classic Preprocess Demo

영수증/문서 이미지에 **OpenCV 고전 전처리**를 적용하는 Streamlit 데모입니다.

## 빠른 시작

```bash
uv sync
uv run streamlit run app.py
```

이미지를 업로드한 뒤 사이드바에서 **전처리 적용**과 **영역 검출**을 원하는 대로 켜고 **실행**을 누르면 결과가 표시됩니다. 전처리를 끄면 원본에 바로 검출을 돌립니다.

## 기본 파이프라인 (`receipt`)

```
원본 → ×3 upscale → CLAHE → Unsharp Mask → mild denoise
```

영수증·문서 OCR 전처리에 맞춘 기본 파이프라인입니다. 구현은 `classic_enhance.py`의 `_apply_receipt()`에 있습니다.

### 1. 원본 로드 (RGB)

- 입력: PNG/JPEG 등 컬러 이미지
- RGBA는 알파 채널을 제거하고 RGB로 변환합니다.
- 이후 단계는 **그레이스케일** 기준으로 처리합니다.

### 2. ×3 Upscale

| 항목 | 값 |
|------|-----|
| 배율 | **3배** (`UPSCALE_FACTOR = 3`) |
| 보간 | `cv2.INTER_CUBIC` (3차 보간) |
| 크기 | `(W, H)` → `(W×3, H×3)` |

저해상도 영수증 사진에서 글자 획이 너무 얇거나 뭉개져 있을 때, 먼저 해상도를 키워 후속 대비·선명화 단계가 세부를 다룰 여지를 만듭니다. 바이큐빅 보간은 선형보다 가장자리가 부드럽게 이어져 글자 윤곽 보존에 유리합니다.

### 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)

| 항목 | 값 |
|------|-----|
| `clipLimit` | **2.0** |
| `tileGridSize` | **8×8** |

이미지를 8×8 타일로 나눈 뒤, 타일마다 히스토그램 평활화로 **국소 대비**를 높입니다. `clipLimit`으로 과도한 대비 증폭(노이즈 강조)을 제한합니다. 영수증처럼 조명이 고르지 않거나 흐릿한 구역이 있는 경우, 전역 히스토그램 평활화보다 글자와 배경의 구분이 잘 살아납니다.

### 4. Unsharp Mask (언샤프 마스크)

| 항목 | 값 |
|------|-----|
| `amount` | **1.5** |
| `sigma` | **1.0** |

원리:

```
blurred = GaussianBlur(gray, sigma=1.0)
sharp   = gray × (1 + amount) − blurred × amount
        = gray × 2.5 − blurred × 1.5
```

원본에서 가우시안 블러 이미지를 빼서 **고주파(윤곽·획)** 성분을 강조합니다. CLAHE로 대비를 올린 뒤 획을 또렷하게 만들어 OCR이 글자 경계를 읽기 쉽게 합니다.

### 5. Mild Denoise (약한 노이즈 제거)

| 항목 | 값 |
|------|-----|
| 필터 | `cv2.bilateralFilter` |
| `d` | **5** (필터 직경) |
| `sigmaColor` | **25** |
| `sigmaSpace` | **25** |

양방향 필터로 **색(밝기) 차이가 큰 경계(글자 획)는 유지**하면서, CLAHE·언샤프로 생긴 미세 노이즈만 완화합니다. 강한 가우시안 블러와 달리 획이 뭉개지지 않도록 `sigma`를 낮게 잡은 설정입니다.

### 처리 흐름 요약

```
RGB 원본
  └─ resize ×3 (INTER_CUBIC)
       └─ RGB → Gray
            └─ CLAHE (clip=2.0, tile=8×8)
                 └─ Unsharp Mask (amount=1.5, σ=1.0)
                      └─ Bilateral denoise (d=5, σc=σs=25)
                           └─ Gray → RGB (출력)
```

출력 해상도는 원본의 **3배**입니다. 예: 400×600 → 1200×1800.

### 다른 method

| method | 설명 |
|--------|------|
| `receipt` | 위 파이프라인 (기본) |
| `clahe` | 대비 보정만 (업스케일·언샤프·디노이즈 없음) |
| `unsharp` | 언샤프 마스크만 |
| `adaptive` | adaptive 이진화 (block=31, C=10) |
| `otsu` | Otsu 이진화 |

## 영역 검출·텍스트 인식 (선택)

전처리와 검출은 각각 켜고 끌 수 있습니다. 검출은 두 단위입니다.

| 검출 단위 | 기본 모델 | 잡는 것 |
|-----------|-----------|---------|
| 레이아웃 영역 | `PP-DocLayoutV3` | 제목·본문·표 같은 큰 덩어리. 영수증 2장이면 보통 박스 2개 |
| 텍스트 라인 | `PP-OCRv6_medium_det` | 줄·필드 단위. 카드번호, 총액 같은 행 |

레이아웃은 [PP-DocLayout](https://www.paddleocr.ai/latest/en/version3.x/module_usage/layout_detection.html)으로 text / title / table / figure 등을 찾습니다. 다른 선택지는 `PP-DocLayoutV2`, `PP-DocLayout_plus-L`, `PP-DocLayout-L`, `PP-DocLayout-M`, `PP-DocLayout-S`입니다. CPU에서 더 빠르게 돌리려면 `S` / `M`을 고르세요.

텍스트 라인의 다른 선택지는 `PP-OCRv5_server_det`입니다. **텍스트 인식 (rec)** 을 켜면 박스마다 글자를 읽습니다. 기본은 `korean_PP-OCRv5_mobile_rec`이고, `en_PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`도 고를 수 있습니다.

1. (한 번) `uv sync --extra ocr`
2. 사이드바에서 **전처리 적용**, **영역 검출**을 원하는 대로 켭니다.
3. 검출 단위에서 **레이아웃 영역** 또는 **텍스트 라인**을 고릅니다.
4. 텍스트 라인을 켠 뒤 **텍스트 인식 (rec)** 을 켜면 박스 글자가 테이블에 나옵니다.
5. **실행** → 원본 / After(전처리 켠 경우) / OCR(검출 켠 경우)이 표시됩니다.

첫 실행 때 모델 가중치를 다운로드합니다.

구현: `layout_detect.py`

- 레이아웃: `detect_layout`, `parse_layout_boxes`, `draw_layout_boxes`
- 텍스트 라인: `detect_text_lines`, `parse_text_boxes`
- 인식: `detect_and_recognize`, `recognize_boxes`

## 구성

| 경로 | 설명 |
|------|------|
| `pyproject.toml` | uv 의존성. 검출·인식은 `ocr` extra |
| `uv.lock` | 잠긴 의존성 |
| `app.py` | Streamlit UI. 전처리·검출·인식을 각각 켜고 끕니다 |
| `classic_enhance.py` | OpenCV 전처리. `working_image()`가 검출에 넣을 이미지를 고릅니다 |
| `layout_detect.py` | PP-DocLayout 영역 검출, `PP-OCRv6_medium_det` 텍스트 라인, rec |
