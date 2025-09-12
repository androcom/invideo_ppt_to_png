# InVideo PPT to PNG

강의 영상(.mp4 등)에서 발표 자료(PPT) 화면이 변경될 때마다 해당 프레임을 자동으로 감지하여 이미지(.png) 파일로 추출하고, 유사한 슬라이드끼리 그룹화해주는 파이썬 스크립트입니다.


---

## ✨ 주요 기능

* **자동 슬라이드 감지**: 영상의 프레임을 비교하여 PPT 화면이 전환되는 시점을 자동으로 감지합니다.
* **정교한 비교 알고리즘**: 2단계 비교 방식을 통해 정확도를 높였습니다.
    1.  **Pixel Difference**: 픽셀 단위의 차이를 빠르게 계산하여 명확한 화면 전환을 1차로 감지합니다.
    2.  **SSIM (Structural Similarity)**: 1차에서 감지되지 않은 미세한 텍스트 변경 등은 구조적 유사성을 통해 2차로 정교하게 감지합니다.
* **유사 슬라이드 그룹화**: 추출된 슬라이드들을 대상으로 유사도를 다시 분석하여, 내용이 비슷한 이미지들끼리 `Group_X` 폴더로 자동 분류합니다.
* **상세 로그 기록**: 각 프레임의 비교 결과(유사도 점수, 저장 여부 등)를 `.log` 파일로 저장하여, 임계값 튜닝 및 디버깅이 용이합니다.
* **직관적인 설정**: `config.yaml` 파일을 통해 영상 폴더 경로, 비교 간격(초 단위), 민감도(임계값) 등을 쉽게 설정할 수 있습니다.

---

## ⚙️ 사용 환경

* Python 3.8 이상
* Windows 10/11 (한글 경로 지원)

---

## 🚀 빠른 시작

1.  **[Releases 페이지](https://github.com/androcom/invideo_ppt_to_png/releases)**에서 최신 버전의 `InVideo_PPT_to_PNG.zip` 파일을 다운로드합니다.
2.  원하는 위치에 압축을 해제합니다.
3.  `INPUT` 폴더 안에 분석하고 싶은 동영상 파일들을 넣습니다.
4.  **`#run.bat`** 파일을 더블클릭하여 실행합니다.
5.  프로그램이 자동으로 필요한 라이브러리를 설치하고 분석을 시작합니다.
6.  완료되면 `EXTRACTED_SLIDES` 폴더에 결과물이 생성됩니다.

---

## 🔧 상세 설정 (`config.yaml`)

`config.yaml` 파일을 수정하여 프로그램의 동작을 세부적으로 제어할 수 있습니다.

```yaml
# config.yaml

# --- 기본 설정 ---
# 영상 폴더 경로
input_folder_path: "INPUT"
# 저장 최상위 폴더 경로
master_output_folder: "EXTRACTED_SLIDES"
# 영상 파일 확장자 (리스트 형식)
video_extensions:
  - .mp4
  - .mov
  - .avi
  - .mkv

# --- 처리 옵션 ---
# 프레임 비교 간격 (초)
frame_interval_sec: 1

# --- 각 방식별 임계값 --- (0에 가까울수록 유사)
# 픽셀 차이 임계값
pixel_diff_threshold: 0.02
# SSIM 차이 임계값
ssim_diff_threshold: 0.01
```

---

## 📦 프로젝트 구조

```
.
├── INPUT/                    # 분석할 영상들을 넣는 폴더
├── EXTRACTED_SLIDES/         # 결과물이 저장되는 폴더 (자동 생성)
├── main.py                   # 메인 실행 로직
├── video_processor.py        # 영상 처리 및 슬라이드 추출 기능
├── post_processor.py         # 추출된 이미지 그룹화 기능
├── comparisons.py            # 이미지 비교 알고리즘 (Pixel Diff, SSIM)
├── config.yaml               # 설정 파일
├── requirements.txt          # 필요 라이브러리 목록
└── #run.bat                  # 윈도우용 간편 실행 파일
```
