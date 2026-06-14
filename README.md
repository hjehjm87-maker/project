# project
AI project 4조
# AI-project 

## 1. 프로젝트 개요

### 주제 
실제 블랙박스 환경을 위한 YOLOv11 기반 도로 위험물 탐지 고도화 및 데이터 환경 차이에 따른 전처리 한계 분석

### 문제 정의 및 개발 동기
* **실전 도로의 치명적 위험성:** 포트홀, 균열, 맨홀 결함은 차량 파손 및 주행 중 대형 2차 교통사고를 유발하는 핵심 위험 요인입니다.
* **기존 공개 데이터셋의 한계:** 기존 연구 데이터셋은 대부분 정지 상태의 고화질(A급) 이미지 위주로 구성되어 있어, 진동·속도·조도 변화가 심한 실제 블랙박스 주행 환경 적용 시 탐지 성능이 급격히 저하되는 한계가 있습니다.
* **실전 동적 데이터 검증의 필요성:** 본 연구는 이러한 도메인 격차를 해소하기 위해, 실제 차량 블랙박스와 유사한 동적 주행 데이터 환경에서 도로 위험물의 탐지 및 실시간 추적 타당성을 실증적으로 검증합니다.

### 연구 목적 및 개발 목적
* **도메인 격차에 따른 전처리 한계 실증:** 실제 블랙박스의 동적 주행 환경과 기존 연구용 정적 데이터셋 간의 격차를 규명하고, 일반적인 전처리 기법의 탐지 성능 한계를 실험적으로 입증합니다.
* **전처리 및 증강 기법의 영향도 분석:** 성능 개선(`mAP50`)을 위해 도입한 기술들(`ROI Crop`, `CLAHE`, `Augmentation`)이 실제 도로 위험물 탐지에서 성능 저하를 유발한 원인을 정량적·정성적으로 분석합니다.
* **실전형 데이터 정제 방향성 제안:** 맥락 정보 손실 및 노이즈 증폭 원인 분석을 바탕으로, 이미지 품질 향상을 넘어 **실제 주행 환경 특성을 반영한 최적의 데이터 가이드라인**을 도출하고 향후 연구 방향을 제안합니다.

### 프로젝트 파이프라인 아키텍처
<img width="509" height="284" alt="image" src="https://github.com/user-attachments/assets/ba63c12d-b3c9-43db-afba-7360ae88c167" />


## 2. 필요한 라이브러리 및 환경
본 프로젝트는 구글 코랩(Google Colab - Tesla T4 GPU) 및 로컬 가속 컴퓨팅 환경에서 `Ultralytics` 프레임워크를 기반으로 파이프라인을 구축하고 전처리 성능 실험을 수행하였습니다.

* **핵심 프레임워크 (Core Framework):** `Ultralytics YOLOv11`
* **의존성 라이브러리 (Dependencies):** * `Torch` (딥러닝 모델 연산 및 GPU 가속)
  * `OpenCV-Python` (ROI Crop 및 CLAHE 국소 명암 대비 알고리즘 구현)
  * `tqdm` (데이터 전처리 및 루프 대량 연산 진행률 모니터링)
  * `PyYAML`, `Matplotlib`, `os`, `glob`

### 개발 환경 구축 및 의존성 설치
구글 코랩 환경 필수 패키지 및 라이브러리 일괄 설치
pip install ultralytics python-dotenv opencv-python

## 3.데이터 세트
데이터셋 출처 및 기본 정보
* **공식 명칭:** Road Damage Dataset: Potholes, Cracks and Manholes
* **제공 기관:** Kaggle (글로벌 오픈소스 데이터 플랫폼)
* **데이터 규모:** 총 2,009장의 도로 노면 데이터 및 결함 라벨링 데이터

데이터 가공 및 정밀 분할 구조
* **데이터셋 가공:** 원본 도로 전경 데이터에서 탐지 모델의 효율성을 극대화하기 위해 파손 의심 영역을 정밀하게 크롭(Crop) 및 라벨링하여 데이터 완성도를 고도화했습니다.
* **정답 유출 방지 (Data Leakage Filter):** 학습을 진행하기 전, 검증 데이터셋(Validation Dataset)에 포함된 시험 문제가 학습 데이터셋(Train Dataset)에 중복 포함되어 모델이 답안을 외우는 현상을 방지하기 위해, 중복 데이터를 전수 추적하여 필터링 및 제거하는 고도화 공정을 거쳤습니다.
* **데이터셋 분할 비율:** 원본 데이터는 **Train(80%) : Validation(20%)**의 독립적인 평가 구조로 분할되었습니다.
  * **Train Dataset (학습 데이터):** 1,607장 (이미지 및 라벨 텍스트 매칭 완료)
  * **Validation Dataset (검증 데이터):** 402장
 

### 수립된 데이터셋 로컬 디렉토리 구조

```text
C:/Users/konyang/dataset_yolo/
├── data.yaml                  # 클래스 수(nc: 3) 및 경로 정의 파일
├── train/
│   ├── images/                # 학습용 도로 노면 이미지 (1,607장)
│   └── labels/                # 정밀 크롭 및 라벨링 텍스트 (1,607개 .txt)
└── val/
    ├── images/                # 검증용 도로 노면 이미지 (402장)
    └── labels/                # 검증용 라벨링 텍스트 (402개 .txt)
```

## 4. 모델 설명 및 개발 내용
실제 블랙박스 주행 데이터 환경에 대응하는 최적의 파이프라인을 도출하기 위해, 본 프로젝트에서는 `YOLOv11` 구조를 기반으로 모델 체급 확장 및 다양한 영상 처리 가공 기법을 조합한 총 5가지 실험 트랙을 설계하여 한계 분석을 수행하였습니다.

실험별 전처리 및 학습 파이프라인

* 1차 실험 Baseline 학습 (YOLOv11s)
기준 탐지 성능 지표를 수립하기 위해, 추가적인 화질 가공 없이 정제된 기본 데이터셋만을 활용하여 Small 모델로 가볍게 학습을 진행했습니다.

```python
from ultralytics import YOLO

# Baseline: YOLOv11s 가중치 기반 기본 학습 파이프라인 구축
model_base = YOLO("yolo11s.pt")
model_base.train(
    data="data.yaml", 
    epochs=50, 
    batch=16, 
    imgsz=640, 
    name="yolo11s_baseline"
)
```

* 2차 실험 Model Upgrade (YOLOv11m)
모델의 백본(Backbone) 네트워크 파라미터 체급을 확장하여, 도로 위험물의 미세 특징 추출 성능을 고도화하고자 Medium 모델로 스케일업 실험을 진행했습니다.

# Model Upgrade: YOLOv11m 모델 스케일업 학습
```python
model_mid = YOLO("yolo11m.pt")
model_mid.train(
    data="data.yaml", 
    epochs=50, 
    batch=16, 
    imgsz=640, 
    name="yolo11m_upgrade"
)
```
* 3차 실험: Data Augmentation 적
차량 주행 중 발생할 수 있는 노면의 물리적 각도 변화 및 기상 조건 변화를 강제로 모사하기 위해, 광학적·기하학적 증강 옵션을 부여하여 모델의 일반화 성능을 유도했습니다.

# Augmentation: 조도 변화 및 흔들림 가혹 환경 모사 학습
```python
model_aug = YOLO("yolo11s.pt")
model_aug.train(
    data="data.yaml", 
    epochs=50, 
    batch=16, 
    imgsz=640, 
    hsv_v=0.6,       # 명도 변형 확장
    hsv_s=0.5,       # 채도 변형 확장
    degrees=10.0,    # 미세 회전 강제
    translate=0.2,   # 평행 이동 변형
    name="yolo11s_augmentation"
)
```
* 4차 실험: ROI Crop 전처리 처리
도로 전경 이미지에서 하늘, 가로수, 건물 등 탐지와 무관한 불필요한 상단 배경을 파이썬 스크립트로 필터링하고, 실제 결함이 존재하는 노면(Region of Interest) 영역만 정밀하게 잘라내어(Crop) 데이터셋을 정제했습니다.

# ROI Crop 전처리가 반영된 데이터셋으로 학습 연동
```python
model_crop = YOLO("yolo11s.pt")
model_crop.train(
    data="crop_data.yaml",  # 크롭 전용 데이터 명세
    epochs=50, 
    batch=16, 
    imgsz=640, 
    name="yolo11s_roi_crop"
)
```

5차 실험: CLAHE 국소 명암 대비 강조 적용
아스팔트 노면과 어두운 균열(Crack) 경계선의 식별력을 극대화하기 위해, OpenCV를 활용한 국소 적응형 히스토그램 균등화(CLAHE) 알고리즘을 이미지 전반에 적용하여 대비를 극대화했습니다.

```python
import cv2
# OpenCV 활용 CLAHE 이미지 가공 파이프라인 구현 샘플
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# (노트북 소스코드를 통해 전체 이미지를 변환 처리한 후 YOLOv11 학습 수행)
```

비디오 인퍼런스 및 실시간 추적(Tracking) 알고리즘

```python
from ultralytics import YOLO

# ByteTrack 알고리즘 연동 및 실제 도로 주행 비디오 인퍼런스 추적
best_model = YOLO("C:/Users/konyang/runs/detect/weights/best.pt")
results = best_model.track(
    source="C:/Users/konyang/dataset_yolo/맑은날.mp4", 
    show=True, 
    tracker="bytetrack.yaml"
)
```

## 5. 실험 결과 및 분석 (Results & Analysis)
본 프로젝트는 실제 도로 환경과 유사한 조건에서 각 전처리 및 학습 기법이 미치는 영향력을 평가하기 위해, 검증 데이터셋(Validation Dataset)을 기준으로 정량적 성능 메트릭을 도출하여 비교 분석을 수행하였습니다.

### 모델 및 전처리 기법별 최종 정량 성능 비교 테이블

| 실험 구분 | 적용된 핵심 기법 (Methods) | Epochs | mAP50 (Val) | 주요 원인 분석 및 결과 요약 |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | **YOLOv11s 기본 모델** | 50 | **0.557** | 해외 데이터셋 기준 표준 탐지 성능 지표 확립 |
| **Model Upgrade**| **YOLOv11m 체급 확장** | 50 | **0.557** | 모델 파라미터를 확장했으나 Baseline 수치와 동일하게 수렴 |
| **Augmentation** | **데이터 증강 기법 적용** | 50 | **0.543** | 인위적인 조도/기하학 변형이 실제 데이터에선 노이즈로 작용 |
| **Processing (1)**| **ROI Crop 전처리** | 50 | **0.535** | 상단 배경 제거 시 도로의 원근감 및 주변 맥락 정보가 상실됨 |
| **Processing (2)**| **CLAHE 명암 대비 강조** | 50 | **0.477** | 아스팔트 특유의 거친 질감이 노이즈로 과잉 강화되어 오탐지 급증 |

### 핵심 분석 결과 및 인사이트 (Core Insights)

* **전처리 기법의 도메인 한계:** 이미지 개선을 위해 도입한 `ROI Crop`과 `CLAHE` 기법이 Baseline보다 성능을 저하시키는 역효과(Trade-off)를 확인했습니다.
* **도로 탐지 내 '맥락 정보'의 중요성:** 도로 위험물 탐지에는 객체 자체의 특징뿐만 아니라 **원근감, 소실점, 주변 노면과의 연속성(Context)**이 중요합니다. ROI Crop은 이 맥락 정보를 단절시켜 오탐지를 유발했습니다.
* **노이즈 증폭 현상:** `CLAHE` 기법은 아스팔트 고유의 거친 질감마저 균열이나 포트홀의 경계선 노이즈로 증폭시켜 모델의 혼선과 오탐지를 급격히 증가시켰습니다.

### 6. 한계점 및 향후 계획 (Limitations & Future Work)

### 연구의 핵심 한계점 및 검증 전략

* **클래스 인덱스 불일치 및 정성적 우회 검증:** 학습 데이터셋과 실전 테스트 데이터 간의 `data.yaml` 내 클래스 순서(Class Index Mapping) 불일치 오류가 발생했습니다. 포맷 차이로 정량적 수치(Test mAP) 도출이 어려워짐에 따라, **실제 주행 영상(`맑은날.mp4`)에서 바운딩 박스가 노면에 정확히 매핑·추적되는지 육안으로 확인하는 '정성적 검증'으로 실효성을 입증했습니다.
* **기상 및 주행 도메인의 편향성:** 현재 검증 파이프라인이 '맑은 날' 영상 데이터에만 편향되어 있습니다. 흐린 날, 우천 시, 야간 등 조도가 낮거나 노면 반사가 심한 가혹 주행 환경에서의 신뢰성을 확보하기 위해 향후 추가 데이터 확보가 필수적입니다.

### 향후 연구 계획 (Future Work)

* **클래스 인덱스 표준화 모듈 개발:** 외부 데이터셋 결합 시 발생하는 클래스 넘버링 충돌을 차단하기 위해, 인덱스를 자동으로 매핑·동기화하는 표준화 전처리 파이프라인을 구축할 예정입니다.
* **맥락 보존 전처리 및 엣지 가속화 연구:** 배경을 무조건 잘라내는 방식 대신 가중치 마스킹을 통해 도로의 원근감을 보존하는 전처리를 연구할 계획입니다. 또한, 블랙박스 환경에서 실시간 추론이 가능하도록 모델을 **NVIDIA TensorRT** 포맷으로 변환하고 **양자화(Quantization, INT8/FP16)** 처리를 적용하는 경량화를 추진할 예정입니다.
