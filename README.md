# 1st-Generation-Pokemon-Classifier
**본 프로젝트는 사전 학습된(Pre-trained) 다양한 CNN 모델들을 활용하여 150종의 포켓몬을 분류하는 전송 학습(Transfer Learning) 실험입니다. 2012년의 AlexNet부터 2019년의 EfficientNet까지 총 5가지의 Backbone 네트워크를 비교 분석하여, 모델의 구조적 진화가 실제 분류 성능에 미치는 영향을 확인하였습니다.**


### 데이터셋
**Data: PokemonData**
**클래스 수: 150**

### Data Split

Training Set: 80% 

Validation Set: 20%

Input Size: 224 x 224 (ImageNet 표준 규격) 

### 하이퍼파라미터 설정
Batch Size : 32

Learning Rate : 0.001

Epochs : 5

Optimizer : Adam

Loss Functio : CrossEntropyLoss

###  분류에 사용할 모델

***AlexNet (2012):  The first CNN-based winner in ImageNet classification 2012***

***GoogLeNet (2014) : The 1st place in ImageNet classification 2014, “Deeper network with computational efficiency”***

***ResNet-18  (2015) :  “Very deep network with residual connections”***

***ResNet-34 (2015) : Stable learning in deep layers through Skip Connection***

***EfficientNet-B0 (2019) : Automation of finding optimal network architectures***


### 결과및 분석

| Model Backbone | Final Val Acc | Training Time | Analysis |
| :--- | :---: | :---: | :--- |
| **AlexNet** | 1.03% | ~2m 20s |BN(Batch Normalization) 부재 및 거친 필터 크기로 인해 복잡한 포켓몬 이미지의 특징을 전혀 학습하지 못함. |
| **GoogLeNet** | 85.56% | ~3m 03s | Inception 모듈의 효율성 덕분에 상대적으로 적은 파라미터로도 매우 우수한 정확도를 기록함. |
| **ResNet-18** | 83.80% | ~2m 48s | Residual Connection을 통해 안정적인 학습과 빠른 수렴 속도를 보여줌.|
| **ResNet-34** | 83.28% | ~3m 15s | 18 레이어보다 깊은 구조를 가졌으나, 본 데이터셋 규모에서는 성능 향상이 크지 않음|
| **EfficientNet-B0** | **90.84%** | ~3m 12s |Compound Scaling 기법을 통해 가장 정교하고 높은 신뢰도의 분류를 수행함.|

### 학습성능 시각화
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/6b88373a-905b-4947-9d3d-38b0cf65e2c0" />


---

## 결론

*   **모델 구조의 진화**: 2012년 모델인 **AlexNet**과 최신 모델들 간의 압도적인 성능 차이를 확인했습니다 이를 통해 배치 정규화, 잔차 연결 등 현대적인 기술이 실제 분류 정확도와 학습 안정성에 결정적인 영향을 미친다는 것을 확인하였습니다.
*   **Transfer Learning**: ImageNet으로 사전 학습된 가중치를 활용함으로써, 단 5 Epoch의 짧은 학습만으로도 150종이라는 복잡한 클래스 분류 문제를 90% 이상의 정확도로 해결할 수 있었습니다.
*   **최적의 모델**: 본 포켓몬 데이터셋 분류에서는 **EfficientNet-B0**가 정확도와 효율성 측면에서 가장 우수한 성능를 보여주었습니다.

---

## GUI (Streamlit)

### 주요 기능
1.  **모델 선택**: 사이드바를 통해 학습된 5가지 모델을 자유롭게 선택 가능
2.  **이미지 업로드**: JPG, PNG 형식의 포켓몬 이미지 업로드 및 미리보기 지원
3.  **예측 결과 시각화**: 
    *   **Top-5 예측**: 모델이 가장 가능성 높게 판단한 상위 5개 포켓몬 이름 출력
    *   **확률표시**: 각 예측 결과의 확률(%)을 시각화

##  실행 방법

### 사전 요구 사항
*   **Python 버전**: 3.9 이상 권장
*   **필수 라이브러리**: `torch`, `torchvision`, `streamlit`, `matplotlib`, `pillow`
*   **데이터 경로**: 프로젝트 루트 폴더에 `PokemonData` 폴더가 위치해야 합니다.

###  실행
1.  **모델 학습**
    *   `data.ipynb` 파일을 실행합니다.
    *   실행이 완료되면 폴더 내에 5개의 모델 가중치 파일(`pokemon_*.pth`)이 생성됩니다.
2.  **GUI 실행**:
    *   터미널(CMD)에서 아래 명령어를 입력합니다.
    ```bash
    streamlit run gui.py
