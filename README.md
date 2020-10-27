# 20f-creative-integrated-design-1
창의적통합설계1 (2020년도, 2학기, M1522.000200_001)

## 몰라페우스 w. Morpheus3D [Homepage](https://morpheus3d.co.kr/wp/en/)
### 팀 구성
* **이진호**, **오아영**, **손상준**
### 프로젝트 기획
* 프로젝트 목표: **Deep learning 기반 얼굴 주름 감지 기술 개발**
  1. Data Annotation 주름영역 masking 및 데이터 labelling 
      - 단, data annotation의 경우 시간이 많이 걸리기 때문에 본사에서 작업된 결과를 함께 활용함.
  2. Image Processing Edge detection, Laplacian filter 등 방법 적용
  3. Deep Learning Model Development 선행 연구 조사 및 딥러닝 모델 train/validate
* **Difficulties**
  1. Domain의 특수성
      * 일반적인 사진이 아니라 3차원 영역의 색을 2차원에 매핑한 texture 이미지를 사용
      * 기존의 Deep learning 기반 segmentation 기법들은 이미지에서 target이 차지하는 영역이 대부분
      * 3D 스캔된 얼굴 사진 중 주름의 차지 비중이 현저히 낮음
      * Loss function 및 Metric 설계의 어려움
  2. Shortage of Data
      * 기존에 제공하려던 이미지가 사용자의 보험 및 초상권 문제로 회사 외부 공유 불가
      * 예상 확보량보다 적은 200-300장의 data로 학습을 진행해야 하는 상황
* **Applications**
  1. 피부과에서 환자에게 치료과정을 보다 쉽고 정확하게 안내 가능 
  2. 피부과에서 상담 시에 객관적인 자료로 치료 유도 가능
  3. 피부과에서 시각적인 자료를 이용하여 환자에게 치료 효과 설명 가능
  4. 노화 예측 가능 
* **Method Comparison**
  1. Proposed Method
  > 딥러닝 + 이미지 프로세싱 기법들에 기반한 알고리즘
  2. Existing Methods
  > 기존 회사 내부 모듈 (w. & w.o. Control parameter tuning) <br>
  > 이미지 프로세싱 기법들을 여러가지 종합한 알고리즘 
### Background Research
* Fully Convolutional Neural Network
* U-Net: Convolutional Networks for Biomedical Image Segmentation
    * [Homepage](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
    * https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
