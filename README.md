Title: YOLOv7을 이용한 수어 번역기
==========
Members
===========
* 송서빈(융합전자공학부/2020059189) skmssb@hanyang.ac.kr
  
: 자료 조사 및 정리, 코드 작성 및 실행, 발표

* 이예림(연극영화과/2018034057) sdyerim@naver.com
  
: 참고 자료 조사및 검, 코드 작성 및 검토

발표 영상
==========
https://youtu.be/OJhZf3OWHuY


Results Video
==========
https://youtu.be/Mds_EWV29I0?si=EFwF-B9OAF_5ZHXp


I. Proposal (Option 1)
=============================
### Motivation: Why are you doing this?

소통을 하고자 하는 것은 인간의 기본적인 욕구다. 음성 언어로 소통하지 못하는 특수한 사람들의 상황을 반영하여, 소통을 가능하게 하려는 동기에서 프로젝트를 시작하였다.
평범한 이들이 소통하는 방식 이외의 방법으로 소통할 수 있는 가능성을 보여줌으로써, 평등한 인류를 도모하고, 소통을 위한 각자의 방식이 있을 수 있음을 역설하고자 한다.

### What do you want to see at the end?

웹캠에서 손동작을 입력하여 이 손을 yolov7을 통해 object detection 한 후, 손의 움직임을 인식하는 것이다.
  
II. Datasets
======================================

### 1) Preparing Dataset

* #### About ASL

https://public.roboflow.com/object-detection/american-sign-language-letters/1


<img src="https://github.com/hoootteok2/aix_project/assets/168548944/4dec0d96-3973-4cf9-8138-ec9c630ce13b" width="50%">


ASL이란 American Sign Language로, 미국과 캐나다에 살고 있는 농인들을 위한 수어이다. 
우리는 이 ASL image datasets을 Roboflow에서 제공하는 것으로 사용했다.
Robflow에서 제공하는 데이터셋들은 이미지 주석, 데이터 전처리, 증강 및 다양한 지원들을 제공하는 사이트이므로 본 프로젝트에서 유연하게 데이터셋을 이용할 수 있다.
아래 이미지는 해당 dataset의 예시 이미지이다.

<img width="572" alt="dataset예시" src="https://github.com/hoootteok2/aix_project/assets/168548944/359d2d10-8288-4392-9f80-96a8896c70c3" width="50%">

아래 코드는 구글colab을 이용하여 데이터셋을 준비하는 과정이다.

```
# dataset download from ROBOFLOW
!curl -L "https://public.roboflow.com/ds/DGExMteSHE?key=XRBFhckXHE" -o roboflow.zip
!unzip roboflow.zip -d ./dataset
!rm roboflow.zip
```

* #### Datasets Overview

이 데이터셋은 총 1728장의 이미지이며 아래 이미지와 같이 Train/Test Split은 Train 1512, Valid 144, Test 72로 구성되어 있다.

<img src="https://github.com/hoootteok2/aix_project/assets/168548944/356e20d7-9cf3-4005-bb98-07d03024acb9" width="50%">



또한 dataset의 data.yaml 구성은 classes 총 26개, 알파벳의 개수로 하였다.

```
train: datasets/asl/train/images
val: datasets/asl/valid/images
test: datasets/asl/test/images

nc: 26
names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
```

아래 막대 그래프는 train datasets 1512장의 각 class 마다 존재하는 이미지들의 수이다.
<img width="617" alt="trainset" src="https://github.com/hoootteok2/aix_project/assets/168548944/99141172-fc1c-4865-9446-f6ffd26a1a73" width="30%">



III. Methodology
========================
### model & approving weights methods

### 1) YOLOv7

<img width="551" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/84a35333-b71c-4ffc-b9d3-ef74be2ee82f">

YOLO(You Only Look Once)는 딥러닝 실시간 객체 탐지 모델로 이미지를 처리하여 객체를 탐지하고 분류하는 모델이다. 이 모델의 주요 특징은 실시간 객체 탐지가 가능하며 정밀도가 높은 점이 있다.

YOLOv7의 model architecture를 간단히 설명하면 다음과 같다.

- input layer 이미지 입력
- Backbone Network 특징 추출하는 신경망으로 이미지의 특징을 추출
- Neck Network에서 다양한 스케일의 특징들을 결합하여 객체의 크기와 위치를 파악
- Head Network에서 실제로 객체를 탐지하고 분류, 예측
- Output Layer에서 최종 탐지된 결과를 출력

이와 같은 특징은 아래 그래프아 같이 YOLOv7이 다른 모델들에 비해 높은 정확도와 속도를 보여줄 수 있게 되었다.

![image](https://github.com/hoootteok2/aix_project/assets/168548944/7b091b40-57f3-4a47-8869-6f9f4d94d509)

특히, YOLOv7은 이전 YOLO 버전들과 다른 특이 기술적 개선 사항들이 존재하는데 그 중에 가장 대표적으로 ELAN이 있다.

<img width="615" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/fcd0768c-c406-4ed0-8306-25475d6499df">

ELAN(Efficient Layer Aggregation Networks)는 특징 추출 및 집계에 효율적인 기술이다. ELAN은 위 그림과 같이 3 * 3 과 1 * 1 컨볼루션 레이어를 사용하는데 이 블록 내부의 여려 경로들을 통해 특징을 동시에 처리할 수 있어서 연산 효율성을 높일 수 있게 된다. 또한 CSP(Cross Stage Partial Connections)을 사용하여 일부 레이어의 연결을 분리하는데 이는 계산 비용을 줄이는 동시에 메모리 사용량을 최적화하고 학습 속도를 개선한다.


### 2) hyper parameters tuning

딥러닝 모델의 성능을 최적화하기 위해 하이퍼파라미터를 조절하는 과정을 겪는다. 이 값들은 사용자가 직접 수정해야 하는 수치인 것이 특징이다. 대표적인 하이퍼파라미터들은 다음과 같다.

- Learning Rate : 가중치 업데이트 하는 속도
  
- Batch Size : 한 번에 학습하는 데이터 샘플의 수

- Epochs : 전체 데이터셋 학습 반복 횟수

우리가 진행할 프로젝트에서는 Learning Rate를 유지하며 Batch size를 증가시키는 방법으로 선택할 것이다.
이는 상기 논문(Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2021). Don't decay the learning rate, increase the batch size. Google Brain.)을 기반으로 진행된다.

![image](https://github.com/hoootteok2/aix_project/assets/168548944/fc888fe5-8bd3-4c2b-8a46-6da92d6584aa)

일반적으로 하이퍼파라미터 튜닝은 batch size를 고정하고 learning rate를 줄이는 방법을 사용하는데 이와 다르게 learning rate를 고정하고 batch size를 키우는 방법을 선택하면 위 그림과 같이 업데이트 할 parameter update수가 적어지면서 동시에 짧은 시간 내에 테스트의 정확도를 크게 변화 없이 빠르게 학습할 수 있다. 또한 learning rate와 batch size의 비례성도 있다. 이는 learning rate를 줄이는 방법을 통해 생기는 문제인 global optimum을 찾지 못하고 local minima에 빠지는 문제를 해결할 수 있게 된다.

### 3) Transfer Learning

![image](https://github.com/hoootteok2/aix_project/assets/168548944/fe381e92-4784-4180-be54-37c261bb9c87)

전이학습 Transfer Learning이란 이미 학습된 모델을 기반으로 새로운 데이터 셋에 재학습 시키는 방법이다. 이는 모델을 처음부터 학습하는 것보다 더 빠르고 효율적으로 학습할 수 있게 해준다. 이 프로젝트는 COCO dataset 으로 pretrained된 yolov7.pt 를 train 하는 과정에서 이용했다.

test와 train codes는 아래와 같다.

```
!python train.py --img 416 --batch-size 16 --epochs 50 --data dataset/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --device 0

!python test.py --weights runs/train/exp6/weights/best.pt --data dataset/data.yaml --img-size 416 --batch-size 16 --device 0
```



IV. Results & Evaluation & Analysis
========================

### 1) Results

https://youtu.be/Mds_EWV29I0?si=EFwF-B9OAF_5ZHXp

ASL_detection.py를 실행시켰으며 ASL 제스처 중 랜덤함 6가지 제스처를 웹캠에 인식시켰다. 인식시킨 제스처는 Y, V, W, L, R, I 이다.


### 2) Evaluation & Analysis

 learning rate를 0.01, epochs로 fiexed, batch size 8, 16 두 가지 경우를 비교한다.
왼쪽은 batch size 8의 test.py의 결과, 오른쪽은 batch size 16의 test.py결과로 고정한다.

이때, x축 값으로 주로 confidence를 이용할 것이며 confidence는 객체 탐지 모델에서 예측된 객체가 실제로 그 객체일 확률을 나타내는 값이고 0과 1 사이의 실수값으로 표현한다.

- Confusion Matrix

혼돈 행렬 Confusion Matrix는 모델의 예측 결과와 실제 결과를 비교한 표로, 대각선이 정확히 예측된 샘플의 비율이다. 해당 값이 높을수록 클래스를 정확히 예측했다는 의미이며 비대각선 요소가 많을 수록 잘못 예측됨을 나타낸다.

<img width="834" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/ffc70b73-2477-45b3-8de4-61413b9896e1">

두 혼동 행렬을 비교하면 우측 그래프 batch size 16일 때가 좌측 그래프 bach size 8 일때보다 확연히 대각선의 요소들이 더 많이 차지됨을 확인할 수 있다. 즉 batch size 8 일 때보다 더 클래스를 정확히 예측했음을 나타낸다.

- F1 Score vs Confidence Curve

F1 Score은 모델의 전체적인 성능을 평가하는 지표이며 정밀도와 재현율의 조화 평균 값이다.

<img width="848" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/de129868-137e-4d83-bd33-72d2550b3c4c">

좌측 그래프의 F1 최대 스코어와 그 때의 confidence는 각각 0.35, 0.1333이고 우측 그래프의 F1 최대 스코어와 그 때의 confidence는 0.60, 0.224이다. 이는 우측 그래프의 모델인 batch size 16이 더 좋은 성능을 보임을 의미한다. 또한 좌측 그래프의 비해 더 안정적으로 높은 F1 스코어를 유지함을 볼 수 있다.

- Precision vs Confidence Curve

모델이 예측한 양성 샘플 중 실제로 양성인 샘플의 비율인 precision을 보인 그래프이다. 

<img width="731" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/1c178508-97a8-4243-bec7-2ea6da751bc0">

좌측 그래프의 최대 precision과 그 때의 confidence는 0.922, 0.732이고 우측은 1.00, 0.843이다. 특히 좌측 그래프는 낮은 신뢰도를 유지하다가 confidence 0.4에서 급격히 증가하는 반면, 우측 그래프에서는 precision이 꾸준히 상승하는 개형을 보여준다. 즉, batch size 16 모델이 더 높은 성능을 보인다.

- Precision-Recall Curve

정밀도와 재현율 간의 관계를 보여주는 그래프이며 모델의 최적 성능 지점을 파악할 수 있게 해준다. 이때 mAP는 여러 클래스별로 얼마나 잘 수행되는지를 나타내주는 종합 성능 지표이다.

  <img width="826" alt="image" src="https://github.com/hoootteok2/aix_project/assets/168548944/5dec971a-1bef-43ee-a90a-9130adead189">

좌측 그래프의 mAP@0.5는 0.378, 우측 그래프는 0.776이며 동시에 좌측 그래프에 비해 우측 그래프가 높은 Precision과 recall을 보인다.

즉, 전반적으로 batch size 16일 때 batch size 8 일때보다 모델의 성능이 더 우수했다.

V. Related Work
==================

- for asl dataset recognition

https://public.roboflow.com/object-detection/american-sign-language-letters

- for YOLOv7 recognition 

https://www.plugger.ai/blog/yolov7-architecture-explanation

https://blog.roboflow.com/yolov7-breakdown/

- for ASL_detection.py

opencv, torch

- Hyper parameter tuning

 Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2021). Don't decay the learning rate, increase the batch size. Google Brain.


VI. Conclusion: Discussion
=======================

본 프로젝트를 통해 YOLOv7에 ASL 데이터 셋을 학습시키고 학습된 모델을 이용해 웹캠의 손동작을 인식시켜보았다. Model Training에서는 batch size 가 8일때보다 batch size 16일 때가 모델의 성능이 더 뛰어났음을 모델 성능 평가 자료들을 통해 확인할 수 있었다.

하지만, 대부분의 손의 제스처는 잘 인식 되었지만, 주먹을 쥔 포즈(A, E, M, S)같을 경우 인식이 잘 되지 않는다는 문제를 발견하였다. 따라서 이 문제를 해결하기 위해서는 모델의 정확도를 향상시켜야 함을 알 수 있다.
이 문제의 개선 방안으로는 Transfer learning을 이용한 다음과 같이 제시할 수 있다.

먼저, 위 프로젝트를 통해 사전 학습된 가중치 best.pt를 모델 학습에 그대로 이용한다. 이 때 모델의 구조를 수정하는데, output layer를 모델 클래스의 수 만큼 조절 한 후 새롭게 구성된 ASL Datsets으로 재학습 시킬 것이다. 이런 접근 방식을 이용하면 모델이 주먹 제스처를 더욱 정확히 인식할 수 있을 것이라고 기대한다.

