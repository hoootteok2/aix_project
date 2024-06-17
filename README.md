Title: YOLOv7을 이용한 수어 번역기
==========
Members
===========
* 송서빈(융합전자공학부/2020059189) : skmssb@hanyang.ac.kr
* 이예림(연극영화과/2018034057) : sdyerim@naver.com


I. Proposal (Option 1)
=============================
### Motivation: Why are you doing this?

소통을 하고자 하는 것은 인간의 기본적인 욕구다. 음성 언어로 소통하지 못하는 특수한 사람들의 상황을 반영하여, 소통을 가능하게 하려는 동기에서 프로젝트를 시작하였다.
평범한 이들이 소통하는 방식 이외의 방법으로 소통할 수 있는 가능성을 보여줌으로써, 평등한 인류를 도모하고, 소통을 위한 각자의 방식이 있을 수 있음을 역설하고자 한다.

### What do you want to see at the end?

웹캠에서 손동작을 입력하여 이 손을 yolov7을 통해 object detection 한 후, 손의 움직임을 object detection하는 것이다.
  
II. Datasets
======================================

### 1) Preparing Dataset

* #### About ASL

https://public.roboflow.com/object-detection/american-sign-language-letters/1

ASL이란 American Sign Language로, 미국과 캐나다에 살고 있는 농인들을 위한 수어이다. 
우리는 이 ASL image datasets을 Roboflow에서 제공하는 것으로 사용했다.
Robflow에서 제공하는 데이터셋들은 이미지 주석, 데이터 전처리, 증강 및 다양한 지원들을 제공하는 사이트이므로 본 프로젝트에서 유연하게 데이터셋을 이용할 수 있다.
따라서, 우리는 이 사이트에서 제공되는 기능 augmentation output을 3으로 설정하여 데이터를 준비하였다. 아래 이미지는 해당 dataset의 예시 이미지이다.

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






### 2) hyper parameters tuning (이론)

이때, hyper parameter tuning과정은 논문 인용할 것.


### 3) Settings & Codes - hyper parameter tunings with transfer learning

- 실행환경

google colab, anaconda prompt(나중에 버전들 쭉 나열해놓기)

gpu(tpu2 이미지 첨부)


- 실행코드(prompt/python) - 간단하게

hyper parameter tuning은 batch size 조절




IV.  Results & Evaluation, Analysis
=====================
  
###  Results 

-final video (url)


-analysis of hyper~ / best.pt (graph, tables..) : batch size에 따른 결과들 분석석




V. Related Work
==================

- for asl dataset recognition

https://public.roboflow.com/object-detection/american-sign-language-letters

- for YOLOv7 recognition 

https://www.plugger.ai/blog/yolov7-architecture-explanation

https://blog.roboflow.com/yolov7-breakdown/

- for display algorithm

opencv

- Hyper parameter tuning

 Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2021). Don't decay the learning rate, increase the batch size. Google Brain.



VI. Conclusion: Discussion
=======================


