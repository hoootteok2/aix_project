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

이 데이터셋은 총 1728장의 이미지이며 아래 이미지와 같이 이 데이터셋의 Train/Test Split은 Train 1512, Valid 144, Test 72로 구성되어 있다.

<img src="https://github.com/hoootteok2/aix_project/assets/168548944/356e20d7-9cf3-4005-bb98-07d03024acb9" width="50%">


또한 dataset의 class는 data.yaml을 통해 확인하면 총 26개, 알파벳의 개수만큼 class가 있다.

```
train: ../train/images
val: ../valid/images

nc: 26
names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
```

### 2) Data Augmentatino

(augmentation, scaling,,, 설명 + 이유도 설명)
(dataset 관련 image 첨부)



III. Methodology
========================
### model & approving weights methods

### 1) YOLOv7 (이론)


### 2) transfer learning & hyper parameters tuning (이론)

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




V. Related Work (e.g., existing studies)
==================

- Tools, libraries, blogs, or any documentation that you have used to do this project.

- https://public.roboflow.com/object-detection/american-sign-language-letters

for asl dataset recognition

- opencv

for display algorithm
  

- Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2021). Don't decay the learning rate, increase the batch size. Google Brain.

Hyper parameter tuning





VI. Conclusion: Discussion (6/17)
=======================


