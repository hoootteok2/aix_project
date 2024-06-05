Title: YOLOv7을 이용한 수어 번역기
==========
Members
===========
* 송서빈 
* 이예림

I. Proposal (Option 1)
=============================
### Motivation: Why are you doing this?

소통을 하고자 하는 것은 인간의 기본적인 욕구다. 음성 언어로 소통하지 못하는 특수한 사람들의 상황을 반영하여, 소통을 가능하게 하려는 동기에서 프로젝트를 시작하였다.
평범한 이들이 소통하는 방식 이외의 방법으로 소통할 수 있는 가능성을 보여줌으로써, 평등한 인류를 도모하고, 소통을 위한 각자의 방식이 있을 수 있음을 역설하고자 한다.

### What do you want to see at the end?

(웹캠에서 손동작을 yolov7을 통해 object detection 한 후, 의미를 텍스트로 디스플레이)
  
II. Datasets & Model
======================================
- Describing your dataset
#### 1) Preparing Dataset

https://public.roboflow.com/object-detection/american-sign-language-letters/1

Roboflow에서 제공하는 American Sign Language Letters 데이터셋을 사용했다.
Robflow에서 제공하는 데이터셋들은 이미지 주석, 데이터 전처리, 증강 및 다양한 지원들을 제공하는 사이트이다. 
따라서, 우리는 이 사이트에서 제공되는 기능 augmentation output을 3으로 설정하여 데이터를 준비하였다.

아래는 구글colab을 사용하여 데이터셋을 준비하는 과정이다.

```
# dataset download from ROBOFLOW
!curl -L "https://public.roboflow.com/ds/DGExMteSHE?key=XRBFhckXHE" -o roboflow.zip
!unzip roboflow.zip -d ./dataset
!rm roboflow.zip
```
이 ASL데이터셋은 총 1728장의 이미지이며 이 데이터셋의 Train/Test Split은 Train 1512, Valid 144, Test 72로 구성되어 있다다.
이 데이터셋의 class는 data.yaml을 통해 총 26개의 class가 있다.

```
train: ../train/images
val: ../valid/images

nc: 26
names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
```

(augmentation, scaling,,, 설명 + 이유도 설명)
(dataset 관련 image 첨부)

#### 2) YOLOv7

- About YOLOv7

(yolov7 model 설명)

- Code

준비된 customized dataset을 YOLOv7에 학습시킬 것이며, 코드는 아래와 같다.

```
# YOLOv7
!git clone https://github.com/WongKinYiu/yolov7.git
%cd yolov7

!pip install -r requirements.txt
```


YOLOv7에서 coco dataset으로 pretrained 된 가중치 yolov7.pt를 사용하기 위해 다음과 같이 코드를 작성한다.
```
import os

%cd /content/yolov7
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
model training과 testing 과정은 다음과 같다.
이때 training 의 batch size, learning rate, epochs는 필요에 따라 수정한다. 마찬가지로 필요에 따라 best.pt를 이용하여 tranfer learning을 통해 high accuracy를 도출한다.
```
%cd /content/yolov7
!python train.py --img 416 --batch-size 16 --epochs 50 --data dataset/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --device 0
```

test.py 실행 코드는 다음과 같다. 마찬가지로 추후 필요에 따라 img, batch 또한 변형한다.
```
!python test.py --weights runs/train/exp6/weights/best.pt --data dataset/data.yaml --img-size 416 --batch-size 16 --device 0
```



III. Methodology
========================
### tranfer learning & fine tuning

#### 1) Tranfer learning & Fine tuning


#### 2) Settings & Codes

- 실행환경

google colab, anaconda prompt(나중에 버전들 쭉 나열해놓기)

gpu(tpu2?? 이미지 첨부)


- 실행코드(prompt/python)

fine tuning 과정들 쭉


#### 3) Trials

-Trial 1

(이미지 추가 첨부 후 분석)



<img src="https://github.com/hoootteok2/aix_project/assets/168548944/f29cf099-5682-440c-a82f-d982756b8a5f" width="40%">


<img src="https://github.com/hoootteok2/aix_project/assets/168548944/a1745343-2930-4ecb-9e1a-f03c04acf9d6" width="40%">
<img src="https://github.com/hoootteok2/aix_project/assets/168548944/92fcaf47-7158-416c-a4df-438f6ea91e68" width="40%">

왼쪽의 이미지는 labels, 오른쪽의 이미지는 preds



#### 4) algorithm

detect.py (for video)






- Explaining your choice of algorithms (methods)
- Explaining features (if any)


IV.  Results & Evaluation, Analysis
=====================

- Graphs, tables, any statistics (if any)
  
####  1) Results

video (url)


best.pt (graph, tables..)


####  2) Evaluation & Analysis
    

V. Related Work (e.g., existing studies)
==================

- https://public.roboflow.com/object-detection/american-sign-language-letters

for asl dataset recognition

- opencv

for display algorithm
  
- Tools, libraries, blogs, or any documentation that you have used to do this project.



VI. Conclusion: Discussion
=======================

### 세부 일정

#### 6/4, 6/5 : model trial basement for accuracy (조사)

goal1 : improving & comparing the results
goal2 : checking

- 6/5 : transfer learning / batch, epoch, lr를 어떻게 적용 시켜서 updated할 지 계획
- 6/5 : batch, epoch, lr 에 관한 정리할 것

-> batch, epoch, lr조절 단계

(참고 논문 첨부할 것, lr를 조절하지 말고 batch를 조절하라는 논문 : 사전 project를 통해 batch 와 lr를 어느정도 비례관계 시키는게 좋다는 결과 이용할 것)

-> transfer leraning을 통해 조절(fine tuning)

-> 각 step으로 인한 변화 data추출 및 비교를 어떻게 할 지 결정

#### 6/6 ~ 14 : model second trial -> updated for high accuracy (goal : over 50)

-> export 할 때 마다 data정리 제발 git에 올려서 정리 꼭 해

- 6/6 : model trials 2, tranfer~ / ble 중 선택해서 ,,,.pt 파일 export할 것
- 6/7 : model trials 3, tranfer~ / ble 중 선택해서 ,,,.pt 파일 export할 것
- 6/8, 6/9 : implementing algorithms
  (알고리즘 문제 없는지 체크하고 model import)

-6/10~6/14 : model trials n updated


#### 6/8, 6/9 : implementing algorithms

- 6/8 : dev (scanning, text, video)
- 6/9 : implementing 

