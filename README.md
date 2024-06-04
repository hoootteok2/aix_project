Title: YOLOv7을 이용한 수어 번역기
==========
Members
===========
* 송서빈 
* 이예림

I. Proposal (Option 1)
=============================
### Motivation: Why are you doing this?

소통을 하고자 하는 것은 인간의 기본적인 욕구입니다. 음성 언어로 소통하지 못하는 특수한 사람들의 상황을 반영하여, 소통을 가능하게 하려는 동기에서 프로젝트를 시작하였습니다.
평범한 이들이 소통하는 방식 이외의 방법으로 소통할 수 있는 가능성을 보여줌으로써, 평등한 인류를 도모하고, 소통을 위한 각자의 방식이 있을 수 있음을 역설하고자 합니다.

### What do you want to see at the end?

웹캠에서 손동작을 yolov7을 통해 object detection 한 후, 의미를 텍스트로 디스플레이
  
II. Datasets
======================================
- Describing your dataset

https://public.roboflow.com/object-detection/american-sign-language-letters/1

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters

1) explain dataset

2) explain yolov7-object detection


III. Methodology & Results
========================
1) 실행환경


2) 실행코드(prompt/python)

3) algorithm

4) Results
video
result model



- Explaining your choice of algorithms (methods)
- Explaining features (if any)
- 
IV. Evaluation & Analysis
=====================
- Graphs, tables, any statistics (if any)
  1) model accuracy updated graph
  2) 
- 
V. Related Work (e.g., existing studies)
==================
- https://public.roboflow.com/object-detection/american-sign-language-letters

for asl dataset recognition

- opencv

for display algorithm
  
- Tools, libraries, blogs, or any documentation that you have used to do this project.
- 
VI. Conclusion: Discussion
=======================
### 세부 일정

#### 6/4, 6/5 : model trial basement for accuracy (조사)

- 6/4 : transfer learning git 정리 후 output 정리 계획

- 6/5 : transfer learning / batch, epoch, lr를 어떻게 적용 시켜서 updated할 지 계획
- 6/5 : batch, epoch, lr 에 관한 정리할 것

-> batch, epoch, lr조절 단계

(참고 논문 첨부할 것, lr를 조절하지 말고 batch를 조절하라는 논문 : 사전 project를 통해 batch 와 lr를 어느정도 비례관계 시키는게 좋다는 결과 이용할 것)

-> transfer leraning을 통해 조절

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

