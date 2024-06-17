import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

# model load
weights_path = 'final_best.pt'  # model custom train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
model.eval()

# class
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# webcam
cap = cv2.VideoCapture(0)

def preprocess_image(image, img_size=416):
    # image processing
    img = letterbox(image, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def postprocess(prediction, img, img0, conf_thres=0.25, iou_thres=0.45):

    pred = non_max_suppression(prediction, conf_thres, iou_thres)
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    return pred

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   
    img = preprocess_image(frame)
   

    with torch.no_grad():
        pred = model(img, augment=False)[0]
   

    pred = postprocess(pred, img, frame)
   
    # 
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = f'{classes[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=[255, 0, 0], line_thickness=2)
   
    cv2.imshow('Webcam', frame)
   
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()