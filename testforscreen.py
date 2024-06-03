import argparse
import json
import os
from pathlib import Path
from threading import Thread
import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2
from mss import mss

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

def test(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, save_json=False, 
         single_cls=False, augment=False, verbose=False, model=None, dataloader=None, save_dir=Path(''), 
         save_txt=False, save_hybrid=False, save_conf=False, plots=True, wandb_logger=None, compute_loss=None, 
         half_precision=True, trace=False, is_coco=False, v5_metric=False, image=None, device=''):
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device(device, batch_size=batch_size)
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        if trace:
            model = TracedModel(model, device, imgsz)
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        t0 = time_synchronized()
        out, train_out = model(img, augment=augment)
        t1 = time_synchronized()
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
        t2 = time_synchronized()

    # 비최대 억제 결과 처리
    for i, det in enumerate(out):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

    return out, model.names if hasattr(model, 'names') else model.module.names


def draw_boxes(image, detections, names):
    for det in detections:
        if len(det):
            det = det[det[:, 4].argmax().item()].unsqueeze(0)
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=2)
    return image

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='testforscreen.py')
    parser.add_argument('--weights', type=str, default='runs/train/exp8/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='datasets/aslv2/data.yaml', help='*.data path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)
    print(opt)
    
    device = select_device(opt.device, batch_size=32)
    model = attempt_load(opt.weights, map_location=device)
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, s=gs)
    if not opt.no_trace:
        model = TracedModel(model, device, imgsz)
    model.to(device).eval()
    half = device.type != 'cpu'
    if half:
        model.half()
    
    with mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 720, "height": 960}
        while True:
            img = np.array(sct.grab(monitor))
            detections, names = test(opt.data, opt.weights, imgsz=imgsz, conf_thres=opt.conf_thres,
                                     iou_thres=opt.iou_thres, model=model, half_precision=half,
                                     augment=opt.augment, image=img, device=device)
            img = draw_boxes(img, detections, names)
            cv2.imshow("Screen Capture", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
