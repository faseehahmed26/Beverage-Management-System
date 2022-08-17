# -*- coding: utf-8 -*-
# @Date    : 14-07-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.plots import colors, Annotator
from PIL import Image

def get_model(model_path):
    """
    load model
    Args:
         model_path: model path
    Returns:
            - weight loaded model
    """
    return attempt_load(model_path, map_location='cpu')


def load(model, image, confidence_threshold, image_size):
    """
    model interface
    Args:
        model : object detection model (loaded)
        image : image (H, W, C)
        confidence_threshold: confidence interval
        image_size: image size

    Returns:
        - np.array
    """
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    stride = int(model.stride.max())
    imgsz = check_img_size(image_size, s=stride)  # check image size
    img = letterbox(image, imgsz, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    pred = model(img, augment=True)[0]

    conf_thres = confidence_threshold
    iou_thres = 0.45

    classes = None
    agnostic_nms = False
    max_det = 1000
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    counting = dict((k, 0) for k in names)
    for i, det in enumerate(pred):
        s, im0 = '', image.copy()
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                hide_labels = False
                hide_conf = False
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                counting[names[c]] += 1
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=None)
        return im0, counting
                
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            
       