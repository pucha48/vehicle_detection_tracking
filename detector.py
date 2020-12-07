
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from utilities import *
from constants import *
import torch
import numpy as np
import cv2




class YOLOv3(object):
    def __init__(self):
        print("Starting")
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else DEVICE)
        img_size = (320, 192) if ONNX_EXPORT else IMG_SIZE  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.model = Darknet(CFG_FILE, img_size)
        self.conf_thres = 0.3
        self.iou_thres = 0.6
        self.classes = None
        self.agnostic_nms = False
        # Load weights
        if WEIGHTS.endswith('.pt'):  # pytorch formatzz
            self.model.load_state_dict(torch.load(WEIGHTS, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, WEIGHTS)

    def detect(self, frame):
        bbox_list = []

        # Eval mode
        self.model.to(self.device).eval()

        # Export mode

        # Get names and colors
        names = load_classes(CLASS_FILE_PATH)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()

        t = time.time()
        im0 = frame
        img = create_letterbox(frame, new_shape=IMG_SIZE)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *x, conf, cls in det:
                    # print(xyxy)
                    bbox = [int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(cls)]
                    bbox_list.append(bbox)
                    #

        # print('Done. (%.3fs)' % (time.time() - t0))
        return bbox_list


if __name__ == "__main__":
    yolov3 = YOLOv3()
    CLASS_FILE_PATH = "black_cfg/obj.names"
    # CLASS_FILE_PATH = "black_cfg_6/obj.names"
    frames =0
    frame_count = 0
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    while True:
        ret, frame_current = cap.read()
        bbox = yolov3.detect(frame_current)
        for box in bbox:
            print(box)
            cls_id = int(box[4])
            cv2.rectangle(frame_current, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (255, 0, 0), 1)
            cv2.putText(frame_current, str('%s' %cls_id), (int(box[0]), int(box[1])), 0,
                        5e-3 * 200,
                        (0, 255, 0), 2)
        cv2.imshow('image', frame_current)
        cv2.waitKey(1)