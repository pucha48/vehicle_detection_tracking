import os, sys
import json
import warnings
import threading
import time, datetime
from PIL import ImageFile

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from utility.utils import *
from constants import *
from detector import YOLOv3 as YOLO
import cv2
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True
curr_dir = os.getcwd()
idx_sel = 0
warnings.filterwarnings('ignore')
click_flag = 0

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def change_parms(tracker, metric):
    max_cosine_distance = 0.9  # 0.9
    nn_budget = 10000
    nms_max_overlap = 0.1
    metric.change_cost(matching_threshold=max_cosine_distance, budget=nn_budget)
    tracker.change_params(metric, max_iou_distance=0.7, n_init=100)
    return tracker


def deepsort_detections(encoder, frame, boxs, nms_max_overlap):
    features = encoder(frame, boxs)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # 0.6
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    temp = len(scores)
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    return detections, temp

def frame_read_fun(url):
    global cap, ret, frame_current
    frame_count = 0
    URL = url
    cap = cv2.VideoCapture(URL)
    while True:
        frame_count += 1
        ret, frame_current = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break


def to_deepsort(bboxs_human, img, cp_boxs, frame, encoder, nms_max_overlap, tracker_cp):
    height, width = img.shape[:2]
    cp_detections, cp_current_det = 0, 0
    for box in bboxs_human:
        cls_id = box[4]
        cord = return_box(box=box, format='xywh', width=width, height=height)
        # cord = box[:4]
        cp_boxs.append(cord)
        cv2.putText(frame, 'V', (int(cord[0]), int(cord[1])), 0, 5e-3 * 50, (0, 255, 0), 1)
    if len(cp_boxs) > 0:
        cp_detections, cp_current_det = deepsort_detections(encoder=encoder, frame=frame, boxs=cp_boxs,
                                                            nms_max_overlap=nms_max_overlap)
        tracker_cp.predict()
        tracker_cp.update(cp_detections)
    return cp_detections, cp_current_det, frame

def main():
    yolo = YOLO()
    global idx_list, idx_sel, class_sel, frame_current
    cp_current_det = 0
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    max_cosine_distance = MAX_COSINE_DISTANCE
    nn_budget = NN_BUDGET
    c = 0

    # To make the code work online and offline
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", help="run in online mode", action='store_true')
    parser.add_argument("--offline", help="run in offline mode", action='store_true')
    args = parser.parse_args()

    if args.online:
        print("Starting in online mode.")
        # fetch cv2 from url
        thread_read_frame = threading.Thread(target=frame_read_fun, name='thread_read_frame', args=(ROS_URL,))
        thread_read_frame.start()

    elif args.offline:
        print("Starting in offline mode.")
        try:
            cap = cv2.VideoCapture(VIDEO_PATH)
        except Exception as e:
            print(e)

    else:
        sys.exit(1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_cp = Tracker(metric, max_iou_distance=0.5, max_age=50800, n_init=10, ww=2, vv=160)  # UPDATE FEB 25
    cp_previous_det = 0
    '''Loading Yolo file and passing each frame for inference'''
    model_filename = os.path.join('model_data', 'market1501.pb')  # model for person # NEW UPDATE
    encoder = gdet.create_box_encoder(model_filename, batch_size=128)  # NEW UPDATE
    while True:
        try:
            if args.offline:
                ret, frame_current = cap.read()
                if not ret:
                    print('Video Not available')
                    break
            frame = frame_current
            if cp_previous_det <= 1:
                max_cosine_distance = 0.9  # 0.9
                nn_budget = 10000
                nms_max_overlap = 0.1
                metric.change_cost(matching_threshold=max_cosine_distance, budget=nn_budget)
                tracker_cp.change_params(metric, max_iou_distance=0.7, n_init=100)
            else:
                max_cosine_distance = 0.3  # 0.9
                nn_budget = 10000
                nms_max_overlap = 0.1
                metric.change_cost(matching_threshold=max_cosine_distance, budget=nn_budget)
                tracker_cp.change_params(metric, max_iou_distance=0.7, n_init=100)
            c += 1
            boxs_cars = []
            cp_boxs = []
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxs = yolo.detect(frame_current)
            bboxs_vehicle = [i for i in bboxs if i[4] in [2,3,4,8]]    # Taking bicycle, car, motorcycle and truck as vehicles
            cp_detections, cp_current_det, frame = to_deepsort(bboxs_vehicle, img, cp_boxs, frame, encoder,nms_max_overlap, tracker_cp)
            if (cp_current_det != 0 and cp_current_det >= cp_previous_det) or len(boxs_cars) > 0:
                if len(cp_boxs) > 0:
                    for track in tracker_cp.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        bbox = track.to_tlbr()

                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255, 255, 255), 1)
                        cv2.putText(frame, str('%s' % track.track_id), (int(bbox[2]), int(bbox[1])), 0, 5e-3 * 50,
                                    (0, 255, 0), 1)
            cv2.imshow('image', frame)
            cv2.waitKey(1)
            cp_previous_det = cp_current_det
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
