'''This code is using yolo for both detection and classification of all 5 classes, Tracking is done by a single
deep sort for persons and for vehciles auto encoder is being used'''

import os, sys
import json
import warnings
import threading
import time, datetime
from PIL import ImageFile

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolo_data.yolo_utils import plot_boxes_cv2
from tools import generate_detections as gdet

from yolo_data.yolo_library import *
from utility.utils import *
from constants import *

from sklearn.cluster import MiniBatchKMeans

import cv2
import numpy as np
from utility.auto_encoder import Autoencoder

ImageFile.LOAD_TRUNCATED_IMAGES = True
curr_dir = os.getcwd()
idx_sel = 0
warnings.filterwarnings('ignore')
objects = []
click_flag = 0

timer = 0
class_dict = {0: 'cp', 1: 'mp', 2: 'cv', 3: 'mv', 4: 'ugv'}


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


def on_mouse(event, x, y, flags, params):
    global mouse_x, mouse_y, idx_sel
    global class_sel, id_sel, idx_list
    global click_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        print(mouse_x, mouse_y)
        for idx, value in enumerate(objects):
            x1 = value['bbox'][0]
            y1 = value['bbox'][1]
            x2 = value['bbox'][2]
            y2 = value['bbox'][3]
            if mouse_x > x1 and mouse_x < x2 and mouse_y > y1 and mouse_y < y2:
                print("in_mouse")
                class_sel = value['label']
                idx_sel = value['idx']
                idx_list = idx

        click_flag = 1


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
    imag_area = 2 * (height + width)
    cp_detections, cp_current_det = 0, 0
    # for box in bboxs_human:
    #     obj_size = return_box(box=box, format='xywh', width=width, height=height)
    #     box_iou = calculate_area(obj_size[2], obj_size[3], imag_area)
    #     if box_iou > 60:
    #         continue
    for box in bboxs_human:
        cls_id = box[6].item()
        cord = return_box(box=box, format='xywh', width=width, height=height)
        cp_boxs.append(cord)
        cv2.putText(frame, str('%s' % class_dict[cls_id]), (int(cord[0]), int(cord[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
    if len(cp_boxs) > 0:
        cp_detections, cp_current_det = deepsort_detections(encoder=encoder, frame=frame, boxs=cp_boxs,
                                                            nms_max_overlap=nms_max_overlap)

        # startexecution3 = datetime.datetime.now()
        tracker_cp.predict()
        tracker_cp.update(cp_detections)
        # stopexecution3 = datetime.datetime.now()
        # totalexectiontime_tracking = stopexecution3 - startexecution3
        # print('tracking time', totalexectiontime_tracking.total_seconds() * 1000)

    return cp_detections, cp_current_det, frame

kmeans_fit = None


def to_autoencoder(bboxs_vehicle, img, frame_count, frame, autoencoder_cnn, feature_list):
    global kmeans_fit
    labeled_list, boxs_cars, feature_list_partial = [], [], []
    height, width = img.shape[:2]
    # imag_area = 2 * (height + width)
    for box in bboxs_vehicle:
        cls_id = box[6].item()
        cord = return_box(box=box, format='min-max', width=width, height=height)
        boxs_cars.append({'cord': cord, 'tag': class_dict[cls_id]})
        cv2.rectangle(frame, (int(cord[0]), int(cord[1])), (int(cord[2]), int(cord[3])), (0, 255, 0), 2)
    if len(boxs_cars) == 0:
        pass
    else:
        if frame_count < 174:
            for boxes in boxs_cars:
                box = boxes['cord']
                croped_img = frame[box[1]:box[3], box[0]:box[2]]
                test_img = cv2.resize(croped_img, (100, 100))
                test_image = (test_img[..., ::-1].astype(np.float32)) / 255.0
                test_image = np.stack([test_image] * 1)
                feature = autoencoder_cnn.predict(test_image)
                feature_list.append(feature)
        elif frame_count == 175:
            kmeans = MiniBatchKMeans(n_clusters=len(boxs_cars), random_state=0, batch_size=6)
            X = np.reshape(feature_list, (len(feature_list), 100 * 100 * 3))
            kmeans_fit = kmeans.fit(X)
        else:
            for boxes in boxs_cars:
                box = boxes['cord']
                croped_img = frame[box[1]:box[3], box[0]:box[2]]
                if len(croped_img) == 0:
                    print(0)
                test_img = cv2.resize(croped_img, (100, 100))
                test_image = (test_img[..., ::-1].astype(np.float32)) / 255.0
                test_image = np.stack([test_image] * 1)
                time_autoencoder_start = datetime.datetime.now()
                test_feature = autoencoder_cnn.predict(test_image)
                feature_list_partial.append(test_feature)
                X = np.reshape(test_feature, (1, 100 * 100 * 3))

                # if frame_count % 10 == 0:
                #     X = np.reshape(feature_list_partial, (len(feature_list_partial), 100 * 100 * 3))
                #     kmeans.partial_fit(X)
                try:
                    id = kmeans_fit.predict(X)
                except Exception as e:
                    id = ['u_0']
                stopexecution4 = datetime.datetime.now()
                total_encoder_time = (stopexecution4 - time_autoencoder_start).total_seconds() * 1000
                # print('autoencoder time %s' % total_encoder_time)
                labeled_list.append({"id": (str(boxes['tag'] + '_' + str(id[0]))), 'bbox': box})
    return labeled_list


def main():
    yolo3 = YOLOv3(YOLO_CFG, YOLO_WEIGHT, YOLO_OBJ_NAME, is_plot=True)

    global idx_list, objects, idx_sel, class_sel, frame_current
    autoencoder = Autoencoder()
    feature_list = []
    cp_current_det = 0
    reference_time = time.time()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', on_mouse)
    frames = 0
    class_names = CLASS_FILE_PATH  # NEW UPDATE
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
    # NEW UPDATE
    classes = load_coco_names(class_names)  # NEW UPDATE
    mp_previous_det = 0
    cp_previous_det = 0
    frame_count = 0
    previous_det_cars = 0
    '''Loading Yolo file and passing each frame for inference'''

    model_filename = os.path.join('model_data', 'market1501.pb')  # model for person # NEW UPDATE
    encoder = gdet.create_box_encoder(model_filename, batch_size=128)  # NEW UPDATE
    autoencoder_cnn = autoencoder.initiate_model()
    print("-------")
    while True:
        try:
            tracker_flag = False
            if args.offline:
                ret, frame_current = cap.read()
                if not ret:
                    break
            frames += 1
            frame_count += 1
            frame = frame_current
            startexecution2 = datetime.datetime.now()
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
            objects = []
            boxs_cars = []
            cp_boxs = []
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            startexecution1 = datetime.datetime.now()
            bboxs = yolo3(img)
            stopexecution1 = datetime.datetime.now()
            totalexectiontime = stopexecution1 - startexecution1
            # print('detectiontime', totalexectiontime.total_seconds() * 1000)

            bboxs_human = np.asarray([i for i in bboxs if i[6].item() in [0, 1]], dtype=np.float64)
            bboxs_vehicle = np.asarray([i for i in bboxs if i[6].item() in [2, 3, 4]], dtype=np.float64)
            feature_list_partial = []
            labeled_list = []
            if ALL_OBJ:
                cp_detections, cp_current_det, frame = to_deepsort(bboxs_human, img, cp_boxs, frame, encoder,
                                                                   nms_max_overlap, tracker_cp)
                labeled_list = to_autoencoder(bboxs_vehicle, img, frame_count, frame, autoencoder_cnn, feature_list)
            elif MC_P:
                cp_detections, cp_current_det, frame = to_deepsort(bboxs_human, img, cp_boxs, frame, encoder,
                                                                   nms_max_overlap, tracker_cp)
            elif MCU_V:
                labeled_list = to_autoencoder(bboxs_vehicle, img, frame_count, frame, autoencoder_cnn, feature_list)

            if (cp_current_det != 0 and cp_current_det >= cp_previous_det) or len(boxs_cars) > 0:
                if len(cp_boxs) > 0:
                    for track in tracker_cp.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        bbox = track.to_tlbr()

                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255, 255, 255), 2)
                        cv2.putText(frame, str('%s' % track.track_id), (int(bbox[2]), int(bbox[1])), 0, 5e-3 * 200,
                                    (0, 255, 0), 2)
                        # cv2.imwrite(curr_dir + "/crop_images/" + str(frame_count) + ".jpg", frame)
                        objects.append({'idx': '%s' % track.track_id, 'label': 'person',
                                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                        'actions': {'zoom': 0, 'sentry_mode': 0}})
            if len(labeled_list) > 0:
                for label in labeled_list:
                    '''Zoom functions: 0 No zoom, 1 Zoom in, 2 Zoom out'''
                    bbox = label['bbox']
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, str('V_%s' % label['id']), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                                (0, 255, 0), 2)
                    # cv2.imwrite(curr_dir + "/crop_images/" + str(frame_count) + ".jpg", frame)
                    objects.append({'idx': '%s' % label['id'], 'label': 'car',
                                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                    'actions': {'zoom': 0, 'sentry_mode': 0}})
            # Added for testing

            if len(objects) == 0:
                pass
            else:
                if click_flag == 1:
                    for list_obj in objects:
                        if list_obj['idx'] == idx_sel and list_obj['label'] == class_sel:
                            click_obj = list_obj['bbox']
                            if calculate_aspect_ratio(click_obj[0], click_obj[1], click_obj[2], click_obj[3],
                                                      zoom_in_threshold=ZOOM_IN_THRESHOLD,
                                                      zoom_out_threshold=ZOOM_OUT_THRESHOLD) == 1:
                                # print("Zoom In")
                                frame = write_text(frame, 'Command Zoom In', 20, 55)
                                list_obj["actions"]["zoom"] = 0  # 1
                            elif calculate_aspect_ratio(click_obj[0], click_obj[1], click_obj[2], click_obj[3],
                                                        zoom_in_threshold=ZOOM_IN_THRESHOLD,
                                                        zoom_out_threshold=ZOOM_OUT_THRESHOLD) == 2:
                                # print("Zoom Out")
                                frame = write_text(frame, 'Command Zoom Out', 20, 55)
                                list_obj["actions"]["zoom"] = 0  # 2
                            else:
                                # frame = write_text(frame, 'Idle', 10, 20)
                                list_obj["actions"]["zoom"] = 0
                                # print("Don't zoom")

                            # diff = [frame_center[1] - bbox_center[0], frame_center[0] - bbox_center[1]]
                            # print("DIFFERENCE", diff)
                            cordinates = json.dumps(list_obj)
                            ros_send_time = datetime.datetime.now()
                            if args.online:
                                post_boxes_to_ros(cordinates, ROS_IP_FILE)
                            ros_end_time = datetime.datetime.now()
                            # print('ross_time_after_click', (ros_end_time - ros_send_time).total_seconds() * 1000)
                            reference_time = time.time()
                            tracker_flag = True
                            frame = write_text(frame, 'Tracking %s' % class_sel, 780, 80)
                            cv2.rectangle(frame, (int(click_obj[0]), int(click_obj[1])),
                                          (int(click_obj[2]), int(click_obj[3])), (255, 255, 0), 2)
                else:
                    blank_object = sentry_mode()
                    cordinates = json.dumps(blank_object)
                    ros_send_time = datetime.datetime.now()
                    if args.online:
                        post_boxes_to_ros(cordinates, ROS_IP_FILE)
                    ros_end_time = datetime.datetime.now()
                    # print('ross_time_after_click', (ros_end_time - ros_send_time).total_seconds() * 1000)
            if not tracker_flag:
                # current_time = time.time()
                # frame = write_text(frame, 'Lost Tracking', 780, 80)
                # timer = current_time - reference_time
                if timer > SENTRY_TIME:
                    payload = sentry_mode()
                    idx_sel = ''
                    class_sel = ''
                    cordinates = json.dumps(payload)

                    frame = write_text(frame, 'In Sentry Mode', 20, 80)
                    if args.online:
                        post_boxes_to_ros(cordinates, ROS_IP_FILE)
                else:
                    object = []
                    cordinates = json.dumps(object)
                    if args.online:
                        post_boxes_to_ros(cordinates, ROS_IP_FILE)

            cv2.imshow('image', frame)
            cv2.waitKey(1)
            cp_previous_det = cp_current_det
            # stopexecution2 = datetime.datetime.now()
            # totalexectiontime = stopexecution2 - startexecution2
            # print('totalexectiontime', totalexectiontime.total_seconds() * 1000, 'number of objects', len(bboxs))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    auto_encode = Autoencoder()
    main()
