import os

current_dir = os.getcwd()
ZOOM_IN_THRESHOLD = 150  # person
ZOOM_OUT_THRESHOLD = 450  # person
CLASS_FILE_PATH = "yolo_data/original/yolov3.names"
YOLO_MODEL_WEIGHTS = "/media/bharatforge/Ubuntu_data/Saranjit/DW2TF/data/yolo.pb"
ROS_IP_FILE = os.path.join(current_dir, 'ros_ip.txt')
# ROS_URL = "http://root:1234@192.168.0.63/axis-cgi/mjpg/video.cgi"

# ROS_URL ="rtsp://root:1234@192.168.0.61/MJPEG"
VIDEO_FILE_PATH= "/media/antpc/main_drive/vishal/datasets/vdo.avi"
# VIDEO_FILE_PATH = 0
# DeepSort configs
SIZE = 416  # NEW UPDATE
CONF_THRESHOLD = 0.7  # NEW UPDATE
IOU_THRESHOLD = 0.3  # NEW UPDATE
GPU_MEMORY_FRACTION = 0.95  # NEW UPDATE

MAX_COSINE_DISTANCE = 0.7  # 0.9
NN_BUDGET = 10
SENTRY_TIME = 10
AUONCODER_MODEL = os.path.join(current_dir, 'yolo_data', 'model_weights_ae_cnn_new.h5')
# YOLO_CFG, YOLO_WEIGHT, YOLO_OBJ_NAME = "yolo_data/yolo.cfg", "yolo_data/yolo_9000.weights", "yolo_data/obj.names"
YOLO_CFG, YOLO_WEIGHT, YOLO_OBJ_NAME = "yolo_data/original/yolov3.cfg", "yolo_data/original/yolov3.weights", "yolo_data/original/yolov3.names"
# VIDEO_PATH = "/media/antpc/main_drive/ankit/odct_main/odct_bitbucket/odct_4classes/videos_data/5objects.mp4"

# VIDEO_PATH = " rtsp://admin:admin1234@192.168.0.10/cam/realmonitor?channel=1&subtype=01&authbasic=YWRtaW46YWRtaW4xMjM0"
ROS_URL = "rtsp://admin:admin1234@192.168.0.10/cam/realmonitor?channel=1&subtype=01&authbasic=YWRtaW46YWRtaW4xMjM0"
MC_P = False
MCU_V = False
ALL_OBJ = True
