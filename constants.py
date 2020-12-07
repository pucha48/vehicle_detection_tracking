import os
current_dir = os.getcwd()
ZOOM_IN_THRESHOLD=150 #person
ZOOM_OUT_THRESHOLD=450 #person
# CLASS_FILE_PATH="black_6_cfg/obj.names"
# CLASS_FILE_PATH= os.path.join(current_dir, 'cfg', 'obj.names')
# CFG_FILE = os.path.join(current_dir, 'cfg', 'yolo_v3_4.cfg')
# WEIGHTS = os.path.join(current_dir, 'weights', 'yolo_v3_16000.weights')

CLASS_FILE_PATH= os.path.join(current_dir, 'weights','original', 'yolov3.names')
CFG_FILE = os.path.join(current_dir, 'weights','original','yolov3.cfg')
WEIGHTS = os.path.join(current_dir,'weights','original', 'yolov3.weights')

YOLO_MODEL_WEIGHTS="/media/bharatforge/Ubuntu_data/Saranjit/DW2TF/data/yolo.pb"
ROS_IP_FILE=os.path.join(current_dir, 'ros_ip.txt')
ROS_URL="http://root:1234@192.168.0.63/axis-cgi/mjpg/video.cgi"
VIDEO_PATH="testing_video/123.avi"
# VIDEO_PATH= os.path.join(current_dir, 'test_video', 'VID_20191001_162304.mp4')
# ROS_URL ="rtsp://root:1234@192.168.0.61/MJPEG"
# VIDEO_FILE_PATH = 0
#DeepSort configs
SIZE = 416  # NEW UPDATE
CONF_THRESHOLD = 0.5 # NEW UPDATE
IOU_THRESHOLD = 0.2  # NEW UPDATE
GPU_MEMORY_FRACTION = 0.95  # NEW UPDATE

MAX_COSINE_DISTANCE = 0.7  # 0.9
NN_BUDGET = 10
SENTRY_TIME = 10
# AUONCODER_MODEL = os.path.join(current_dir,'model_data', 'model_weights_ae_cnn_new.h5')
AUONCODER_MODEL = '/media/antpc/main_drive/Saranjit/Autoencoder/autoencoder/model_weights_ae_cnn_new.h5'

DEVICE = ''
IMG_SIZE = 416

MC_P = False
MCU_V = False
ALL_OBJ = True