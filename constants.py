import os
current_dir = os.getcwd()
CLASS_FILE_PATH= os.path.join(current_dir, 'weights','original', 'yolov3.names')
CFG_FILE = os.path.join(current_dir, 'weights','original','yolov3.cfg')
WEIGHTS = os.path.join(current_dir,'weights','original', 'yolov3.weights')

VIDEO_PATH="testing_video/final.mp4"
SIZE = 416  # NEW UPDATE
CONF_THRESHOLD = 0.5 # NEW UPDATE
IOU_THRESHOLD = 0.2  # NEW UPDATE
GPU_MEMORY_FRACTION = 0.95  # NEW UPDATE

MAX_COSINE_DISTANCE = 0.7  # 0.9
NN_BUDGET = 10

DEVICE = ''
IMG_SIZE = 416
