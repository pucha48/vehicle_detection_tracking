import math
import cv2
import requests
import numpy as np
import os

def __init__():
    pass

def calculate_area(w, h, image_area):
    area = 2 *(w + h)
    iou = (area / image_area) * 100
    return iou


def create_letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def return_box(box, format, width, height):
    cord = []
    if format == 'min-max':
        x1 = abs(int(round(((box[0] - box[2] / 2.0) * width).item())))
        y1 = abs(int(round(((box[1] - box[3] / 2.0) * height).item())))
        x2 = abs(int(round(((box[0] + box[2] / 2.0) * width).item())))
        y2 = abs(int(round(((box[1] + box[3] / 2.0) * height).item())))
        cord = [x1, y1, x2, y2]
    elif format == 'xywh':
        x1 = int(round(((box[2] - box[0] / 2.0) * width).item()))
        y1 = int(round(((box[3] - box[1] / 2.0) * height).item()))
        w = int(round((box[2] * width).item()))
        h = int(round((box[3] * height).item()))
        cord = [x1, y1, w, h]
    return cord


def return_box_2(box, format, width, height):
    cord = []
    if format == 'min-max':
        x1 = abs(int(round(((box[0] - box[2] / 2.0) * width).item())))
        y1 = abs(int(round(((box[1] - box[3] / 2.0) * height).item())))
        x2 = abs(int(round(((box[0] + box[2] / 2.0) * width).item())))
        y2 = abs(int(round(((box[1] + box[3] / 2.0) * height).item())))
        cord = [x1, y1, x2, y2]
    elif format == 'xywh':
        x1 = int(round(box[0]))
        y1 = int(round(box[1]))
        w = int(round(box[2] - box[0]))
        h = int(round(box[3] - box[1]))
        cord = [x1, y1, w, h]
    return cord




def write_frames(box, frame, frame_count):
    height, width = frame.shape[:2]
    cls_id = box[6].item()
    cord = return_box(box=box, format='min-max', width=width, height=height)
    # print(cord)
    crop_img = frame[cord[1]:cord[3], cord[0]:cord[2]]
    folder = os.path.join(os.getcwd(), 'Crop_images', str(cls_id))
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = 'crop_%s.jpg'%  frame_count
    cv2.imwrite(os.path.join(folder, file_name), crop_img)


def calculate_aspect_ratio(x1, y1, x2, y2, zoom_in_threshold, zoom_out_threshold):
    # Eucledian distance
    distance = math.sqrt(math.pow(float(x1) - float(x2), 2) + math.pow(float(y1) - float(y2), 2))
    if distance  <= zoom_in_threshold:
        # print('distance',distance)
        return 1 #1
    elif distance > zoom_out_threshold:
        # print('distance',distance)
        return 2#2
    else:
        # print('distance', distance)
        return 0


def sentry_mode():
    objects = ({'idx': 'null', 'label': 'null', 'bbox': [0, 0, 0, 0], 'actions': {'zoom': 0,
                                                                                  'sentry_mode': 0}}) #1
    return objects


def create_histogram(frame):
    bin_image = cv2.imread(frame)
    image = cv2.cvtColor(bin_image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# Function to read frames from IP
def post_boxes_to_ros(payload, file_path):
    with open(file_path, 'r') as myfile:
        ROS_HTTP_SERVER = myfile.read().replace('\n', '')

    ROS_HTTP_SERVER = ROS_HTTP_SERVER.strip()

    try:
        r = requests.post(ROS_HTTP_SERVER, data=payload)
        # print('content', r.content)
    except Exception as e:
        print("error in http request", e)


def write_text(frame, text, x, y):
    out_frame = cv2.putText(frame, str(text), (x, y), 0, 5e-3 * 200, (0, 0, 255), 2)
    return out_frame

# class Autoencoder:


    # def encoder(self, input_img):
    #     conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    #     pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    #     conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    #     pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    #     conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #     return conv3
    #
    # def decoder(self, conv3):
    #     conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
    #     up1 = tf.keras.layers.UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
    #     conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
    #     up2 = tf.keras.layers.UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
    #     decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)
    #     return decoded
    #
    # def autoencoder(self, input_img):
    #     encoder_output = self.encoder(input_img)
    #     decoder_output = self.decoder(encoder_output)
    #     return decoder_output
    #
    # def intiate_model(self, graph):
    #     with graph.as_default():
    #         img_width, img_height = 100, 100
    #         input_img = tf.keras.layers.Input(shape=(img_width, img_height, 3))
    #         autoencoder_cnn = tf.keras.models.Model(input_img, self.autoencoder(input_img))
    #
    #         autoencoder_cnn.summary()
    #         autoencoder_cnn.load_weights('model_weights_ae_cnn.h5', by_name=True)
    #         autoencoder_cnn._layers.pop(-1)
    #         autoencoder_cnn._layers.pop(-1)
    #         autoencoder_cnn._layers.pop(-1)
    #         autoencoder_cnn._layers.pop(-1)
    #         autoencoder_cnn._layers.pop(-1)
    #         return autoencoder_cnn
