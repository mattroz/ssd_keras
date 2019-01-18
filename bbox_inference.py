import os
import cv2

from argparse import ArgumentParser

import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 300
img_width = 300

video_height = 1920
video_width = 1080

def init_model(model_file=None):
    K.clear_session()
    global img_height
    global img_width

    if not model_file:
        img_height = 300
        img_height = 300
        model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

        # TODO: Set the path of the trained weights.
        weights_path = './VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
        model.load_weights(weights_path, by_name=True)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    else:

        img_height = 224
        img_width = 224
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model = keras.models.load_model(model_file, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                                    'L2Normalization': L2Normalization,
                                                                    'DecodeDetections': DecodeDetections,
                                                                    'compute_loss': ssd_loss.compute_loss})

    print(model.summary())

    return model


def get_img_with_bbox(model, frame):

    xmin, ymin, xmax, ymax = [0 for i in range(4)]

    orig_images = []
    input_images = []

    print('\nFrame shape before feeding: ', frame.shape)
    orig_images.append(frame)

    img = Image.fromarray(frame)
    img = img.resize((img_height, img_width), Image.ANTIALIAS)
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=True,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded[0])

    for box in y_pred_decoded[0]:
        assert orig_images[0].shape[0] == video_height
        assert orig_images[0].shape[1] == video_width
        # Transform the predicted bounding boxes to the original image size.
        xmin = box[2] * video_width / img_width
        ymin = box[3] * video_height / img_height
        xmax = box[4] * video_width / img_width
        ymax = box[5] * video_height / img_height
        print(xmin, ymin, xmax, ymax)

    return xmin, ymin, xmax, ymax


def resize_and_rotate_frame(frame):
    w,h,c = frame.shape
    h //= 2
    w //= 2
    frame = cv2.resize(frame, (img_height, img_width))
    frame = frame.transpose(1, 0, 2)

    return frame


def inference_on_video(model, path_to_video, path_to_save_output=None):
    global video_width
    global video_height

    file = path_to_video.split('/')[-1]

    if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.MOV'):
        print("Processing %s" % file)
        cap = cv2.VideoCapture(path_to_video)
        out = None

        ### cap ###
        bbox_found = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # cv2.imwrite('/'.join(path_to_save_output.split('/')[:-1]) + '/frame.jpg', frame)

            if out is None and path_to_save_output:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(path_to_save_output, fourcc, 30.0, (video_width, video_height))
                print('Save to ', path_to_save_output)

            if not bbox_found:
                xmin, ymin, xmax, ymax = get_img_with_bbox(model, frame)
                print(xmin, ymin, xmax, ymax)

            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    argparser = ArgumentParser()
    argparser.add_argument('--video', type=str, required=True)
    argparser.add_argument('--model', type=str, required=False, default=None)
    args = argparser.parse_args()

    model = init_model(args.model)

    files = list(os.walk(args.video))

    # Check if the path throwed is a file or directory with files
    if files == []:
        root = '/'.join(args.video.split('/')[:-1])
        filename_to_save = args.video.split('/')[-1].split('.')[0] + '_ssd.avi'
        path_to_save = os.path.join(root, filename_to_save)
        inference_on_video(model, args.video, path_to_save)
    else:
        for root, dirs, filenames in files:
            if filenames == []:
                continue
            for inner_file in filenames:
                path_to_file = os.path.join(root, inner_file)
                filename_to_save = inner_file.split('.')[0] + '_ssd.avi'
                path_to_save_file = os.path.join(root, filename_to_save)
                inference_on_video(model, path_to_file, path_to_save_file)


if __name__=='__main__':
    main()