import os
import cv2
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

video_height = 1920#640#512
video_width = 1080#480#256

def init_model():
    # 1: Build the Keras model

    K.clear_session() # Clear previous models from memory.

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

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.
    weights_path = './VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    # print(model.summary())
    return model


def get_img_with_bbox(model, frame):

    xmin, ymin, xmax, ymax = [0 for i in range(4)]

    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # # We'll only load one image in this example.
    # img_path = 'examples/vipad3.png'
    print(frame.shape)
    orig_images.append(frame)
    #frame = cv2.resize(frame, (img_height, img_width))
    #img = image.load_img(img_path, target_size=(img_height, img_width))
    img = Image.fromarray(frame)
    img = img.resize((img_height, img_width), Image.ANTIALIAS)

    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    #print('pred: ', y_pred)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    if len(y_pred_thresh) == 0:
        return xmin, ymin, xmax, ymax

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

    for box in y_pred_thresh[0]:
        #if box[0] != 15.0:
        #    continue
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = int(np.round(box[2] * orig_images[0].shape[1] / img_width))
        ymin = int(np.round(box[3] * orig_images[0].shape[0] / img_height))
        xmax = int(np.round(box[4] * orig_images[0].shape[1] / img_width))
        ymax = int(np.round(box[5] * orig_images[0].shape[0] / img_height))
        print(xmin, ymin, xmax, ymax)

    return xmin, ymin, xmax, ymax


def resize_and_rotate_frame(frame):
    w,h,c = frame.shape
    h //= 2
    w //= 2
    frame = cv2.resize(frame, (h,w))
    frame = frame.transpose(1, 0, 2)

    return frame


def main():
    model = init_model()

    cur_dir = '/home/matsvei.rozanau/Documents/fittable_test' #'/home/matsvei.rozanau/dev/pose-analytics/videos/test_video'
    for dir_with_vids in os.listdir(cur_dir):
        #if not os.path.isdir(dir_with_vids):
        #    continue
        for file in os.listdir(os.path.join(cur_dir, dir_with_vids)):
            if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.MOV'):
                print("Processing %s" % file)
                videofile = os.path.join(cur_dir, dir_with_vids, file)
                cap = cv2.VideoCapture(videofile)
                out = None

                ### cap ###
                bbox_found = False
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    #frame = resize_and_rotate_frame(frame)

                    if out is None:
                        file_to_save = os.path.join(cur_dir, dir_with_vids, file.split('.')[0] + '_ssd.avi')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(file_to_save, fourcc, 30.0, (video_width, video_height))

                    if not bbox_found:
                        xmin, ymin, xmax, ymax = get_img_with_bbox(model, frame)
                        if xmin is not None:
                            #bbox_found = True
                            low_crop_threshold = ymin
                            high_crop_threshold = ymax
                        else:
                            print('no bbox')
                            #continue
                        print(xmin, ymin, xmax, ymax)

                    frame_cp = np.array(frame).copy()
                    #frame_cp = frame[low_crop_threshold : high_crop_threshold, :]
                    #height, width = frame_cp.shape[:2]
                    #frame_cp = cv2.resize(frame_cp, (video_width, video_height), interpolation = cv2.INTER_CUBIC)

                    cv2.rectangle(frame, (xmin, ymin),(xmax, ymax),(255,0,0),3)
                    # cv2.imshow('frame', frame)
                    print('final crop res: ', frame_cp.shape)
                    out.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                #cv2.destroyAllWindows()

                ### end cap ###


if __name__=='__main__':
    main()
