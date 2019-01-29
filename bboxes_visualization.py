import cv2
import numpy as np

SCALES = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
ASPECT_RATIOS = [[1, 2, 0.5],
                [1, 2, 0.5, 3, 0.3333333333333333],
                [1, 2, 0.5, 3, 0.3333333333333333],
                [1, 2, 0.5, 3, 0.3333333333333333],
                [1, 2, 0.5],
                [1, 2, 0.5]]
IMG_HEIGHT = 224
IMG_WIDTH = 224


def calculate_bboxes_for_layer(layer_aspect_ratios, layer_scale):
    bboxes = []
    size = min(IMG_HEIGHT, IMG_WIDTH)

    for ar in layer_aspect_ratios:
        if ar == 1:
            bh = bw = size * layer_scale
            bboxes.append((bh,bw))
            print(f"Boxes for layer: {(bh, bw)}")
        else:
            bh = layer_scale * size / np.sqrt(ar)
            bw = layer_scale * size * np.sqrt(ar)
            print(f"Boxes for layer: {(bh,bw)}")
            bboxes.append((bh, bw))
    
    return bboxes


def visualize_boxes(layer_boxes, layer_idx):
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    img[:,:,:] = (255,255,255)
    center_x, center_y = IMG_HEIGHT // 2, IMG_WIDTH // 2

    for layer_box in layer_boxes:
        box_height, box_width = layer_box
        left_upper_corner_x = max(0, int(center_x - box_width // 2))
        left_upper_corner_y = max(0, int(center_y - box_height // 2))
        right_lower_corner_x = min(IMG_WIDTH, int(center_x + box_width // 2))
        right_lower_corner_y = min(IMG_HEIGHT, int(center_y + box_height // 2))

        img = cv2.rectangle(img, (left_upper_corner_x, left_upper_corner_y),
                                 (right_lower_corner_x, right_lower_corner_y),
                                 (255,0,0),
                                 (1 if layer_idx < 3 else 3))
    cv2.imshow('layer' + str(layer_idx), img)
    cv2.waitKey()


def main():
    for i in range(len(ASPECT_RATIOS)):
        boxes = calculate_bboxes_for_layer(ASPECT_RATIOS[i], SCALES[i])
        visualize_boxes(boxes, i)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()