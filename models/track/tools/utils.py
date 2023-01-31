import os
import cv2

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")


def get_frame_num(source):
    cap = cv2.VideoCapture(source)
    i = 0
    while True:
        _, cur_frame = cap.read()
        if cur_frame is None:
            break
        i += 1

    return i


def bbox_scale_up(x_min, y_min, x_max, y_max, height, width, scale):
    h = y_max - y_min
    w = x_max - x_min
    x_min = int(max(0, x_min - w // scale))
    y_min = int(max(0, y_min - h // scale))
    x_max = int(min(width, x_max + w // scale))
    y_max = int(min(height, y_max + h // scale))
    return x_min, y_min, x_max, y_max