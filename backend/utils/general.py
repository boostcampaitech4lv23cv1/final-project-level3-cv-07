import os
import cv2
import shutil


# from convert import parsing_results

def create_directory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

def remove_if_exists(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    
    
def write_results(filename, results):
    with open(filename, "a", encoding="UTF-8") as f:
        f.writelines(results)
        
def get_frame_num(source):
    cap = cv2.VideoCapture(source)
    frame_list = []
    i = 0
    while True:
        ret, cur_frame = cap.read()
        if cur_frame is None:
            break
        i += 1

    return i
