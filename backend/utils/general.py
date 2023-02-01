import os
import cv2
import requests
import time
from pathlib import Path

# from convert import parsing_results

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

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

def print_running_time(opt):
    target_path = opt.target
    save_dir = Path(opt.work_dir)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)

    if opt.verbose:
        time_1 = time.time()
        print("\n[ Start Target Feature Extraction ]")
    requests.get("/extract_feature")
    
    if opt.verbose:
        print("\n[ Target Feature Extraction Done ]")
        time_2 = time.time()
        print("\n[ Start Tracking and Similarity Check]")
    tracker, results_temp, tracklet_dir, num_frames, fps = detection_and_tracking(opt)
    
    if opt.verbose:
        print("\n[ Target Feature Extraction Done]")
        time_3 = time.time()
        print("\n[ Start Similarity Check ]")
    # valid_ids = get_valid_results(opt, tracker, results_temp, tracklet_dir, save_dir)
    
    if opt.verbose:
        print("\n[ Similarity Check Done ]")
        time_4 = time.time()
        print("\n[ Start Result Parsing ]")
    final_lines = parsing_results(opt, valid_ids, num_frames)
    
    if opt.verbose:
        print("\n[ Result Parsing Done ]")
        time_5 = time.time()
        print("\n[ Start Swapping Video and Saving Video ]")
    save_face_swapped_vid(opt, final_lines, save_dir, fps)
    
    if opt.verbose:
        print("\n[ Swapping Video and Saving Video Done]")
        time_6 = time.time()
        print("[ All Process Successfully Done ]")
        print()
        print("{:-^70}".format(" Summary "))
        print(
            "{:<68}".format(
                f"| Time Elapsed to extract target feature : {round(time_2-time_1,2)} (s)"
            ),
            "|",
        )
        print(
            "{:<68}".format(
                f"| Time Elapsed to detection and tracking : {round(time_3-time_2,2)} (s)"
            ),
            "|",
        )
        print(
            "{:<68}".format(
                f"| Time Elapsed to get valid face by similarity check : {round(time_4-time_3,2)} (s)"
            ),
            "|",
        )
        print(
            "{:<68}".format(
                f"| Time Elapsed to parsing result : {round(time_5-time_4,2)} (s)"
            ),
            "|",
        )
        print(
            "{:<68}".format(
                f"| Time Elapsed to swap face and save video : {round(time_6-time_5,2)} (s)"
            ),
            "|",
        )
        print(
            "{:<68}".format(f"| Total Elapsed Time : {round(time_6-time_1,2)} (s)"), "|"
        )
        print("{:-^70}".format("-"))