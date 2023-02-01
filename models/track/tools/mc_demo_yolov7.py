import argparse
import time
from pathlib import Path
import sys
import os
from collections import defaultdict
import glob
import re

import cv2
import math
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
from numpy import random
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
# import face_recognition

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)
from yolov7.detect_temp import detect_image

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

from fast_reid.fast_reid_interfece import FastReIDInterface

sys.path.insert(0, "./yolov7")
sys.path.append(".")

def calculate_similarity(target_feature, tracker_feat, sim_thres):
    print("Similairties(cosine) list: ")
    print(
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="cosine",
        )
    )
    print("Similairties(Euclidean) list: ")
    print(
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="euclidean",
        )
    )
    print(f"Similarity Threshold : {opt.sim_thres}")
    sim = (
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="cosine",
        )
        > sim_thres
    )  # distance가 1 이상인 (즉, 비슷하지 않은) tracker 찾기
    t_ids = np.asarray(list(tracker_feat.keys()))
    valid_ids = t_ids[sim[0]]  # key에 넣어서 해당 tracker ID만을 뽑아내기
    return valid_ids

def dbscan(target_dir,tracklet_dir):
    tracklet_imgs = glob.glob(tracklet_dir+'/*.png')
    # encodings = [DeepFace.represent(img_path=img,enforce_detection=False,model_name="Facenet512") for img in tracklet_imgs]
    data = []
    for imagePath in tracklet_imgs : 
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        boxes = face_recognition.face_locations(rgb,
		    model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
        for (box, enc) in zip(boxes, encodings)]
        data.extend(d)
    encodings = [d["encoding"] for d in data]
    # dump the facial encodings data to disk
    stime = time.time()
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)
    etime = time.time()
    print(f"DBSCAN time elapsed :{etime - stime}")
    label_ids = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(label_ids>-1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    return


def parsing_results(valid_ids, save_dir, num_frames):
    with open(os.path.join(save_dir, "results.txt"), "r") as f:
        lines = f.readlines()

        parsed_lines = []

        # Remove unnecessary info and casting data type (str -> int)
        for line in lines:
            line = line.split(",")[:-3]
            line = list(map(int, list(map(float, line))))
            parsed_lines.append(line)

        # Remove redundant info and soted by frame > obj_id
        parsed_lines = sorted(
            list(map(list, set(list(map(tuple, parsed_lines))))),
            key=lambda x: (x[0], x[1]),
        )

        # Summary info (per frame)
        final_lines = []
        i = 1
        for line in parsed_lines:
            frame, obj_id, x, y, w, h, conf = line
            
            ### save valid face
            if obj_id in valid_ids:
                if not final_lines or frame != final_lines[-1][0]:
                    final_lines.append([frame, x, y, x + w, y + h])
                else:
                    final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

            ### save all face (for debugging)
            # if not final_lines or frame != final_lines[-1][0]:
            #     final_lines.append([frame, x, y, x + w, y + h])
            # else:
            #     final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

                
        total_lines = []
        
        idx = 1
        for i in range(len(final_lines)):
            if idx < final_lines[i][0]:
                while idx < final_lines[i][0]:
                    total_lines.append([idx+1])
                    idx += 1
            total_lines.append(final_lines[i])
            idx += 1
            
        while len(total_lines) < num_frames:
            total_lines.append([total_lines[-1][0] + 1])
                
        # for debugging            
        '''
        for line in total_lines:
            print(line[0], end=" / ")
            print(num_frames, end = " : ")
            print(line)
        ''' 
    return total_lines


def calc_euclidean_dist(x, y, cx, cy):
    return math.sqrt((cx - x) ** 2 + (cy - y) ** 2)


def calc_manhattan_dist(x, y, cx, cy):
    return abs(cx - x) + abs(cy - y)


# mask generator v0 (exactly same as not using mask)
def mask_generator_v0(x_min, y_min, x_max, y_max):
    w = x_max - x_min
    h = y_max - y_min
    mask = np.ones(shape=(h, w), dtype=np.float16)
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


# mask generator v1 (using Euclidean distance and thresholding)
def mask_generator_v1(x_min, y_min, x_max, y_max, thr=0.7):
    w = x_max - x_min
    h = y_max - y_min
    cx = w // 2
    cy = h // 2
    mask = np.zeros(shape=(h, w), dtype=np.float16)
    max_dist = calc_euclidean_dist(0, 0, cx, cy)

    # fill mask with L2 distance (from each pixel to center pixel)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            mask[i, j] = calc_euclidean_dist(j, i, cx, cy)
    mask /= max_dist  # normalize all dist
    mask = 1 - mask
    mask[mask >= thr] = 1
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


# mask generator v2 (using Manhattan distance)
def mask_generator_v2(x_min, y_min, x_max, y_max, thr=0.7):
    w = x_max - x_min
    h = y_max - y_min
    cx = w // 2
    cy = h // 2
    mask = np.zeros(shape=(h, w), dtype=np.float16)
    max_dist = calc_manhattan_dist(0, 0, cx, cy)

    # fill mask with L1 distance (from each pixel to center pixel)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            mask[i, j] = calc_manhattan_dist(j, i, cx, cy)
    mask /= max_dist  # normalize all dist
    mask = 1 - mask
    mask[mask >= thr] = 1
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


def extract_feature(target_path, save_dir):
    mtcnn = MTCNN(margin=30)
    img = Image.open(target_path)
    img_cropped = mtcnn(img, save_path=str(save_dir) + "/target_detect.png")
    # resnet = InceptionResnetV1(pretrained="vggface2").eval()
    # img_embedding = resnet(img_cropped.unsqueeze(0))



def detect(opt, save_img=False):
    start_time_total = time.time()

    source = opt.source
    target_path = opt.target
    weights, view_img, save_txt, imgsz, trace = (
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        opt.trace,
    )
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    save_results = opt.save_results
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # save_dir = Path(
    #     increment_path("runs" / Path(opt.project) / opt.name, exist_ok=False)
    # )  # increment run

    # Directories
    save_dir = Path(opt.work_dir) # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    extract_feature(target_path, save_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:  # False
        model = TracedModel(model, device, opt.img_size)

    if half:  # FP 16
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride
        )  # 설정한 video load : img_size, nframes

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once

    t0 = time.time()
    results_tlbr = defaultdict(dict)
    results = []
    for frame, path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            # Run tracker
            detections = []
            if len(det):  # detection이 존재하면!
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()  # [[bbox1],[bbox2]..]
                detections = det.cpu().numpy()
                detections[:, :4] = boxes  #

            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh  # `(top left x, top left y, width, height)`
                tlbr = t.tlbr  # `(min x, min y, max x, max y)`
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    results.append(
                        {
                            "frame": frame, 
                            "tid": tid, 
                            "tl_x": tlwh[0], 
                            "tl_y": tlwh[1],
                            "width": tlwh[2],
                            "height": tlwh[3],
                            "conf": float(t.score)
                        }
                    )
                    
                    results_tlbr[tid][frame] = np.append(tlbr, t.score)

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f"{tid}, {int(tcls)}"
                        else:
                            label = f"{tid}, {names[int(tcls)]}"
                        plot_one_box(
                            tlbr,
                            im0,
                            label=label,
                            color=colors[int(tid) % len(colors)],
                            line_thickness=2,
                        )
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Stream results
            if view_img:
                cv2.imshow("BoT-SORT", im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )

    print(f"Done. ({time.time() - t0:.3f}s)")

    end_time_total = time.time()
    
    print(f"Total Time Elapsed : {end_time_total - start_time_total}")
    
    targeted_ids, valid_ids = get_valid_tids(
        tracker,
        results_tlbr,
        opt.work_dir + "/tracklet",
        opt.work_dir + "/target_detect.png",
        opt.min_frame,
        opt.conf_thresh,
        opt.work_dir
    )
    
    return results, targeted_ids, valid_ids, fps

def bbox_scale_up(x_min, y_min, x_max, y_max, height, width, scale):
    h = y_max - y_min
    w = x_max - x_min
    x_min = int(max(0, x_min - w // scale)) 
    y_min = int(max(0, y_min - h // scale))
    x_max = int(min(width, x_max + w // scale)) 
    y_max = int(min(height, y_max + h // scale))
    return x_min, y_min, x_max, y_max

def get_valid_tids(tracker, results, tracklet_dir, target_dir, min_length, conf_thresh, work_dir):
    """
    각각의 tracker에서 대표 feature를 뽑고 similarity 계산하기
    1. tracker status에 대한 설명
    - tracker.tracked_stracks : 현재 frame에서 tracking이 이어지고 있는 tracker instance
    - tracker.removed_stracks : tracking이 종료된 tracker instance
    2. TODO
    - 유효한 tracker로 인정하기 위한 최소 frame은 몇으로 잡을지 결정
    - 유효한 tracker에서 feature는 어떻게 뽑을지 결정
    """
    # 영상이 끝난 시점에 tracking 하고 있던 tracker들이 자동으로 removed_stracks로 status가 전환되지 않기 때문에
    # 영상이 끝난 시점에서 tracking을 하고 있었던 tracker와 과거에 tracking이 끝난 tracker들 모두를 관리 해야합니다.
    t_ids = {}
    img = cv2.imread(
        f"{work_dir}/image_orig/frame_1.png"
    )
    height, width, layers = img.shape
    size = (width, height)
        
    tracks = list(set(tracker.removed_stracks + tracker.tracked_stracks + tracker.lost_stracks))
    for i in tracks:
        if (i.tracklet_len > min_length):  # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            frame,value = sorted(results[i.track_id].items(), key = lambda x : x[1][4])[-1]
            x1,y1,x2,y2,conf = value
            if conf > conf_thresh : 
                sx1, sy1, sx2, sy2 = bbox_scale_up(
                    x1, y1, x2, y2, height, width, 3
                )
                frame_img = cv2.imread(
                    f"{work_dir}/image_orig/frame_{frame}.png"
                )
                cv2.imwrite(
                    f"{tracklet_dir}/{i.track_id}.png",
                    np.array(
                        frame_img[
                            int(sy1) : int(sy2), int(sx1) : int(sx2), :
                        ]
                    ),
                )
                
                t_ids[i.track_id]=conf   
    # if opt.dbscan :
    #     dbscan(target_dir,tracklet_dir)
    #     return True
    # else :
    dfs = DeepFace.find(
        img_path=target_dir, db_path=tracklet_dir, enforce_detection=False, model_name= 'VGG-Face'
    )

    targeted_ids = {}
    
    for i in range(len(dfs)):
        id = int(dfs.iloc[i].identity.split("/")[-1].split(".")[0])
        id_conf = t_ids.pop(id)
        targeted_ids[id] = id_conf
    return dict(sorted(targeted_ids.items(),key=lambda x : x[1],reverse=True)), dict(sorted(t_ids.items(),key=lambda x : x[1],reverse=True))

if __name__ == "__main__":
    
    # import requests    
    # req = requests.get("http://ipconfig.kr")
    # server_ip = re.search(r'IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', req.text)[1]

    # base = "/opt/ml/final-project-level3-cv-07"
    # database_info = {
    #     "name": "database",
    #     "dir": f"{base}/database"
    # }

    # cartoonize_info = {
    #     "name": "cartoonize",
    #     "dir": f"{base}/models/track/cartoonize",
    #     "url": f"{server_ip}:30003/",
    # }

    # track_info = {
    #     "name": "track",
    #     "dir": f"{base}.models/track",
    #     "url": f"{server_ip}:30004/",
    # }
    
    
    
    # class Opt:
    #     weights= f"{track_info['dir']}/pretrained/yolov7-tiny.pt"
    #     source = f"{database_info['dir']}/uploaded_video/video.mp4"
    #     target = f"{database_info['dir']}/target/target.jpeg"
    #     cartoon = f"{track_info['dir']}/assets/chim_cartoonized.mp4"
    #     img_size = 1920
    #     conf_thres= 0.09
    #     iou_thres= 0.7
    #     sim_thres= 0.35
    #     device= "0"
    #     view_img= None
    #     save_txt= None
    #     nosave= None
    #     classes = None
    #     agnostic_nms= True
    #     augment= None
    #     update= None
    #     work_dir= f"{database_info['dir']}/work_dir"
    #     name= "exp"
    #     exist_ok= None
    #     trace= None
    #     hide_labels_name= False
    #     save_results = True
    #     save_txt_tidl = None
    #     kpt_label = 5
    #     hide_labels = False
    #     hide_conf = False,
    #     line_thickness = 3
        
    #     # Tracking args
    #     track_high_thresh = 0.3
    #     track_low_thresh = 0.05
    #     new_track_thresh = 0.4
    #     track_buffer = 30
    #     match_thresh = 0.7
    #     conf_thresh = 0.7 # added
    #     aspect_ratio_thresh = 1.6
    #     min_box_area = 10
    #     min_frame = 5 # added
    #     dbscan = True # added
    #     mot20 = True
    #     save_crop = None
        
    #     #CMC
    #     cmc_method = "sparseOptFlow"
        
    #     #ReID
    #     with_reid = False
    #     fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
    #     fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
    #     proximity_thresh = 0.5
    #     appearance_thresh = 0.25
    #     jde= False
    #     ablation= False
    
    # opt = Opt

    # print(opt)
    # # check_requirements(exclude=('pycocotools', 'thop'))

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ["yolov7.pt"]:
    #             detect(opt)
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect(opt)
    pass