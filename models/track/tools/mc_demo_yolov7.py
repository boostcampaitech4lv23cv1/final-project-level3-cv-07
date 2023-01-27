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
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

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


def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")


def get_frame(source):

    cap = cv2.VideoCapture(source)
    frame_list = []

    while True:
        ret, cur_frame = cap.read()
        if cur_frame is None:
            break

        frame_list.append(cur_frame)

    return frame_list


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


def get_valid_tids(tracker, results, frame_list, tracklet_dir, target_dir):

    """
    각각의 tracker에서 대표 feature를 뽑고 similarity 계산하기
    1. tracker status에 대한 설명
    - tracker.tracked_stracks : 현재 frame에서 tracking이 이어지고 있는 tracker instance
    - tracker.removed_stracks : tracking이 종료된 tracker instance
    2. TODO
    - 유효한 tracker로 인정하기 위한 최소 frame은 몇으로 잡을지 결정
    - 유효한 tracker에서 feature는 어떻게 뽑을지 결정
    3. FIXME
    - frame_list에 모든 프레임 정보 저장하지 않고, 뒤에서 필요한 frame만 cv.imread로 불러오기
    """
    # 영상이 끝난 시점에 tracking 하고 있던 tracker들이 자동으로 removed_stracks로 status가 전환되지 않기 때문에
    # 영상이 끝난 시점에서 tracking을 하고 있었던 tracker와 과거에 tracking이 끝난 tracker들 모두를 관리 해야합니다.
    t_ids = []

    createDirectory(tracklet_dir)

    # 과거 종료된 tracker들 중에서
    for i in tracker.removed_stracks:
        if (
            i.tracklet_len > 5
        ):  # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            middle_frame = (i.start_frame + i.end_frame) // 2
            x1, y1, x2, y2 = results[i.track_id][middle_frame]
            cv2.imwrite(
                f"{tracklet_dir}/{i.track_id}.png",
                np.array(
                    frame_list[middle_frame - 1][
                        int(y1) : int(y2), int(x1) : int(x2), :
                    ]
                ),
            )
            t_ids.append(i.track_id)

    # 영상이 끝난 시점에 살아있던 tracker에 대해서
    for i in tracker.tracked_stracks:
        if (
            i.tracklet_len > 5
        ):  # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            # test[i.track_id] = i.smooth_feat
            middle_frame = (i.start_frame + i.end_frame) // 2
            x1, y1, x2, y2 = results[i.track_id][middle_frame]
            cv2.imwrite(
                f"{tracklet_dir}/{i.track_id}.png",
                np.array(
                    frame_list[middle_frame - 1][
                        int(y1) : int(y2), int(x1) : int(x2), :
                    ]
                ),
            )
            t_ids.append(i.track_id)

    for i in tracker.lost_stracks:
        if (
            i.tracklet_len > 5
        ):  # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            # test[i.track_id] = i.smooth_feat
            middle_frame = (i.start_frame + i.end_frame) // 2
            x1, y1, x2, y2 = results[i.track_id][i.start_frame]
            cv2.imwrite(
                f"{tracklet_dir}/{i.track_id}.png",
                np.array(
                    frame_list[middle_frame - 1][
                        int(y1) : int(y2), int(x1) : int(x2), :
                    ]
                ),
            )
            t_ids.append(i.track_id)

    valid_ids = list(set(t_ids))
    dfs = DeepFace.find(
        img_path=target_dir, db_path=tracklet_dir, enforce_detection=False
    )

    targeted_ids = []

    for i in range(len(dfs)):
        id = int(dfs.iloc[i].identity.split("/")[-1].split(".")[0])
        valid_ids.remove(id)
        targeted_ids.append(id)

    return targeted_ids, valid_ids


def save_face_swapped_vid(final_lines, save_dir, fps):
    ## FIXME
    img = cv2.imread(
        f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/{opt.project}/image_orig/frame_1.png"
    )
    height, width, layers = img.shape
    size = (width, height)
    swap_s = time.time()

    frame_array = []
    # face swap per frame
    for line in tqdm(final_lines):
        assert (len(line) - 1) % 4 == 0
        frame_idx = line[0]  # Image Index starts from 1
        orig_img = cv2.imread(
            f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/{opt.project}/image_orig/frame_{frame_idx}.png"
        )
        cart_img = cv2.imread(
            f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/{opt.project}/image_cart/frame_{frame_idx}.png"
        )
        resized_cart_img = cv2.resize(cart_img, size, interpolation=cv2.INTER_LINEAR)
        face_swapped_img = orig_img
        for i in range(((len(line) - 1) // 4) - 1):
            x_min, y_min, x_max, y_max = (
                line[4 * i + 1],
                line[4 * i + 2],
                line[4 * i + 3],
                line[4 * i + 4],
            )  # original bbox
            sx_min, sy_min, sx_max, sy_max = bbox_scale_up(
                x_min, y_min, x_max, y_max, height, width, 2
            )  # scaled bbox ('s' means scaled)

            """
            Select mask generator function
            - mask_generator_v0: same as not using mask
            - mask_generator_v1: using Euclidean distance (L2 distance) and thresholding
            - mask_generator_v2: using Manhattan distance (L1 distance) and thresholding
            - mask_generator_v3: using padding
            """

            mask, inv_mask = mask_generator_v3(sx_min, sy_min, sx_max, sy_max)
            orig_face = orig_img[sy_min:sy_max, sx_min:sx_max]
            cart_face = resized_cart_img[sy_min:sy_max, sx_min:sx_max]
            swap_face = np.multiply(cart_face, mask) + np.multiply(orig_face, inv_mask)
            face_swapped_img[sy_min:sy_max, sx_min:sx_max] = swap_face
        frame_array.append(face_swapped_img)
    swap_e = time.time()
    print(f"Time Elapsed for face swap: {swap_e - swap_s}")

    out = cv2.VideoWriter(
        os.path.join(save_dir, opt.project + "_cartoonized" + ".mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    for i in tqdm(range(len(frame_array))):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def parsing_results(valid_ids, save_dir):

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
        for line in parsed_lines:
            frame, obj_id, x, y, w, h, conf = line
            if obj_id in valid_ids:
                if not final_lines or frame != final_lines[-1][0]:
                    final_lines.append([frame, x, y, x + w, y + h])
                else:
                    final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]
    return final_lines


def write_results(filename, results):
    with open(filename, "a", encoding="UTF-8") as f:
        f.writelines(results)


def bbox_scale_up(x_min, y_min, x_max, y_max, height, width, scale):
    w = y_max - y_min
    h = x_max - x_min
    x_min -= h // scale
    y_min -= w // scale
    x_max += h // scale
    y_max += w // scale

    if x_min < 0:
        x_min = 0

    if y_min < 0:
        y_min = 0

    if x_max > width:
        x_max = width

    if y_max > height:
        y_max = height

    return int(x_min), int(y_min), int(x_max), int(y_max)


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


# mask generator v3 (using padding)
def mask_generator_v3(x_min, y_min, x_max, y_max, level=10, step=3):
    w = x_max - x_min
    h = y_max - y_min

    n_w = w - level * 2 * step
    n_h = h - level * 2 * step

    if n_w <= 0 or n_h <= 0:
        mask = np.ones(shape=(h, w), dtype=np.float16)
        mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
        return mask, 1 - mask

    mask = np.ones(shape=(n_h, n_w), dtype=np.float16)

    for i in range(level):
        const = 1 - (1 / level * (i + 1))
        mask = np.pad(
            mask, ((step, step), (step, step)), "constant", constant_values=const
        )
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


def extract_feature(target_path, save_dir):
    mtcnn = MTCNN()
    img = Image.open(target_path)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    img_cropped = mtcnn(img, save_path=str(save_dir) + "/target_detect.png")
    img_embedding = resnet(img_cropped.unsqueeze(0))


def detect(save_img=False):

    start_time_total = time.time()

    source = (
        "/opt/ml/final-project-level3-cv-07/models/track/assets/" + opt.project + ".mp4"
    )
    target_path = (
        "/opt/ml/final-project-level3-cv-07/models/track/target/" + opt.target + ".jpeg"
    )
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

    # Directories
    save_dir = Path(
        increment_path("runs" / Path(opt.project) / opt.name, exist_ok=False)
    )  # increment run
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
    results_temp = defaultdict(dict)
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
        results = []
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
                        f"{frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                    results_temp[tid][frame] = tlbr
                    if save_results:
                        write_results(os.path.join(save_dir, "results.txt"), results)

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

    frame_list = get_frame(source)
    tracklet_dir = str(save_dir) + "/tracklet"
    targeted_ids, valid_ids = get_valid_tids(
        tracker,
        results_temp,
        frame_list,
        tracklet_dir,
        str(save_dir) + "/target_detect.png",
    )

    if save_results:
        write_results(
            os.path.join(save_dir, "valid_ids.txt"), "targeted tracklet ids:\n"
        )
        for id in targeted_ids:
            write_results(os.path.join(save_dir, "valid_ids.txt"), str(id) + " ")

        write_results(
            os.path.join(save_dir, "valid_ids.txt"), "\n\ncartoonized tracklet ids:"
        )
        for i, id in enumerate(valid_ids):
            if i % 15 == 0:
                write_results(os.path.join(save_dir, "valid_ids.txt"), "\n")
            write_results(os.path.join(save_dir, "valid_ids.txt"), str(id) + " ")

    final_lines = parsing_results(valid_ids, save_dir)
    save_face_swapped_vid(final_lines, save_dir, fps)

    end_time_total = time.time()

    print(f"Total Time Elapsed : {end_time_total - start_time_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default="chim",
        help="name of video project and save results to project/name",
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="/opt/ml/final-project-level3-cv-07/models/track/pretrained/yolov7-tiny.pt",
        help="model.pt path(s)",
    )
    parser.add_argument("--target", default="chim", help="name of the target image")
    parser.add_argument(
        "--img-size", type=int, default=1920, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.09, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.7, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--sim-thres",
        type=float,
        default=0.39,
        help="Similarity threshold for face matching",
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--trace", action="store_true", help="trace model")
    parser.add_argument(
        "--hide-labels-name", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument("--save_results", default=True)
    parser.add_argument(
        "--save-txt-tidl",
        action="store_true",
        help="save results to *.txt in tidl format",
    )
    parser.add_argument("--kpt-label", type=int, default=5, help="number of keypoints")
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )

    # tracking args
    parser.add_argument(
        "--track_high_thresh",
        type=float,
        default=0.3,
        help="tracking confidence threshold",
    )
    parser.add_argument(
        "--track_low_thresh",
        default=0.05,
        type=float,
        help="lowest detection threshold",
    )
    parser.add_argument(
        "--new_track_thresh", default=0.4, type=float, help="new track thresh"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.7,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--min_box_area", type=float, default=10, help="filter out tiny boxes"
    )
    parser.add_argument(
        "--fuse-score",
        dest="mot20",
        default=False,
        action="store_true",
        help="fuse score and iou for association",
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )

    # CMC
    parser.add_argument(
        "--cmc-method",
        default="sparseOptFlow",
        type=str,
        help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc",
    )

    # ReID
    parser.add_argument(
        "--with-reid",
        dest="with_reid",
        default=False,
        action="store_true",
        help="with ReID module.",
    )
    parser.add_argument(
        "--fast-reid-config",
        dest="fast_reid_config",
        default=r"fast_reid/configs/MOT17/sbs_S50.yml",
        type=str,
        help="reid config file path",
    )
    parser.add_argument(
        "--fast-reid-weights",
        dest="fast_reid_weights",
        default=r"pretrained/mot17_sbs_S50.pth",
        type=str,
        help="reid config file path",
    )
    parser.add_argument(
        "--proximity_thresh",
        type=float,
        default=0.5,
        help="threshold for rejecting low overlap reid matches",
    )
    parser.add_argument(
        "--appearance_thresh",
        type=float,
        default=0.25,
        help="threshold for rejecting low appearance similarity reid matches",
    )

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
