import time
from pathlib import Path
import sys
import os
from collections import defaultdict
import glob

import cv2
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
from numpy import random
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from PIL import Image
from facenet_pytorch import MTCNN

# import face_recognition

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    strip_optimizer,
    set_logging,
    increment_path,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
)
from yolov7.detect_temp import detect_image

from tracker.mc_bot_sort import BoTSORT
from fast_reid.fast_reid_interfece import FastReIDInterface

from tools.mask_generator import (
    mask_generator_v0,
    mask_generator_v1,
    mask_generator_v2,
    mask_generator_v3,
)
from tools.utils import createDirectory, get_frame_num, bbox_scale_up

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


def dbscan(target_dir, tracklet_dir):
    tracklet_imgs = glob.glob(tracklet_dir + "/*.png")
    # encodings = [DeepFace.represent(img_path=img,enforce_detection=False,model_name="Facenet512") for img in tracklet_imgs]
    data = []
    for imagePath in tracklet_imgs:
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [
            {"imagePath": imagePath, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)
        ]
        data.extend(d)
    encodings = [d["encoding"] for d in data]
    # dump the facial encodings data to disk
    stime = time.time()
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)
    etime = time.time()
    print(f"DBSCAN time elapsed :{etime - stime}")
    label_ids = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(label_ids > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    return


def get_valid_tids(tracker, results, tracklet_dir, target_dir, min_length, conf_thresh):

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
        f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig/frame_1.png"
    )
    height, width, layers = img.shape
    size = (width, height)

    createDirectory(tracklet_dir)
    tracks = list(
        set(tracker.removed_stracks + tracker.tracked_stracks + tracker.lost_stracks)
    )
    for i in tracks:
        if (
            i.tracklet_len > min_length
        ):  # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            frame, value = sorted(results[i.track_id].items(), key=lambda x: x[1][4])[
                -1
            ]
            x1, y1, x2, y2, conf = value
            if conf > conf_thresh:
                sx1, sy1, sx2, sy2 = bbox_scale_up(x1, y1, x2, y2, height, width, 3)
                frame_img = cv2.imread(
                    f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig/frame_{frame}.png"
                )
                cv2.imwrite(
                    f"{tracklet_dir}/{i.track_id}.png",
                    np.array(frame_img[int(sy1) : int(sy2), int(sx1) : int(sx2), :]),
                )

                t_ids[i.track_id] = conf

    # if opt.dbscan :
    #     dbscan(target_dir,tracklet_dir)
    #     return True
    else:
        dfs = DeepFace.find(
            img_path=target_dir,
            db_path=tracklet_dir,
            enforce_detection=False,
            model_name="VGG-Face",
        )

        targeted_ids = {}

        for i in range(len(dfs)):
            id = int(dfs.iloc[i].identity.split("/")[-1].split(".")[0])
            id_conf = t_ids.pop(id)
            targeted_ids[id] = id_conf

        return dict(
            sorted(targeted_ids.items(), key=lambda x: x[1], reverse=True)
        ), dict(sorted(t_ids.items(), key=lambda x: x[1], reverse=True))


def save_face_swapped_vid(final_lines, save_dir, fps, opt):
    ## FIXME
    img = cv2.imread(
        f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig/frame_1.png"
    )
    height, width, _ = img.shape
    size = (width, height)
    swap_s = time.time()

    frame_array = []
    # face swap per frame
    for line in tqdm(final_lines):
        assert (len(line) - 1) % 4 == 0
        frame_idx = line[0]  # Image Index starts from 1
        orig_img = cv2.imread(
            f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig/frame_{frame_idx}.png"
        )
        cart_img = cv2.imread(
            f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_cart/frame_{frame_idx}.png"
        )
        resized_cart_img = cv2.resize(cart_img, size, interpolation=cv2.INTER_LINEAR)
        face_swapped_img = orig_img
        for i in range(((len(line) - 1) // 4)):
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


def parsing_results(valid_ids, save_dir, num_frames, swap_all_face=False):

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

            # save all face (for debugging)
            if swap_all_face:
                if not final_lines or frame != final_lines[-1][0]:
                    final_lines.append([frame, x, y, x + w, y + h])
                else:
                    final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

            # save valid face
            else:
                if obj_id in valid_ids:
                    if not final_lines or frame != final_lines[-1][0]:
                        final_lines.append([frame, x, y, x + w, y + h])
                    else:
                        final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

        total_lines = []

        idx = 1
        for i in range(len(final_lines)):
            if idx < final_lines[i][0]:
                while idx < final_lines[i][0]:
                    total_lines.append([idx + 1])
                    idx += 1
            total_lines.append(final_lines[i])
            idx += 1

        while len(total_lines) < num_frames:
            total_lines.append([total_lines[-1][0] + 1])

        # for debugging
        """
        for line in total_lines:
            print(line[0], end=" / ")
            print(num_frames, end = " : ")
            print(line)
        """
    return total_lines


def write_results(filename, results):
    with open(filename, "a", encoding="UTF-8") as f:
        f.writelines(results)


def extract_feature(target_path, save_dir):
    mtcnn = MTCNN(margin=30)
    img = Image.open(target_path)
    img_cropped = mtcnn(img, save_path=str(save_dir) + "/target_detect.png")
    # resnet = InceptionResnetV1(pretrained="vggface2").eval()
    # img_embedding = resnet(img_cropped.unsqueeze(0))


def detect(opt, save_img=True):

    start_time_total = time.time()

    source = f"{file_storage}/uploaded_video/{opt.project}.mp4"
    target_path = (
        f"/opt/ml/final-project-level3-cv-07/models/track/target/{opt.target}.jpg"
    )
    weights, imgsz = (
        opt.weights,
        opt.img_size,
    )
    save_results = opt.save_results

    # Directories
    save_dir = Path(
        increment_path("runs" / Path(opt.project) / opt.name, exist_ok=False)
    )  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    extract_feature(target_path, save_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:  # FP 16
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Set Dataloader
    vid_path, vid_writer = None, None
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
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )

        # Process detections
        results = []
        for i, det in enumerate(pred):  # detections per image
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
                    results_temp[tid][frame] = np.append(tlbr, t.score)
                    if save_results:
                        write_results(os.path.join(save_dir, "results.txt"), results)

                    if save_img:  # Add bbox to image
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

            # Save results (image with detections)
            if save_img:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    img = cv2.imread(
                        f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig/frame_1.png"
                    )
                    h, w, _ = img.shape
                    vid_writer = cv2.VideoWriter(
                        save_path[:-4] + "_tracked.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                vid_writer.write(im0)

    print(f"Done. ({time.time() - t0:.3f}s)")

    tracklet_dir = str(save_dir) + "/tracklet"
    targeted_ids, valid_ids = get_valid_tids(
        tracker,
        results_temp,
        tracklet_dir,
        str(save_dir) + "/target_detect.png",
        opt.min_frame,
        opt.conf_thresh,
    )

    if save_results:
        write_results(
            os.path.join(save_dir, "valid_ids.txt"),
            "targeted tracklet ids (id : confidence)\n",
        )
        for id, conf in targeted_ids.items():
            write_results(
                os.path.join(save_dir, "valid_ids.txt"), f"{id} : {conf:.2f} \n"
            )

        write_results(
            os.path.join(save_dir, "valid_ids.txt"),
            "\ncartoonized tracklet ids (id : confidence)\n",
        )
        for id, conf in valid_ids.items():
            write_results(
                os.path.join(save_dir, "valid_ids.txt"), f"{id} : {conf:.2f} \n"
            )

    num_frames = get_frame_num(source)
    final_lines = parsing_results(valid_ids, save_dir, num_frames)
    save_face_swapped_vid(final_lines, save_dir, fps, opt)

    end_time_total = time.time()

    print(f"Total Time Elapsed : {end_time_total - start_time_total}")


if __name__ == "__main__":
    file_storage = "../../database"
    track_dir = "."
    cartoonize_dir = f"cartoonize"

    class Opt:
        weights = f"{track_dir}/pretrained/yolov7-tiny.pt"
        source = f"{file_storage}/uploaded_video/resized_1000_1299_1080p.mp4"
        target = f"HanniPham"
        cartoon = f"{track_dir}/assets/chim_cartoonized.mp4"
        img_size = 1920
        conf_thres = 0.09
        iou_thres = 0.7
        sim_thres = 0.35
        device = "0"
        nosave = None
        classes = None
        agnostic_nms = True
        augment = None
        update = None
        project = f"resized_1000_1299_1080p"
        name = "exp"
        exist_ok = None
        save_results = True
        save_txt_tidl = None
        kpt_label = 5
        hide_conf = (False,)
        line_thickness = 3

        # Tracking args
        track_high_thresh = 0.3
        track_low_thresh = 0.05
        new_track_thresh = 0.4
        track_buffer = 30
        match_thresh = 0.7
        conf_thresh = 0.7  # added
        aspect_ratio_thresh = 1.6
        min_box_area = 10
        min_frame = 5  # added
        dbscan = False  # added
        mot20 = True
        save_crop = None

        # CMC
        cmc_method = "sparseOptFlow"

        # ReID
        with_reid = False
        fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
        fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        jde = False
        ablation = False

    opt = Opt

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
