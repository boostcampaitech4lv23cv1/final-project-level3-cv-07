import argparse
import time
from pathlib import Path
import sys
import os

import cv2
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolov7.detect_temp import detect_image
from fast_reid.fast_reid_interfece import FastReIDInterface

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from deepface import DeepFace

sys.path.insert(0, './yolov7')
sys.path.append('.')

def write_results(filename, results):

    with open(filename, 'a') as f:
            f.writelines(results)

def bbox_scale_up(y_min, x_min, y_max, x_max, height, width, scale):
    w = x_max - x_min
    h = x_max - x_min
    y_min -= h//scale
    x_min -= w//scale
    y_max += h//scale
    x_max += w//scale

    if y_min < 0:
        y_min = 0

    if x_min < 0:
        x_min = 0

    if y_max > height:
        y_max = height

    if x_max > width:
        x_max = width
    
    return y_min, x_min, y_max, x_max


def detect(save_img=False):

    start_time_total = time.time()
    
    target_feature = np.asarray(DeepFace.represent(opt.target)) # yolov7/detect_temp.py의 함수를 그대로 사용하였음

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_results = opt.save_results
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace: # False
        model = TracedModel(model, device, opt.img_size)

    if half: # FP 16
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride) # 설정한 video load : img_size, nframes

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()
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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        results = []

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det): # detection이 존재하면!
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy() # [[bbox1],[bbox2]..]
                detections = det.cpu().numpy()
                detections[:, :4] = boxes #

            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh # `(top left x, top left y, width, height)`
                tlbr = t.tlbr # `(min x, min y, max x, max y)`
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
                    if save_results:
                        write_results(os.path.join(save_dir,'results.txt'),results)
                    
                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
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
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")
        
    print(f'Done. ({time.time() - t0:.3f}s)')

    cap = cv2.VideoCapture(opt.source)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []

    while(True):
        ret, cur_frame = cap.read()
        if cur_frame is None: break

        frame_list.append(cur_frame)
    ''' 
    각각의 tracker에서 대표 feature를 뽑고 similarity 계산하기
    1. tracker status에 대한 설명
    - tracker.tracked_stracks : 현재 frame에서 tracking이 이어지고 있는 tracker instance
    - tracker.removed_stracks : tracking이 종료된 tracker instance 
    2. TODO 
    - 유효한 tracker로 인정하기 위한 최소 frame은 몇으로 잡을지 결정
    - 유효한 tracker에서 feature는 어떻게 뽑을지 결정
        - tracker.curr_feat : 가장 최근 bbox의 feature  
        - tracker.smooth_feat : 매 frame moving average한 feature
        - DeepFace.represent("crop img") : 각 tracker마다 대표 bbox를 선정하고 api를 호출 
    3. FIXME
    - frame_list에 모든 프레임 정보 저장하지 않고, 뒤에서 필요한 frame만 cv.imread로 불러오기
    '''

    test = {} # 딕셔너리 생성 --> {t_id : feature vector, t_id : feature_vector...}

    # 영상이 끝난 시점에 tracking 하고 있던 tracker들이 자동으로 removed_stracks로 status가 전환되지 않기 때문에
    # 영상이 끝난 시점에서 tracking을 하고 있었던 tracker와 과거에 tracking이 끝난 tracker들 모두를 관리 해야합니다. 

    # 과거 종료된 tracker들 중에서
    for i in tracker.removed_stracks:
        if i.tracklet_len > 5 : # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다. 
            xywh = i.xywh
            x1,y1,x2,y2 = float(xywh[0]),float(xywh[1]),float(xywh[0])+float(xywh[2]),float(xywh[1])+float(xywh[3]) # [x,y,w,h]--> [x1,y1,x2,y2]
            # test[i.track_id] = i.smooth_feat
            test[i.track_id] = np.array(DeepFace.represent(frame_list[i.end_frame-1][int(y1):int(y2),int(x1):int(x2),:], enforce_detection=False))
    
    # 영상이 끝난 시점에 살아있던 tracker에 대해서 
    for i in tracker.tracked_stracks:
        if i.tracklet_len > 5 : # 일단 5 프레임 이상 이어졌던 tracker에 대해서만 유효하다고 판단하고 feature를 뽑았습니다.
            xywh = i.xywh
            x1,y1,x2,y2 = float(xywh[0]),float(xywh[1]),float(xywh[0])+float(xywh[2]),float(xywh[1])+float(xywh[3]) # [x,y,w,h]--> [x1,y1,x2,y2]
            # test[i.track_id] = i.smooth_feat
            test[i.track_id] = np.array(DeepFace.represent(frame_list[i.end_frame-1][int(y1):int(y2),int(x1):int(x2),:], enforce_detection=False))
    
    print("Similairties list: ")        
    print(cdist(target_feature.reshape(1,target_feature.size), list(test.values()), metric="euclidean"))
    print(f"Similarity Threshold : {opt.sim_thres}")
    sim = cdist(target_feature.reshape(1,target_feature.size), list(test.values()), metric="euclidean") > opt.sim_thres # distance가 1 이상인 (즉, 비슷하지 않은) tracker 찾기
    t_ids = np.asarray(list(test.keys())) 
    valid_ids = t_ids[sim[0]] # key에 넣어서 해당 tracker ID만을 뽑아내기
    
    # frame_bbox = defaultdict(set)
    # with open(f"{save_dir}/results.txt", "r") as f: # 프레임 단위로 저장된 t_id, bbox 정보
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split(',')
    #         if int(line[1]) in valid_ids: # 만약 위에서 선정한 tracker에 포함된다면
    #             frame_bbox[int(line[0])].add(line[2:6]) # frame_bbox라는 dictionary에 bbox들을 저장
    #             # frame_bbox --> frame : [[bbox1],[bbox2],[bbox3]...]
    #     for k,v in frame_bbox.items() :
    #         frame_bbox[k] = list(v)    
    
    ''' 
    앞 단계에서 넘어오는 정보 
    1. Similarity 계산이 끝난 후, 우리가 cartoonize해야 할 bbox들이 선정되어 frame_bbox 라는 dictionary로 넘어옴
    2. frame_bbox --> {frame : [[bbox1],[bbox2],[bbox3]...], frame : [[bbox1],[bbox2],[bbox3]...]}
    3. bbox = [x,y,w,h] 
    '''
    
    with open(os.path.join(save_dir,'results.txt'), 'r') as f:
        lines = f.readlines()

        parsed_lines = [] 
        
        # Remove unnecessary info and casting data type (str -> int)
        for line in lines:
            line = line.split(",")[:-3]
            line = list(map(int, list(map(float, line))))
            parsed_lines.append(line)    

        # Remove redundant info and soted by frame > obj_id
        parsed_lines = sorted(list(map(list, set(list(map(tuple, parsed_lines))))), key=lambda x : (x[0], x[1]))

        # Summary info (per frame)
        final_lines = []
        for line in parsed_lines:
            frame, obj_id, x, y, w, h, conf = line
            if obj_id in valid_ids:
                if not final_lines or frame != final_lines[-1][0]:
                    final_lines.append([frame, x, y, x+w, y+h])
                else:
                    final_lines[-1] = final_lines[-1]+[x, y, x+w, y+h]

        # ===========================================================================
    
    frame_array = []
    
    ## FIXME 
    img = cv2.imread(f'/opt/ml/BoT-SORT/cartoonize/image_cart/frame_1.png')
    height, width, layers = img.shape
    size = (width,height)
    
    # face swap per frame
    for line in tqdm(final_lines):
        assert (len(line)-1) % 4 == 0
        frame_idx = line[0] # Image Index starts from 0 
        orig_img = cv2.imread(f'/opt/ml/BoT-SORT/cartoonize/image_orig/frame_{frame_idx}.png')
        cart_img = cv2.imread(f'/opt/ml/BoT-SORT/cartoonize/image_cart/frame_{frame_idx}.png')
        face_swapped_img = orig_img

        for i in range(((len(line)-1) // 4)-1):
            y_min, x_min, y_max, x_max = bbox_scale_up(line[4*i+1], line[4*i+2], line[4*i+3], line[4*i+4], height, width, 2)
            face_swapped_img[x_min:x_max, y_min:y_max] = cart_img[x_min:x_max, y_min:y_max]
        
        frame_array.append(face_swapped_img)

    out = cv2.VideoWriter(os.path.join(save_dir,'face_swapped_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in tqdm(range(len(frame_array))):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    end_time_total = time.time()

    print(f"Total Time Elapsed : {end_time_total - start_time_total}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/opt/ml/BoT-SORT/pretrained/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/opt/ml/BoT-SORT/assets/chim.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--target',default="/opt/ml/BoT-SORT/target/chim.jpeg",help='path of the target image')
    parser.add_argument('--cartoon',default="/opt/ml/BoT-SORT/assets/chim_cartoonized.mp4",help='path of the target image')
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--sim-thres', type=float, default=0.35, help='Similarity threshold for face matching')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save_results',default=True)
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
