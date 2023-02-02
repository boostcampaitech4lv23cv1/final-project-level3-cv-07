from pymongo import MongoClient

client = MongoClient()

db = client["cafe"]
collection = db['env']

base_info = collection.find_one({'name': 'base'})
database_info = collection.find_one({'name': 'database'})
backend_info = collection.find_one({'name': 'backend'})
cartoonize_info = collection.find_one({'name': 'cartoonize'})
track_info = collection.find_one({'name': 'track'})

import sys
import uvicorn
import time
from fastapi import FastAPI

from Cartoonize import save_vid_2_img, cartoonize

sys.path.append(base_info['dir'])

class Opt:
        weights= f"{track_info['dir']}/pretrained/yolov7-tiny.pt"
        source = f"{database_info['dir']}/uploaded_video/video.mp4"
        target = f"{database_info['dir']}/target/target.jpeg"
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
        work_dir= f"{database_info['dir']}/work_dir"
        name = "exp"
        exist_ok = None
        save_results = True
        save_txt_tidl = None
        kpt_label = 5
        hide_conf = (False,)
        line_thickness = 3
        save_img = True
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
        swap_all_face = False 
        verbose = False
        # ReID
        with_reid = False
        fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
        fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        jde = False
        ablation = False
opt = Opt()

# FastAPI 객체 생성
app = FastAPI()

@app.get("/cartoonize")
def req_inference():
    model_path =f"{cartoonize_info['dir']}/saved_models"
    load_dir = f"{opt.work_dir}/image_orig"
    save_dir = f"{opt.work_dir}/image_cart"
    input_video = f"{database_info['dir']}/uploaded_video/video.mp4"

    save_vid_2_img(input_video, load_dir)

    s = time.time()
    cartoonize(load_dir, save_dir, model_path)
    e = time.time()
    print(f"Total elapsed time: {e-s}")
    
    
    # requests.get(f"{backend_info['url']}/signal/end_cartoonize")
    
    return 200

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30003, reload=True, access_log=False)
