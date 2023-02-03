
import uvicorn
import requests
import os
import re
import asyncio
import aiohttp
from fastapi import FastAPI, Request

# DB에 Environment 변수들 저장
from pymongo import MongoClient

client = MongoClient()

db = client["cafe"]
db.drop_collection("env")

collection = db["env"]

base = os.getcwd()
req = requests.get("http://ipconfig.kr")
server_ip = re.search(r"IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", req.text)[1]

base_info = {
    "name": "base",
    "dir": base,
    "ip": server_ip,
}

database_info = {
    "name": "database",
    "dir": f"{base}/database"
}

backend_info = {
    "name": "backend",
    "dir": f"{base}/backend",
    "url": f"http://{server_ip}:30002",
}

cartoonize_info = {
    "name": "cartoonize",
    "dir": f"{base}/models/cartoonize",
    "url": f"http://{server_ip}:30003",
}

track_info = {
    "name": "track",
    "dir": f"{base}/models/track",
    "url": f"http://{server_ip}:30004",
}

collection.insert_one(base_info)
collection.insert_one(database_info)
collection.insert_one(backend_info)
collection.insert_one(cartoonize_info)
collection.insert_one(track_info)

from utils.convert import *
from utils.general import *

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

# Client의 업로드 비디오 database/uploaded_video/video.mp4로 저장
@app.post("/upload/video")
async def upload_video(req: Request):
    video = await req.body()
    file = open(f"{database_info['dir']}/uploaded_video/video.mp4", "wb")
    file.write(video)
    file.close()
    
# Client의 업로드 이미지 database/target/target.jpeg로 저장
@app.post("/upload/image")
async def upload_image(req: Request):
    image = await req.body()
    file = open(f"{database_info['dir']}/target/target.jpeg", "wb")
    file.write(image)
    file.close()

# 모델 비동기 호출 요청 *(Lock)
async def inference():  
    async def task(session, url):
        async with session.get(url) as response:
            return await response.text()

    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(task(session, f"{cartoonize_info['url']}/cartoonize"), task(session, f"{track_info['url']}/track"))

# Client 인퍼런스 호출 시 cartoonize, track 적용 후 병합
@app.get("/req_infer")  
def request_inferences():
    db["track_info"].drop()
    remove_if_exists(opt.work_dir)
    
    # 기초 directory 생성
    create_directory(f"{opt.work_dir}/tracklet")
    create_directory(f"{opt.work_dir}/image_orig")
    create_directory(f"{opt.work_dir}/image_cart")
        
    # Cartoonize, Track 모델 비동기 호출 요청
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cartoonized, tracked = loop.run_until_complete(inference())
    loop.close()
    
    # Partial Cartoonize 진행
    valid_ids, num_frames, fps = eval(tracked)
    valid_ids = {int(x): valid_ids[x] for x in valid_ids}
    
    track_info = db['track_info'].find({}, {"_id": False})  # Tracklet 정보 수집
    
    final_lines = parsing_results(track_info, valid_ids, num_frames, opt.swap_all_face)
    save_face_swapped_vid(final_lines, opt.work_dir, fps)
    
    return 200

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30002, reload=True, access_log=False)
