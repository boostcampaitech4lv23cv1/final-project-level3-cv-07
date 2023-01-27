from fastapi import FastAPI, Request
import uvicorn
import os

# FastAPI 객체 생성
app = FastAPI()

base = os.getcwd()
file_storage = "database"
track_dir = "models/track"
cartoonize_dir = "models/track/cartoonize"

# 라우터 '/'로 접근 시 {Hello: World}를 json 형태로 반환
@app.get("/track")
async def req_track():
    from tools.mc_demo_yolov7 import detect
    class Opt:
        weights= f"{track_dir}/pretrained/yolov7-tiny.pt"
        source = f"{file_storage}/uploaded_video/video.mp4"
        target = f"{file_storage}/target/chim.jpeg"
        cartoon = f"{track_dir}/assets/chim_cartoonized.mp4"
        img_size = 1920
        conf_thres= 0.09
        iou_thres= 0.7
        sim_thres= 0.35
        device= "0"
        view_img= None
        save_txt= None
        nosave= None
        classes = None
        agnostic_nms= True
        augment= None
        update= None
        project= f"{track_dir}/runs/detect"
        name= "exp"
        exist_ok= None
        trace= None
        hide_labels_name= False
        save_results = True
        save_txt_tidl = None
        kpt_label = 5
        hide_labels = False
        hide_conf = False,
        line_thickness = 3
        
        # Tracking args
        track_high_thresh = 0.3
        track_low_thresh = 0.05
        new_track_thresh = 0.4
        track_buffer = 30
        match_thresh = 0.7
        aspect_ratio_thresh = 1.6
        min_box_area = 10
        mot20 = True
        save_crop = None
        
        #CMC
        cmc_method = "sparseOptFlow"
        
        #ReID
        with_reid = False
        fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
        fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        jde= False
        ablation= False
    
    opt = Opt
    detect(opt)
    return 200

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30004, reload=True, access_log=False)
