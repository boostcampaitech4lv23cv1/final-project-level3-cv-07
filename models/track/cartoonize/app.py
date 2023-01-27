from fastapi import FastAPI, Request
import uvicorn

import os
import time
from Cartoonize import save_vid_2_img, cartoonize, createDirectory

# FastAPI 객체 생성
app = FastAPI()

file_storage = "database"
track_dir = "models/track"
cartoonize_dir = "models/track/cartoonize"

# 라우터 '/'로 접근 시 {Hello: World}를 json 형태로 반환
@app.get("/cartoonize")
async def req_inference():
    class Opt:
        project = "chim"
    
    opt = Opt
    
    model_path ="/opt/ml/final-project-level3-cv-07/models/track/cartoonize/saved_models"
    load_dir = f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_orig"
    save_dir = f"/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs/{opt.project}/image_cart"
    input_video = f"/opt/ml/final-project-level3-cv-07/models/track/assets/{opt.project}.mp4"
    run_dir = "/opt/ml/final-project-level3-cv-07/models/track/cartoonize/runs"

    createDirectory(run_dir)
    createDirectory(load_dir)
    createDirectory(save_dir)

    save_vid_2_img(input_video, load_dir)

    s = time.time()
    cartoonize(load_dir, save_dir, model_path)
    e = time.time()
    print(f"Total elapsed time: {e-s}")
    
    
    return 200

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30003, reload=True, access_log=False)