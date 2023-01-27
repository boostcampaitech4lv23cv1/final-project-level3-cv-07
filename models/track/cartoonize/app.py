from fastapi import FastAPI, Request
import uvicorn

import os
import time
from Cartoonize import save_vid_2_img, cartoonize

# FastAPI 객체 생성
app = FastAPI()

file_storage = "database"
track_dir = "models/track"
cartoonize_dir = "models/track/cartoonize"

# 라우터 '/'로 접근 시 {Hello: World}를 json 형태로 반환
@app.get("/cartoonize")
async def req_inference():    
    model_path = 'models/track/cartoonize/saved_models'
    load_folder = 'models/track/cartoonize/image_orig'
    save_folder = 'models/track/cartoonize/image_cart'
    input_video = 'database/uploaded_video/video.mp4'
    
    
    if not os.path.exists(load_folder):
        os.mkdir(load_folder)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_vid_2_img(input_video,load_folder)

    s = time.time()
    cartoonize(load_folder, save_folder, model_path)
    e = time.time()
    print(f"Total elapsed time: {e-s}") 
    
    
    return 200

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30003, reload=True, access_log=False)
