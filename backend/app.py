from fastapi import FastAPI, Request
import uvicorn

# FastAPI 객체 생성
app = FastAPI()

# 라우터 '/'로 접근 시 {Hello: World}를 json 형태로 반환
@app.post("/save_video")
async def read_root(req: Request):
    data = await req.body()
    file = open("database/uploaded_video/video.mp4", "wb")
    file.write(data)
    return 200

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30002)
