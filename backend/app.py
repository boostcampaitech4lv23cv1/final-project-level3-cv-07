from fastapi import FastAPI
import uvicorn

# FastAPI 객체 생성
app = FastAPI()

# 라우터 '/'로 접근 시 {Hello: World}를 json 형태로 반환
@app.get("/")
def read_root():
  return {"Hello": "World"}

uvicorn.run(app, host="0.0.0.0", port=8000)