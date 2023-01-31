from fastapi import FastAPI, Request
import uvicorn
import requests

import re
import asyncio
import aiohttp

req = requests.get("http://ipconfig.kr")
server_ip = re.search(r"IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", req.text)[1]

cartoonize_url = "http://115.85.182.51:30003"
track_url = "http://115.85.182.51:30004"

# FastAPI 객체 생성
app = FastAPI()


async def inference():
    async def task(session, url):
        async with session.get(url) as response:
            return await response.text()

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            task(session, f"{cartoonize_url}/cartoonize"),
            task(session, f"{track_url}/track"),
        )


@app.post("/req_infer")
async def read_root(req: Request):
    data = await req.body()
    file = open("database/uploaded_video/video.mp4", "wb")
    file.write(data)
    file.close()

    asyncio.create_task(inference())

    return 200


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30002, reload=True, access_log=False)
