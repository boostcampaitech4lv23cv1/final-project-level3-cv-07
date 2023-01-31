from fastapi import FastAPI, Request
import uvicorn
import requests

import os
import re
import asyncio
import aiohttp

from pymongo import MongoClient

client = MongoClient()

db = client["cafe"]
db.drop_collection("env")

collection = db["env"]

base = os.getcwd()
req = requests.get("http://ipconfig.kr")
server_ip = re.search(r'IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', req.text)[1]

database_info = {
    "name": "database",
    "dir": f"{base}/database"
}

cartoonize_info = {
    "name": "cartoonize",
    "dir": f"{base}/models/track/cartoonize",
    "url": f"http://{server_ip}:30003",
}

track_info = {
    "name": "track",
    "dir": f"{base}/models/track",
    "url": f"http://{server_ip}:30004",
}

collection.insert_one(database_info)
collection.insert_one(cartoonize_info)
collection.insert_one(track_info)

# FastAPI 객체 생성
app = FastAPI()

async def inference():
    async def task(session, url):
        async with session.get(url) as response:
            return await response.text()

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(task(session, f"{cartoonize_info['url']}/cartoonize"), task(session, f"{track_info['url']}/track"))

@app.post("/req_infer")
async def read_root(req: Request):
    data = await req.body()
    file = open(f"{database_info['dir']}/uploaded_video/video.mp4", "wb")
    file.write(data)
    file.close()
    
    asyncio.create_task(inference())
    
    return 200


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=30002, reload=True, access_log=False)
