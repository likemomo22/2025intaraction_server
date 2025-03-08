import asyncio
import json
from queue import Empty
import sys
from threading import Lock, Thread
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import mediapipe as mp
import os
from contextlib import asynccontextmanager

# 适配 Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
target_file_directory = os.path.abspath(
    os.path.join(os.path.dirname(current_file_path), "../2025_ver3_forU3d/motion_kuazhang/device")
)
sys.path.append(target_file_directory)

from motion_kuazhang import data_queue, exampleAcquisition, stop_flag

# 初始化摄像头和线程
cap = None
cap_lock = Lock()
bluetooth_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI 应用启动")
    yield  # 这里会等待应用运行
    print("FastAPI 应用关闭")
    release_camera()  # 释放摄像头资源

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoints(websocket: WebSocket):
    print("[Startup] 启动蓝牙数据采集线程")
    stop_flag.clear()
    print("stop_flag's status: ", stop_flag.is_set())

    global bluetooth_thread
    bluetooth_thread = Thread(target=exampleAcquisition, args=(None,), daemon=True)
    bluetooth_thread.start()

    open_camera()
    await websocket.accept()

    try:
        async for data in getLandmarkAndVideo():
            await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        stop_flag.set()
        release_camera()

async def getLandmarkAndVideo():
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.resize(image, (256, 256))
            image = cv2.flip(image, 0)

            # 视频数据编码为 JPEG 格式
            _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            video_data = buffer.tobytes()

            # 获取 Landmark 数据
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            landmarks = []
            if result.pose_landmarks:
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        landmarks.append({"idx": idx, "x": landmark.x, "y": landmark.y, "z": landmark.z})

            # 从蓝牙数据队列中获取数据
            try:
                emgDatas = data_queue.get_nowait()
            except Empty:
                emgDatas = None

            combine_data = {
                "landmarks": landmarks,
                "emgDatas": emgDatas if emgDatas is not None else [0.0, 0.0, 0.0],
                "video": video_data.hex()
            }
            yield combine_data
            await asyncio.sleep(0.01 if not data_queue.empty() else 0.03)

def open_camera():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot access camera")
            else:
                print("Camera opened")

def release_camera():
    global cap
    with cap_lock:
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released")
