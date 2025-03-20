import asyncio
import base64
import json
from queue import Empty
import sys
from threading import Lock, Thread
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import mediapipe as mp
import os
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd

# 适配 Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)

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
    stop_flag.set()
    if bluetooth_thread and bluetooth_thread.is_alive():
        bluetooth_thread.join()  # 等待线程结束
    release_camera()  # 释放摄像头资源

app = FastAPI(lifespan=lifespan)

@app.websocket("/sendUserId")
async def sendUserId_endpoint(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] sendUserId")
    await websocket.accept()
    
    try:
        data = await websocket.receive_text()
        data_dict = json.loads(data)
        print(data_dict)
        if data_dict.get("type") == "userId":
            global user_id
            user_id = data_dict["userId"]
            print(f"✅ 收到用户 ID: {user_id}")
            
            while True:
                # 发送确认信息
                await websocket.send_text(json.dumps({"status": "success", "message": f"用户 {user_id} 已注册"}))
                # **等待 Unity 发送 "close" 消息后再关闭**
                close_message = await websocket.receive_text()
                print(f"📨 收到消息: {close_message}")

                if close_message == "close":
                    print(f"🔌 Unity 端请求关闭 WebSocket (用户 ID: {user_id})")
                    await websocket.close()
                    print("🔌 WebSocket sendUserId 连接已关闭")
                    print("-----------------------------------")
                    break
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1001, reason="Server error or disconnect")
                print("🔌 WebSocket sendUserId 连接已关闭")
                print("-----------------------------------")
            except Exception as close_error:
                print(f"❌ 在关闭 WebSocket 时发生错误: {close_error}")


@app.websocket("/getMaxValue")
async def getMaxValue_endpoints(websocket:WebSocket):
    print("-----------------------------------")
    print("[Startup] get max value")
    stop_flag.clear()

    global is_get_max_value
    is_get_max_value=True
    
    filename = os.path.join(current_dir, f"maxValue_ID{user_id}_getMaxValue.csv")

    global bluetooth_thread
    bluetooth_thread = Thread(target=exampleAcquisition, args=(None, "BTH00:07:80:89:7F:C5", 500, 1000, 0x01, filename), daemon=True)
    bluetooth_thread.start()

    await websocket.accept()
    
    #监听客户端的关闭请求
    receive_task=asyncio.create_task(receive_close_request(websocket))

    try:
        async for data in getMaxValue():
            await websocket.send_text(json.dumps(data))
            
            #检查是否受到关闭信号
            if receive_task.done():
                await websocket.close()
                print("🔌 WebSocket getMaxValue 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                is_get_max_value = False
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await clear_data_queue()
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1001, reason="Server error or disconnect")
                print("🔌 WebSocket getMaxValue 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                is_get_max_value = False
            except Exception as close_error:
                print(f"❌ 在关闭 WebSocket 时发生错误: {close_error}")

        
async def receive_close_request(websocket:WebSocket):
    try:
        while True:
            close_message = await websocket.receive_text()
            print(f"📨 收到消息: {close_message}")

            if close_message == "close":
                print(f"🔌 Unity 端请求关闭 WebSocket (用户 ID: {user_id})")
                return
    except WebSocketDisconnect:
        print("❌ 客户端已断开连接")
    except Exception as e:
        print(f"❌ 监听关闭请求时发生错误: {e}")
        
    
async def getMaxValue():
    # 从蓝牙数据队列中获取数据
    global is_get_max_value
    while is_get_max_value:
        try:
            emgDatas = data_queue.get_nowait()
        except Empty:
            emgDatas = None

        combine_data = {
            "emgDatas": emgDatas if emgDatas is not None else [1.0, 1.0, 1.0],
        }
        yield combine_data
        await asyncio.sleep(0.01 if not data_queue.empty() else 0.03)
        
        
        
@app.websocket("/setMaxValue")
async def setMaxValue_endpoints(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] read max value")
    await websocket.accept()   
    
    try:
        data = await websocket.receive_text()
        data_dict = json.loads(data)
        print(data_dict)
        if data_dict.get("type") == "userIdToGetMaxV":
            global userIdToGetMaxV
            userIdToGetMaxV = data_dict["userId"]
            print(f"✅ 收到用户 ID: {userIdToGetMaxV}")

            # 发送确认信息
            await websocket.send_text(json.dumps({"type":"confirmMessage","status": "success", "message": f"用户 {userIdToGetMaxV} 已填写最大值"}))

        results = setMaxValue()
        print(results)
        if results is not None:
            await websocket.send_text(json.dumps({"type":"data","data":results}))  # 发送数据给 Unity
        else:
            await websocket.send_text(json.dumps([7000,7000,7000]))
        
        # **等待 Unity 发送 "close" 消息后再关闭**
        await websocket.close()
        print("🔌 WebSocket setMaxValue 连接已关闭")
        print("-----------------------------------")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1001, reason="Server error or disconnect")
                print("🔌 WebSocket setMaxValue 连接已关闭")
                print("-----------------------------------")
            except Exception as close_error:
                print(f"❌ 在关闭 WebSocket 时发生错误: {close_error}")    
        
@app.websocket("/getVideoData")
async def getVideoData_endpoints(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] get video data")
    filename = os.path.join(current_dir, f"videoData_ID{user_id}_NoneFeedback.csv")
    stop_flag.clear()
    
    global bluetooth_thread
    bluetooth_thread = Thread(target=exampleAcquisition, args=(None, "BTH00:07:80:89:7F:C5", 500, 1000, 0x01, filename), daemon=True)
    bluetooth_thread.start()
    
    await websocket.accept()
    
    receive_task=asyncio.create_task(receive_close_request(websocket))
    
    try:
        async for data in getVideoOnly():
            await websocket.send_text(json.dumps(data))
            
            #检查是否受到关闭信号
            if receive_task.done():
                await websocket.close()
                print("🔌 WebSocket getVideoOnly 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await clear_data_queue()
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1001, reason="Server error or disconnect")
                print("🔌 WebSocket getVideoOnly 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                release_camera()
            except Exception as close_error:
                print(f"❌ 在关闭 WebSocket 时发生错误: {close_error}")
       

@app.websocket("/ws")
async def websocket_endpoints(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] 启动蓝牙数据采集线程")
    filename = os.path.join(current_dir, f"landmarkAndVideo_ID{user_id}_feedback.csv")
    stop_flag.clear()
    
    global bluetooth_thread
    bluetooth_thread = Thread(target=exampleAcquisition, args=(None, "BTH00:07:80:89:7F:C5", 500, 1000, 0x01, filename), daemon=True)
    bluetooth_thread.start()

    # open_camera()
    await websocket.accept()

    receive_task=asyncio.create_task(receive_close_request(websocket))

    try:
        async for data in getLandmarkAndVideo():
            await websocket.send_text(json.dumps(data))
            
            #检查是否受到关闭信号
            if receive_task.done():
                await websocket.close()
                print("🔌 WebSocket ws 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                break
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await clear_data_queue()
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1001, reason="Server error or disconnect")
                print("🔌 WebSocket ws 连接已关闭")
                print("-----------------------------------")
                stop_flag.set()
                release_camera()
            except Exception as close_error:
                print(f"❌ 在关闭 WebSocket 时发生错误: {close_error}")
       

@app.websocket("/calculateRMSRatio")
async def calculate_rms_ratio_endpoint(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] calculate RMS and ratio")
    await websocket.accept()   
    
    try:
        results = calculate_rms()
        if results is not None:
            await websocket.send_text(json.dumps(results))  # 发送数据给 Unity
        else:
            await websocket.send_text(json.dumps([0.8,0.7]))
            
        # **等待 Unity 发送 "close" 消息后再关闭**
        close_message = await websocket.receive_text()
        if close_message == "close":
            print(f"🔌 Unity 端请求关闭 WebSocket (用户 ID: {user_id})")
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()
    except Exception as e:
        await websocket.close()
        print(f"Unexpected error: {e}")
    finally:
        await websocket.close()
        await asyncio.sleep(0.05)  

@app.websocket("/calculateRMSNormalRatio")
async def calculate_rms_ratio_endpoint(websocket: WebSocket):
    print("-----------------------------------")
    print("[Startup] calculate RMS and ratio")
    await websocket.accept()   
    
    try:
        results = calculate_normal_rms()
        if results is not None:
            await websocket.send_text(json.dumps(results))  # 发送数据给 Unity
        else:
            await websocket.send_text(json.dumps([8,7]))
            
        # **等待 Unity 发送 "close" 消息后再关闭**
        close_message = await websocket.receive_text()
        if close_message == "close":
            print(f"🔌 Unity 端请求关闭 WebSocket (用户 ID: {user_id})")
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()
    except Exception as e:
        await websocket.close()
        print(f"Unexpected error: {e}")
    finally:
        await websocket.close()
        await asyncio.sleep(0.05)    
        
#--------------------清空队列--------------------
async def clear_data_queue():
    i=0
    while not data_queue.empty():
        try:
           data_queue.get_nowait()
           i+=1
        except Empty:
           break
    print(f"✅ Data queue cleared: {i} items")
        
def setMaxValue():
    #read "userIdToGetMaxV" 
    #write to "user_id"
    try:
        filename = os.path.join(current_dir, f"maxValue_ID{userIdToGetMaxV}_getMaxValue.csv")

        if not os.path.exists(filename):
            print("文件错误", f"未找到Excel文件: {filename}")
            return

        # 读取Excel文件
        df = pd.read_csv(filename)

        channels = ["Channel1", "Channel2", "Channel3"]

        results = {}

        for channel in channels:
            if channel in df.columns:
                max_values = df[channel].nlargest(100).reset_index(drop=True)
                min_values = df[channel].nsmallest(100).reset_index(drop=True)

                values = (abs(max_values - 32768) + abs(min_values - 32768)) / 2

                trimmed_values = values.iloc[5:-10]

                mean_value = trimmed_values.mean()
                results[channel] = mean_value

            else:
                results[channel] = "ch. not found"

        max_values_file = os.path.join(current_dir, f"maxValue_ID{user_id}_MaxValue.csv")
        max_values_df = pd.DataFrame([results])
        max_values_df.to_csv(max_values_file, index=False)
        
        return results
    except Exception as e:
        print("error", f"deal data with error: {e}")

async def getVideoOnly():
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    cap = cv2.VideoCapture(0)  # 打开摄像头

    try:
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.resize(image, (256, 256))
                image = cv2.flip(image, 0)

                _, buffer = cv2.imencode(".jpg", image)
                video_data = buffer.tobytes()

                # 仅传输视频数据
                yield {"video": video_data.hex()}

                await asyncio.sleep(0.01 if not data_queue.empty() else 0.03)
    finally:
        if cap.isOpened():
            cap.release()
            print("Camera released in getVideoOnly")
        await asyncio.sleep(0.1)

    # with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             print("Ignoring empty camera frame.")
    #             continue

    #         image = cv2.resize(image, (256, 256))
    #         image = cv2.flip(image, 0)

    #         # # 进行背景去除
    #         # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         # segmentation_result = selfie_seg.process(image_rgb)

    #         # mask = segmentation_result.segmentation_mask
    #         # threshold = 0.5  # 阈值，值越高人物边界越硬
    #         # mask = (mask > threshold).astype("uint8")  # 二值化
            
    #         # # 转换 BGR → BGRA（增加 Alpha 通道）
    #         # foreground = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            
    #         # # 设置 Alpha 通道（透明度）
    #         # foreground[:, :, 3] = (mask * 255).astype("uint8")

    #         # # 透明背景 PNG 编码
    #         # _, buffer = cv2.imencode(".png", foreground)
    #         _, buffer = cv2.imencode(".jpg", image)
    #         video_data = buffer.tobytes()

    #         # 仅传输视频数据
    #         yield {"video": video_data.hex()}

    #         await asyncio.sleep(0.03)

async def getLandmarkAndVideo():
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    cap = cv2.VideoCapture(0)  # 打开摄像头
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.resize(image, (256, 256))
                image = cv2.flip(image, 0)

                # # 进行背景去除
                # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # segmentation_result = selfie_seg.process(image_rgb)

                # mask = segmentation_result.segmentation_mask
                # threshold = 0.5  # 阈值，值越高人物边界越硬
                # mask = (mask > threshold).astype("uint8")  # 二值化
                
                # # 转换 BGR → BGRA（增加 Alpha 通道）
                # foreground = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                
                # # 设置 Alpha 通道（透明度）
                # foreground[:, :, 3] = (mask * 255).astype("uint8")

                # # 透明背景 PNG 编码
                # _, buffer = cv2.imencode(".png", foreground)
                _, buffer = cv2.imencode(".jpg", image)
                video_data = buffer.tobytes()

                # 获取 Landmark 数据
                image.flags.writeable = False
                # result = pose.process(image_rgb)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result=pose.process(image)

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
                    "emgDatas": emgDatas if emgDatas is not None else [1.0, 1.0, 1.0],
                    # "video": base64.b64encode(video_data).decode("utf-8")
                    "video": video_data.hex()  # 传输透明背景 PNG 视频帧
                }
                yield combine_data
                await asyncio.sleep(0.01 if not data_queue.empty() else 0.03)
    finally:
        if cap.isOpened():
            cap.release()
            print("Camera released in getLandmarkAndVideo")
        await asyncio.sleep(0.1)

def calculate_rms():
    feedback_csv_file_path = os.path.join(current_dir, f"landmarkAndVideo_ID{user_id}_feedback.csv")
    
    # 读取 CSV 文件，跳过第一行和第一列
    data = pd.read_csv(feedback_csv_file_path, skiprows=1)  # 跳过第一行（从第二行开始读取）
    data = data.iloc[:, 1:]  # 跳过第一列（保留从第二列开始的所有列）

    # 计算每列减去 32768 后的 RMS 值
    rms_values = {}
    for column in data.columns:
        # 将数据转换为浮点型，并减去 32768
        values = pd.to_numeric(data[column], errors='coerce') - 32768
        # 计算 RMS (均方根)
        rms = np.sqrt(np.mean(np.square(values)))
        rms_values[column] = rms
        print("rms: ",rms)
    # 计算第一列与第二列、第一列与第三列的 RMS 比例
    columns = list(rms_values.keys())

    ratio_data = []
    if len(columns) >= 3:
        # 计算比例
        ratio_1_2 = rms_values[columns[0]] / (rms_values[columns[1]] + rms_values[columns[0]]) if rms_values[columns[1]] != 0 else float('inf')
        ratio_1_3 = rms_values[columns[0]] / (rms_values[columns[2]] + rms_values[columns[0]]) if rms_values[columns[2]] != 0 else float('inf')

        # 将比例结果存入列表
        ratio_data = [ratio_1_2, ratio_1_3]

    return ratio_data

def calculate_normal_rms():
    feedback_csv_file_path = os.path.join(current_dir, f"videoData_ID{user_id}_NoneFeedback.csv")

    # 读取 CSV 文件，跳过第一行和第一列
    data = pd.read_csv(feedback_csv_file_path, skiprows=1)  # 跳过第一行（从第二行开始读取）
    data = data.iloc[:, 1:]  # 跳过第一列（保留从第二列开始的所有列）

    # 计算每列减去 32768 后的 RMS 值
    rms_values = {}
    for column in data.columns:
        # 将数据转换为浮点型，并减去 32768
        values = pd.to_numeric(data[column], errors='coerce') - 32768
        # 计算 RMS (均方根)
        rms = np.sqrt(np.mean(np.square(values)))
        rms_values[column] = rms
        print("rms: ",rms)
    # 计算第一列与第二列、第一列与第三列的 RMS 比例
    columns = list(rms_values.keys())

    ratio_data = []
    if len(columns) >= 3:
        # 计算比例
        ratio_1_2 = rms_values[columns[0]] / (rms_values[columns[1]] + rms_values[columns[0]]) if rms_values[columns[1]] != 0 else float('inf')
        ratio_1_3 = rms_values[columns[0]] / (rms_values[columns[2]] + rms_values[columns[0]]) if rms_values[columns[2]] != 0 else float('inf')

        # 将比例结果存入列表
        ratio_data = [ratio_1_2, ratio_1_3]

    return ratio_data

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
            cap = None
            print("Camera released")
