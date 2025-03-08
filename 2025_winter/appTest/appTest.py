import asyncio
import websockets
import json
from PIL import Image
import io

async def test_client():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        try:
            while True:
                message = await websocket.recv()  # 接收数据
                data = json.loads(message)

                # 打印蓝牙数据
                print("emgdatas:", data.get("emgDatas", None))

                # 打印 landmarks 数据
                print("Landmarks Count:", len(data.get("landmarks", [])))

                # 处理视频数据
                video_data_hex = data.get("video", None)
                if video_data_hex:
                    print("video received")
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")

asyncio.run(test_client())
