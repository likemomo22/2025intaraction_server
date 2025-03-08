import csv
from itertools import cycle
import os
import platform
from queue import Queue
import sys
import threading
import time
import pandas as pd
from multiprocessing import Pipe, Process

stop_flag = threading.Event()

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)

alpha = 0.2
NUM_OF_CH = 3
MAX_VALUE = 7000
MIN_VALUE = 0
MAX_RETRIES=3
RETRY_DELAY = 2

# 定义不同操作系统下的PLUX API路径
osDic = {
    "Darwin": f"MacOS/Intel{''.join(platform.python_version().split('.')[:2])}",
    "Linux": "Linux64",
    "Windows": f"Win{platform.architecture()[0][:2]}_{''.join(platform.python_version().split('.')[:2])}",
}

user_id = os.getenv("USER_ID", "unknownID")
test_num = os.getenv("TEST_NUM", "testNumError")

if platform.mac_ver()[0] != "":
    import subprocess
    from os import linesep

    p = subprocess.Popen("sw_vers", stdout=subprocess.PIPE)
    result = p.communicate()[0].decode("utf-8").split(str("\t"))[2].split(linesep)[0]
    if result.startswith("12."):
        print("macOS version is Monterrey!")
        osDic["Darwin"] = "MacOS/Intel310"
        if (
            int(platform.python_version().split(".")[0]) <= 3
            and int(platform.python_version().split(".")[1]) < 10
        ):
            print(
                f"Python version required is ≥ 3.10. Installed is {platform.python_version()}"
            )
            exit()

sys.path.append(
    r"C:\Users\munek\Downloads\2025_winter\2025_winter\2025_ver3_forU3d\PLUX-API-Python3\Win64_39"
)

import plux

data_queue = Queue(maxsize=1000)

class NewDevice(plux.MemoryDev):
    def __init__(self, address):
        super().__init__()
        global fileName
        self.address = address
        self.duration = 0
        self.frequency = 0
        self.filename = fileName
        self.header_written = False  # 表头写入标志
        self.ema_values = [32768] * NUM_OF_CH  # 动态初始化

    def onRawFrame(self, nSeq, data):
        if not stop_flag.is_set():
            if nSeq % 50 == 0:
                abs_values = [self.get_abs_value(d) for d in data]

                # 计算 EMA 值
                self.ema_values = [
                    self.get_ema_value(ema, abs_val, alpha)
                    for ema, abs_val in zip(self.ema_values, abs_values)
                ]
                
                row = [nSeq] + list(data)
                self.save_to_excel(row)
                data_queue.put(self.ema_values)
        else:
            return True

    def get_abs_value(self, data):
        return abs(data - 32768)

    def get_ema_value(self, previous_ema, new_value, alpha=0.2):
        if previous_ema is None:
            return new_value
        ema_value = alpha * new_value + (1 - alpha) * previous_ema
        return ema_value

    def save_to_excel(self, row):
        try:
            columns = ["Sequence Number", "Channel1", "Channel2", "Channel3"]
            df = pd.DataFrame([row], columns=columns)  # 将数据转换为 DataFrame 格式
            if not self.header_written:  # 如果还没有写入表头
                df.to_csv(self.filename, mode="w", index=False, header=True)
                self.header_written = True
            else:  # 追加写入，不覆盖文件
                df.to_csv(self.filename, mode="a", index=False, header=False)
        except Exception as e:
            print(f"Error saving to file: {e}")


# ---------------------------------------#
def get_abs_value( data):
    return abs(data - 32768)


def get_ema_value( previous_ema, new_value, alpha=0.2):
    if previous_ema is None:
        return new_value
    ema_value = alpha * new_value + (1 - alpha) * previous_ema
    return ema_value


def read_test_data():
    try:
        with open(
            # "/Users/higashi/2025_winter/TestData/testData.csv",
            r"C:\Users\munek\Downloads\2025_winter\2025_winter\TestData\testData.csv",
            mode="r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                yield row
    except Exception as e:
        print(f"error: {e}")


# -------------------------------------#
def exampleAcquisition(
    pipe, address="BTH00:07:80:89:7F:C5", duration=500, frequency=1000, code=0x01,filename=None
):
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(1)
            print("-----------------------------------")
            print(f"🚀 Try to create NewDevice [{attempt+1}/{MAX_RETRIES}]")
            global fileName
            fileName=filename
            device = NewDevice(address)
            print("✅ Create NewDevice Success")
            break  # 成功连接，跳出重试循环
        except Exception as e:
            print(f"❌ NewDevice Creating False: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"🔄 等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print("❌ 达到最大重试次数，停止连接")
                return  # 彻底失败，避免后续代码执行
    
    device.duration = int(duration)
    device.frequency = int(frequency)
    device.pipe = pipe
    if isinstance(code, str):
        code = int(code, 16)
    # 配置通道
    analog_source_ch1 = plux.Source()
    analog_source_ch1.port = 1
    analog_source_ch1.freqDivisor = 1
    analog_source_ch1.nBits = 16
    analog_source_ch1.chMask = 0x01

    analog_source_ch2 = plux.Source()
    analog_source_ch2.port = 2
    analog_source_ch2.freqDivisor = 1
    analog_source_ch2.nBits = 16
    analog_source_ch2.chMask = 0x01

    analog_source_ch3 = plux.Source()
    analog_source_ch3.port = 3
    analog_source_ch3.freqDivisor = 1
    analog_source_ch3.nBits = 16
    analog_source_ch3.chMask = 0x01

    device.start(
        device.frequency, [analog_source_ch1, analog_source_ch2, analog_source_ch3]
    )

    # data = cycle(read_test_data())
    ema_values = [0] * NUM_OF_CH
    try:
        while not stop_flag.is_set():  # 循环检查停止标志
            device.loop()  # 执行蓝牙数据采集
            # row = next(data)
            # selected_data = [int(row[1]), int(row[2]), int(row[3])]
            # selected_data = device.ema_values  # 使用设备采集的数据
            # abs_values = [get_abs_value(d) for d in selected_data]
            # ema_values = [
            #     get_ema_value(ema, abs_val, alpha)
            #     for ema, abs_val in zip(ema_values, abs_values)
            # ]
            # data_queue.put(ema_values)
    except Exception as e:
        print(f"Error in Bluetooth acquisition: {e}")
    finally:
        print("-----------------------------------")
        print("🔄 正在停止设备...")
        if device:
            if stop_flag.is_set():
                device.stop()
                device.close()
                print("✅ 蓝牙设备已成功关闭")
