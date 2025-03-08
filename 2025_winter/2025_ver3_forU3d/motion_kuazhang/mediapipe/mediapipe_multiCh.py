#可用，下一个文件是为了将matplot的图表显示在cv2窗口中
import cv2
import mediapipe as mp
import math
import collections
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

if len(sys.argv)>2:
    user_id=sys.argv[1]
    test_num=sys.argv[2]
else:
    user_id="unknownID"
    test_num="testNumError"

def get_max_values():
    # 定义最大值Excel文件路径
    max_values_file = fr'C:\Users\munek\Downloads\2025_exaggerate\data\maxValue_ID{user_id}_TNmax.xlsx'
    # 读取Excel文件
    max_values_df = pd.read_excel(max_values_file)

    # 提取通道的最大值
    max_emg_value1 = abs(max_values_df['Channel1'].iloc[0]-32768)
    max_emg_value2 = abs(max_values_df['Channel2'].iloc[0]-32768)
    max_emg_value3 = abs(max_values_df['Channel3'].iloc[0]-32768)

    return max_emg_value1, max_emg_value2, max_emg_value3

def getCordinate(pipe):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    # 指定窗口的名称
    window_name = 'camera1'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使用 WINDOW_NORMAL 使窗口可调整大小

    # 将窗口设置为500*500
    cv2.resizeWindow(window_name, 1200, 800)

    def get_emg_data():
        try:
            line = pipe.readline().strip()
            channel1, channel2, channel3 = map(float, line.split(','))
            # print(f"Received EMG data: {channel1}, {channel2}, {channel3}")
            return channel1, channel2, channel3
        except Exception as e:
            print(f"Failed to read EMG data: {e}")
            return 0.0, 0.0, 0.0

    

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                # 处理第一个椭圆（9和11之间）
                Ax1 = results.pose_landmarks.landmark[9].x * image.shape[1]
                Ay1 = results.pose_landmarks.landmark[9].y * image.shape[0]
                Bx1 = results.pose_landmarks.landmark[11].x * image.shape[1]
                By1 = results.pose_landmarks.landmark[11].y * image.shape[0]

                mid_x1 = (Ax1 + Bx1) / 2
                mid_y1 = (Ay1 + By1) / 2
                
                length1 = (math.sqrt((Bx1 - Ax1) ** 2 + (By1 - Ay1) ** 2)) // 2
                angle1 = math.atan2(By1 - Ay1, Bx1 - Ax1)
                
                long_axis1 = int(length1)
                short_axis1 = int(length1 / 2)

                # 处理第二个椭圆（11上）
                Ax2 = results.pose_landmarks.landmark[11].x * image.shape[1]
                Ay2 = results.pose_landmarks.landmark[11].y * image.shape[0]

                # 处理第三个椭圆（10和11之间）
                Ax3 = results.pose_landmarks.landmark[13].x * image.shape[1]
                Ay3 = results.pose_landmarks.landmark[13].y * image.shape[0]
                Bx3 = results.pose_landmarks.landmark[15].x * image.shape[1]
                By3 = results.pose_landmarks.landmark[15].y * image.shape[0]

                mid_x3 = (Ax3 + Bx3) / 2
                mid_y3 = (Ay3 + By3) / 2
                
                length3 = (math.sqrt((Bx3 - Ax3) ** 2 + (By3 - Ay3) ** 2)) // 2
                angle3 = math.atan2(By3 - Ay3, Bx3 - Ax3)
                
                long_axis3 = int(length3)
                short_axis3 = int(length3 / 2)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            channel1, channel2, channel3 = get_emg_data()

            if results.pose_landmarks:
                cv2.ellipse(image, (int(mid_x1), int(mid_y1)), (long_axis1 // 2, short_axis1 // 2), math.degrees(angle1), 0, 360, (255,0,0), -1)
                cv2.circle(image, (int(Ax2), int(Ay2)), long_axis1 // 2, (0,255,0), -1)
                cv2.ellipse(image, (int(mid_x3), int(mid_y3)), (long_axis3 // 2, short_axis3 // 2), math.degrees(angle3), 0, 360, (0,0,255), -1)

            # 获取屏幕尺寸并调整图像大小
            screen_width = cv2.getWindowImageRect(window_name)[2]
            screen_height = cv2.getWindowImageRect(window_name)[3]
            image = cv2.resize(image, (screen_width, screen_height))

            cv2.imshow(window_name, cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord('s'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    getCordinate(sys.stdin)
