import os
import subprocess
import tkinter as tk

import time
from tkinter import messagebox,scrolledtext

import pandas as pd


def open_terminal_with_command(user_id, test_num,command_path):
    try:
        plux_path = r"C:\Users\munek\Downloads\2025_exaggerate\2025_ver2\PLUX-API-Python3\Win64_39"

        if os.name == "nt":

            with open("temp_command.bat", "w") as f:
                f.write(f"set PYTHONPATH={plux_path}\n")
                f.write(f"set USER_ID={user_id}\n")
                f.write(f"set TEST_NUM={test_num}\n")
                f.write(f'python "{command_path}"\n')
                f.write("pause\n")

            subprocess.Popen("start cmd /k temp_command.bat", shell=True)

        print("已启动终端并执行命令。")

    except Exception as e:
        print(f"启动终端时出错:{e}")


def get_max_data(user_id,test_num):
    try:
        file_path = fr"C:\Users\munek\Downloads\2025_exaggerate\data\kuazhang_ID{user_id}_TNtest.xlsx"

        if not os.path.exists(file_path):
            messagebox.showerror("文件错误", f"未找到Excel文件: {file_path}")
            return

        # 读取Excel文件
        df = pd.read_excel(file_path)

        channels = ["Channel1", "Channel2", "Channel3"]

        results = {}

        for channel in channels:
            if channel in df.columns:
                max_values = df[channel].nlargest(100).reset_index(drop=True)
                min_values = df[channel].nsmallest(100).reset_index(drop=True)

                values = abs(max_values - 32768) + abs(min_values - 32768)

                trimmed_values = values.iloc[5:-10]

                mean_value = trimmed_values.mean()
                results[channel] = mean_value

            else:
                results[channel] = "ch. not found"

        max_values_file = fr"C:\Users\munek\Downloads\2025_exaggerate\data\maxValue_ID{user_id}_TNmax.xlsx"
        max_values_df=pd.DataFrame([results])
        max_values_df.to_excel(max_values_file,index=False)
        
        show_max_value_box.delete('1.0',tk.END)
        show_max_value_box.insert(tk.END,"The Ave. of Max Value for each ch.: \n")
        for channel,mean_value in results.items():
            show_max_value_box.insert(tk.END,f"{channel}: {mean_value}\n")


    except Exception as e:
        messagebox.showerror("error", f"deal data with error: {e}")


def create_window():
    root = tk.Tk()
    root.title("2025_exaggerat_Ver2.0")
    root.geometry("500x500")
    
    top_frame=tk.Frame(root)
    top_frame.pack(pady=10)
    
    bottom_frame=tk.Frame(root)
    bottom_frame.pack(pady=10)

    #input ID
    input_label = tk.Label(top_frame, text="ID: ")
    input_label.grid(row=0, column=0, padx=5, pady=5, sticky="W")
    input_entry = tk.Entry(top_frame, width=10)
    input_entry.grid(row=0, column=1, padx=5, pady=5, sticky="W")
    
    #input test number
    input_num_label=tk.Label(top_frame,text="input test number: ")
    input_num_label.grid(row=1,column=0,padx=5,pady=5, sticky="W")
    input_num_entry = tk.Entry(top_frame,width=10)
    input_num_entry.grid(row=1, column=1, padx=5, pady=5, sticky="W")

    #button motion_kuazhang 
    button1 = tk.Button(
        bottom_frame,
        text="motion_kuazhang",
        command=lambda: open_terminal_with_command(
            input_entry.get() if input_entry.get() else "unkonwnID",
            input_num_entry.get() if input_num_entry.get() else "testNumError",
            r"C:\Users\munek\Downloads\2025_exaggerate\2025_ver2\motion_kuazhang\device\motion_kuazhang.py",
        ),
    )

    #get max value from each ch.
    get_max_button = tk.Button(
        bottom_frame,
        text="get max value for each ch.",
        command=lambda: get_max_data(
            input_entry.get() if input_entry.get() else "unkonwnID",
                        input_num_entry.get() if input_num_entry.get() else "testNumError"
        ),
    )
    
    #show max value
    global show_max_value_box
    show_max_value_box=scrolledtext.ScrolledText(root,width=50,height=20)

    button1.pack(pady=5)
    get_max_button.pack(pady=5)
    show_max_value_box.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_window()
