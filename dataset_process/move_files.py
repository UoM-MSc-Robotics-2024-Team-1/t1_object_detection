import os
import shutil

def move_files(source_folder, destination_folder):
    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 源文件路径
            source_file_path = os.path.join(root, file)
            # 目标文件路径
            destination_file_path = os.path.join(destination_folder, file)
            # 移动文件
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved {source_file_path} to {destination_file_path}")

# 源文件夹路径
source_folder = r'C:\Users\25166\Desktop\object\Image-Capture-With-RealSense-master\pictures\pictures'
# 目标文件夹路径
destination_folder = r'C:\Users\25166\Desktop\object\Image-Capture-With-RealSense-master\pictures\newone'

# 执行文件移动
move_files(source_folder, destination_folder)
