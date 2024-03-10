import os

def batch_rename_files(folder_path, new_prefix):
    # 获取文件夹中所有文件名
    files = os.listdir(folder_path)
    
    # 遍历文件夹中的每个文件
    for index, file_name in enumerate(files):
        for i in range(500):

            if (i+1) == int(os.path.splitext(file_name)[0]):
            
                new_file_name = f"{new_prefix}_{i+1}{os.path.splitext(file_name)[1]}"
                
                # 旧文件路径
                old_file_path = os.path.join(folder_path, file_name)
                # 新文件路径
                new_file_path = os.path.join(folder_path, new_file_name)
                
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")

# 文件夹路径
folder_path = r'C:\Users\25166\Desktop\dataset\labels-20240310T101911Z-001\diamond'
# 新文件名的前缀
new_prefix = 'desk_'

# 执行批量重命名
batch_rename_files(folder_path, new_prefix)
