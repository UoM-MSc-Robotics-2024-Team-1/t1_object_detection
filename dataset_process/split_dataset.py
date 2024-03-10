from sklearn.model_selection import train_test_split
import os
import shutil

# 假设你的图片和标签文件位于以下目录
images_dir = r'C:\Users\25166\Desktop\dataset\cube_img'  # 图片文件夹路径
labels_dir = r'C:\Users\25166\Desktop\dataset\cube_labels'  # 标签文件夹路径

# 输出目录
train_images_dir = r'C:\Users\25166\Desktop\dataset\cube_dataset\images\train'
train_labels_dir = r'C:\Users\25166\Desktop\dataset\cube_dataset\labels\train'
val_images_dir = r'C:\Users\25166\Desktop\dataset\cube_dataset\images\val'
val_labels_dir = r'C:\Users\25166\Desktop\dataset\cube_dataset\labels\val'

# 获取图片文件列表
images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
# 生成对应的标签文件列表
labels = [f.replace('.jpg', '.txt') for f in images]  # 假设图片为.jpg, 标签为.txt

# 随机划分训练集和验证集，这里以80%训练集，20%验证集为例
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 复制文件到相应的目录
# def copy_files(files, source_dir, target_dir):
#     for f in files:
#         shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))
def copy_files(file_list, source_dir, target_dir):
    for f in file_list:
        # 跳过desktop.ini文件
        if f == 'desktop.ini':
            continue
        src = os.path.join(source_dir, f)
        dst = os.path.join(target_dir, f)
        try:
            shutil.copy(src, dst)
        except PermissionError as e:
            print(f"权限错误：{e}")
        except FileNotFoundError as e:
            print(f"文件未找到：{e}")


# 复制图片和标签到训练集和验证集目录
copy_files(train_images, images_dir, train_images_dir)
copy_files(train_labels, labels_dir, train_labels_dir)
copy_files(val_images, images_dir, val_images_dir)
copy_files(val_labels, labels_dir, val_labels_dir)

print("数据集划分完成。")
