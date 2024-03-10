import os

# 设定你的目标文件夹路径
folder_path = r'C:\Users\25166\Desktop\dataset\labels-20240310T161557Z-001\labels'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # 确认文件是txt文件
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 修改第一个字符
        try:
            if content:  # 确保文件不是空的
                content = '0' + content[1:]
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
        except:
            pass
