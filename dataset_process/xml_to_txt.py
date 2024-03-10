import xml.etree.ElementTree as ET
import os
import glob

# 定义您的类别
classes = ['red_cube',
'green_cube',
'blue_cube',
'yellow_cube']

def convert(size, box):
    dw = 1.0 / (size[0] + 1)
    dh = 1.0 / (size[1] + 1)
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    # 根据您的实际文件路径修改
    in_file = os.path.join('C:\\Users\\25166\\Desktop\\dataset\\cube_xml', image_id + '.xml')
    out_file = os.path.join('C:\\Users\\25166\\Desktop\\dataset\\cube_xml', image_id + '.txt')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(out_file, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':
    # 注意：请根据您的文件结构调整glob路径
    for image_path in glob.glob('C:\\Users\\25166\\Desktop\\dataset\\cube_img\\*.jpg'):
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        convert_annotation(image_id)
