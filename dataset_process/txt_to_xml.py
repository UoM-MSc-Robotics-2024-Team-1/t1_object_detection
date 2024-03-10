# from xml.dom.minidom import Document
# import os
# import cv2
# from tqdm import tqdm


# # def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
# def makexml(picPath, txtPath, xmlPath, datasetName):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
#     """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
#     在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
#     """
#     # dic = {'0': "cube",  # 创建字典用来对类型进行转换
#     #        '1': "bridge",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
#     #        '2': "cuboid",
#     #        '3': "diamond",
#     #        '4': "cuboid_fat",
#     #        '5': "cylinder",
#     #        '6': "semicycle",
#     #        '7': "trangle",
#     #        '8': "trangle_small",
#     #        '9': "cuboid_squre",
#     #        }
#     dic = {'0': "diamoind",  # 创建字典用来对类型进行转换

#            }

#     files = os.listdir(txtPath)
#     for i, name in enumerate(tqdm(files)):
#         if name == 'classes.txt':
#             continue
#         xmlBuilder = Document()
#         annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
#         xmlBuilder.appendChild(annotation)
#         txtFile = open(os.path.join(txtPath, name))
#         txtList = txtFile.readlines()
#         img = cv2.imread(os.path.join(picPath, name.replace('txt', 'jpg')))
#         Pheight, Pwidth, Pdepth = img.shape

#         folder = xmlBuilder.createElement("folder")  # folder标签
#         foldercontent = xmlBuilder.createTextNode(datasetName)
#         folder.appendChild(foldercontent)
#         annotation.appendChild(folder)  # folder标签结束

#         filename = xmlBuilder.createElement("filename")  # filename标签
#         filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
#         filename.appendChild(filenamecontent)
#         annotation.appendChild(filename)  # filename标签结束

#         size = xmlBuilder.createElement("size")  # size标签
#         width = xmlBuilder.createElement("width")  # size子标签width
#         widthcontent = xmlBuilder.createTextNode(str(Pwidth))
#         width.appendChild(widthcontent)
#         size.appendChild(width)  # size子标签width结束

#         height = xmlBuilder.createElement("height")  # size子标签height
#         heightcontent = xmlBuilder.createTextNode(str(Pheight))
#         height.appendChild(heightcontent)
#         size.appendChild(height)  # size子标签height结束

#         depth = xmlBuilder.createElement("depth")  # size子标签depth
#         depthcontent = xmlBuilder.createTextNode(str(Pdepth))
#         depth.appendChild(depthcontent)
#         size.appendChild(depth)  # size子标签depth结束

#         annotation.appendChild(size)  # size标签结束

#         for j in txtList:
#             oneline = j.strip().split(" ")
#             object = xmlBuilder.createElement("object")  # object 标签
#             picname = xmlBuilder.createElement("name")  # name标签
#             namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
#             picname.appendChild(namecontent)
#             object.appendChild(picname)  # name标签结束

#             pose = xmlBuilder.createElement("pose")  # pose标签
#             posecontent = xmlBuilder.createTextNode("Unspecified")
#             pose.appendChild(posecontent)
#             object.appendChild(pose)  # pose标签结束

#             truncated = xmlBuilder.createElement("truncated")  # truncated标签
#             truncatedContent = xmlBuilder.createTextNode("0")
#             truncated.appendChild(truncatedContent)
#             object.appendChild(truncated)  # truncated标签结束

#             difficult = xmlBuilder.createElement("difficult")  # difficult标签
#             difficultcontent = xmlBuilder.createTextNode("0")
#             difficult.appendChild(difficultcontent)
#             object.appendChild(difficult)  # difficult标签结束

#             bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
#             xmin = xmlBuilder.createElement("xmin")  # xmin标签
#             mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
#             xminContent = xmlBuilder.createTextNode(str(mathData))
#             xmin.appendChild(xminContent)
#             bndbox.appendChild(xmin)  # xmin标签结束

#             ymin = xmlBuilder.createElement("ymin")  # ymin标签
#             mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
#             yminContent = xmlBuilder.createTextNode(str(mathData))
#             ymin.appendChild(yminContent)
#             bndbox.appendChild(ymin)  # ymin标签结束

#             xmax = xmlBuilder.createElement("xmax")  # xmax标签
#             mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
#             xmaxContent = xmlBuilder.createTextNode(str(mathData))
#             xmax.appendChild(xmaxContent)
#             bndbox.appendChild(xmax)  # xmax标签结束

#             ymax = xmlBuilder.createElement("ymax")  # ymax标签
#             mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
#             ymaxContent = xmlBuilder.createTextNode(str(mathData))
#             ymax.appendChild(ymaxContent)
#             bndbox.appendChild(ymax)  # ymax标签结束

#             object.appendChild(bndbox)  # bndbox标签结束

#             annotation.appendChild(object)  # object标签结束

#         f = open(os.path.join(xmlPath, name.replace('txt', 'xml')), 'w')
#         xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
#         f.close()


# if __name__ == "__main__":
#     picPath = r"C:\Users\25166\Desktop\dataset\diamond_img"         # 原图文件夹路径
#     txtPath = r"C:\Users\25166\Desktop\dataset\diamond_label"    # 原txt标签文件夹路径
#     xmlPath = r"C:\Users\25166\Desktop\dataset\diamond_xml"    # 保存xml文件夹路径
#     datasetName = r'MyDataset'      # 数据集名称
#     os.makedirs(xmlPath) if not os.path.exists(xmlPath) else None
#     makexml(picPath, txtPath, xmlPath, datasetName)

#     # tasks = ['train', 'test', 'val']          # 当前任务
#     # for task in tasks:
#     #     picPath = r"...\images\{}".format(task)         # 原图文件夹路径
#     #     txtPath = r"...\labels\{}".format(task)         # 原txt标签文件夹路径
#     #     xmlPath = r"...\labels\{}_xml".format(task)     # 保存xml文件夹路径
#     #     datasetName = r'MyDataset'
#     #     os.makedirs(xmlPath) if not os.path.exists(xmlPath) else None
#     #     makexml(picPath, txtPath, xmlPath, datasetName)


from xml.dom.minidom import Document
import os
import cv2
from tqdm import tqdm

def makexml(picPath, txtPath, xmlPath, datasetName):
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {
        '0': "diamond",

    }

    files = os.listdir(txtPath)
    for i, name in enumerate(tqdm(files)):
        if name == 'classes.txt' or not name.endswith('.txt'):
            continue

        img_file_path = os.path.join(picPath, name.replace('txt', 'jpg'))
        if not img_file_path.lower().endswith('.jpg'):
            print(f"跳过非图片文件: {img_file_path}")
            continue

        img = cv2.imread(img_file_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_file_path}，请检查文件路径和文件名。")
            continue

        Pheight, Pwidth, Pdepth = img.shape
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")
        xmlBuilder.appendChild(annotation)

        folder = xmlBuilder.createElement("folder")
        foldercontent = xmlBuilder.createTextNode(datasetName)
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)

        filename = xmlBuilder.createElement("filename")
        filenamecontent = xmlBuilder.createTextNode(name[:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)

        size = xmlBuilder.createElement("size")
        width = xmlBuilder.createElement("width")
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)

        height = xmlBuilder.createElement("height")
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)

        depth = xmlBuilder.createElement("depth")
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)

        annotation.appendChild(size)

        txtFile = open(os.path.join(txtPath, name))
        txtList = txtFile.readlines()
        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)

            # 添加其他必要的标签和属性
            # ...

            annotation.appendChild(object)

        f = open(os.path.join(xmlPath, name.replace('txt', 'xml')), 'w')
        xmlBuilder.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
        f.close()

if __name__ == "__main__":
    picPath = r"C:\Users\25166\Desktop\dataset\diamond_img"         # 原图文件夹路径
    txtPath = r"C:\Users\25166\Desktop\dataset\diamond_label"    # 原txt标签文件夹路径
    xmlPath = r"C:\Users\25166\Desktop\dataset\diamond_xml"    # 保存xml文件夹路径
    datasetName = 'MyDataset'                                  # 数据集名称
    os.makedirs(xmlPath, exist_ok=True)
    makexml(picPath, txtPath, xmlPath, datasetName)
