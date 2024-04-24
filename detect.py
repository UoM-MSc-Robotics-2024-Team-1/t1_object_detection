# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
#多从几个角度拍然后根据概率大小确定物体



"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
import torch


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# from geometry_msgs.msg import PointStamped



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)

# # 设置RealSense管道
# pipeline = rs.pipeline()
# config = rs.config()
# # 配置管道以启动深度和彩色流
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # 启动管道
# pipeline.start(config)


from utils.torch_utils import select_device, smart_inference_mode

# def get_dominant_color(image, k=4):
#     """
#     使用K-means聚类算法找到图像中的主要颜色。
#     :param image: 要分析的图像区域
#     :param k: 聚类中心的数量
#     :return: 主要颜色的BGR值
#     """
#     pixels = np.float32(image.reshape(-1, 3))
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     _, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     most_common = centroids[np.argmax(np.bincount(labels.flatten()))]
#     return most_common.astype(int)

# 这些颜色值是近似值，可能需要根据你的需要进行调整
color_dict = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'purple': (128, 0, 128),
    'yellow': (255, 255, 0),
    'green': (0, 128, 0),
    'pink': (255, 192, 203)
}

def get_dominant_color(image, k=4, color_space='RGB'):
    """
    image: 输入图像
    k: 聚类中心数
    color_space: 图像的色彩空间，'BGR'表示OpenCV默认格式，'RGB'表示已经是RGB格式。
    """
    # 如果图像是BGR格式，转换为RGB
    if color_space == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pixels = np.float32(image.reshape(-1, 3))

    # 检查是否有足够的像素进行聚类
    if pixels.shape[0] < k:
        print("Not enough pixels to perform k-means.")
        return np.array([0, 0, 0])  # 返回黑色作为默认颜色

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    most_common = centroids[np.argmax(np.bincount(labels.flatten()))]
    return most_common.astype(int)
# def get_dominant_color(image, k=4):
#     pixels = np.float32(image.reshape(-1, 3))

#     # 检查是否有足够的像素进行聚类
#     if pixels.shape[0] < k:
#         print("Not enough pixels to perform k-means.")
#         # 可以返回一个默认颜色或者根据实际情况处理
#         return np.array([0, 0, 0])  # 返回黑色作为默认颜色

#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     _, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     most_common = centroids[np.argmax(np.bincount(labels.flatten()))]
#     return most_common.astype(int)

def get_color_name(rgb_color, color_dict=color_dict):
    min_distance = float('inf')
    closest_color_name = None
    
    for color_name, color_value in color_dict.items():
        distance = np.sqrt(sum((np.array(rgb_color) - np.array(color_value))**2))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name
            
    return closest_color_name



import pyrealsense2 as rs
import numpy as np
import cv2
import torch

class RealSenseLoader:
    def __init__(self, img_size=(640, 480), stride=32):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.img_size = img_size
        self.stride = stride

        # Configure the pipeline to use the RealSense camera
        self.config.enable_stream(rs.stream.depth, img_size[0], img_size[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        
    def __iter__(self):
        return self

    def __next__(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise StopIteration

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to PyTorch tensor, normalize, and add batch dimension
        color_tensor = torch.from_numpy(color_image).float()
        color_tensor = color_tensor.permute(2, 0, 1) / 255.0  # BGR to RGB, HWC to CHW
        color_tensor = torch.unsqueeze(color_tensor, 0)  # Add batch dimension

        # return color_tensor, depth_image, color_image
        return color_image

    def stop(self):
        self.pipeline.stop()


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=6,  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        color_str = "No detections"
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # if len(det):
            #     # Rescale boxes from img_size to im0 size
            #     det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            #     # Print results
            #     for c in det[:, 5].unique():
            #         n = (det[:, 5] == c).sum()  # detections per class
            #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


            #     # Write results
            #     for *xyxy, conf, cls in reversed(det):
            #         c = int(cls)  # integer class
            #         label = names[c] if hide_conf else f"{names[c]}"
            #         confidence = float(conf)
            #         confidence_str = f"{confidence:.2f}"

            #         # color_str = "No detections"
            #         xyxy = [int(x) for x in xyxy]  # 转换为整数
            #         crop_img = im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    
            #         # 获取主要颜色
            #         dominant_color = get_dominant_color(crop_img)
            #         color_str = f"Color: {dominant_color}"  # 主要颜色字符串
            if len(det):
                # 遍历每个检测到的对象
                for *xyxy, conf, cls in reversed(det):
                    # 获取边界框坐标并转换为整数
                    xyxy_int = [int(x) for x in xyxy]
                    x_min, y_min, x_max, y_max = xyxy_int

                    # 截取边界框内的图像区域
                    crop_img = im0[y_min:y_max, x_min:x_max]

                    # 获取主要颜色
                    dominant_color = get_dominant_color(crop_img)
                    color_name = get_color_name(dominant_color,color_dict)

                    # 打印坐标和颜色
                    print(f"Bounding box: {xyxy_int},Class: {model.names[int(cls)]}, Dominant color: {dominant_color}")

                    bbox_str = f"({x_min}, {y_min}, {x_max}, {y_max})"
                    class_name = model.names[int(cls)]
                    color_str = ','.join(map(str, dominant_color))

                    # 调用 ROS2 发布函数
                    ros_node.publish_info(bbox_str, class_name, color_str)
            

                    # if save_csv:
                    #     write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms,{color_str},{xyxy}")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp24/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=6, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "/data/object_shape.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", default=True,action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt



class ObjectInfoPublisher(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.publisher_ = self.create_publisher(String, 'object_info', 10)

    def publish_info(self, bbox, class_name, color):
        # 创建一个信息字符串
        msg = String()
        msg.data = f"BBox: {bbox}, Class: {class_name}, Dominant Color: {color}"
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "{}"'.format(msg.data))




def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    rclpy.init()
    ros_node = ObjectInfoPublisher()
    opt = parse_opt()
    main(opt)
    # 确保在程序的最后关闭 ROS2 节点
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass  # 处理 Ctrl+C 中断
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()



