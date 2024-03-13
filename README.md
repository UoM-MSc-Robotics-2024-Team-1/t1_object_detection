# Leo Rover Object Detection with YOLOv5n

This project enhances the Leo Rover's environmental interaction by using the lightweight YOLOv5n model for object detection, focusing on recognizing the shape, position, and color of objects.

## Features

- **Shape Recognition**: Classify basic shapes to aid in navigation and interaction.
- **Position Detection**: Determine the precise location of objects for mapping and obstacle avoidance.
- **Color Recognition**: Identify colors of objects for specific tasks like sorting or analysis.

## Getting Started

Clone the repository to get started with enhancing your Leo Rover:

```bash
git clone https://github.com/UoM-MSc-Robotics-2024-Team-1/t1_object_detection.git
```

### Prerequisites

Ensure you have Python 3.8+ installed on your system:

```bash
python --version
```

### Installation

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Usage
To run the detection script, use the following command:

```python
python detect.py
```
The script supports several optional arguments:

`--weights`: Set the model path or triton URL. Default is pre-configured to a specific path. Change this to where your model's weights are stored.

```python
python detect.py --weights path/to/your/model_weights.pt
```


`--source`: Define the source of the input. It can be a file path, directory, URL, glob pattern, or a camera ID ('0' for webcam). The default is 6, adjust as needed.

```python
python detect.py --source yourimage.jpg
```


`--data`: Specify the dataset configuration file path (dataset.yaml). The default path is set; update it according to your dataset or configuration file's location.

```python
python detect.py --data path/to/your/dataset.yaml
```
Ensure to modify these arguments according to your setup and the requirements of your detection task.

