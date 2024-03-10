# Leo Rover Object Detection with YOLOv5n

This project enhances the Leo Rover's environmental interaction by using the lightweight YOLOv5n model for object detection, focusing on recognizing the shape, position, and color of objects.

## Features

- **Shape Recognition**: Classify basic shapes to aid in navigation and interaction.
- **Position Detection**: Determine the precise location of objects for mapping and obstacle avoidance.
- **Color Recognition**: Identify colors of objects for specific tasks like sorting or analysis.

## Getting Started

Clone the repository to get started with enhancing your Leo Rover:

```bash
git clone https://github.com/yourusername/leo-rover-yolov5n-detection.git
```

### Prerequisites

Ensure you have Python 3.6+ installed on your system:

```bash
python --version
```

### Installation

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the detection script with:

```python
python detect.py --source yourimage.jpg
```

Replace `yourimage.jpg` with the path to the image you want to process.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for how you can contribute to this project.
