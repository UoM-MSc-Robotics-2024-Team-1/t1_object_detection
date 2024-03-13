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

Run the detection script with:

```python
python detect.py --source yourimage.jpg
```

Replace `yourimage.jpg` with the path to the image you want to process.

## Usage

To run the detection script, execute the following command:

```bash
python detect.py --source yourimage.jpg
Replace yourimage.jpg with the path to the image you wish to process.

Optional Arguments
Enhance your detection process by leveraging these optional arguments:

--weights
Specifies the path to the model weights.

Default: "path/to/default/weights.pt"
Example usage:
bash
Copy code
python detect.py --source yourimage.jpg --weights path/to/your/model_weights.pt
--source
Defines the source input for the detection process.

Default: 6 (for a specific camera ID)
Example for an image file:
bash
Copy code
python detect.py --source yourimage.jpg
--data
Points to the path of the dataset configuration file.

Default: "path/to/default/dataset.yaml"
Example usage:
bash
Copy code
python detect.py --source yourimage.jpg --data path/to/your/dataset.yaml
Adjust the above arguments according to your project's specifications and the resources you have.

