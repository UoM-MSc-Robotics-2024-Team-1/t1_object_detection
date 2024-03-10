YOLOv5n Object Detection for Leo Rover
Overview
This project integrates the lightweight and efficient YOLOv5n object detection model with the Leo Rover, enhancing its ability to recognize and understand its surroundings. By identifying the shape, position, and color of objects, we aim to significantly improve the Leo Rover's interaction with its environment, making it smarter and more adaptable.

Features
Shape Recognition: Allows the Leo Rover to detect and classify basic shapes, aiding in complex navigation and interaction tasks.
Position Detection: Equips the rover with the ability to determine the precise location of objects, essential for accurate mapping and effective obstacle avoidance.
Color Recognition: Enhances the rover's perception capabilities by identifying colors, facilitating tasks that require color-based object differentiation.
Getting Started
To get started with this project, follow the steps below:

Clone the repository:
bash
Copy code
git clone https://github.com/your-repository/leo-rover-yolov5n.git
Install dependencies:
bash
Copy code
cd leo-rover-yolov5n
pip install -r requirements.txt
Setup your Leo Rover:
Ensure your Leo Rover is setup and connected as per the official documentation.

Run the detection script:

bash
Copy code
python detect.py --weights yolov5n.pt --source 0  # for webcam
Replace --source 0 with the path to your rover's camera feed if necessary.

Contributing
We welcome contributions and suggestions! Whether it's adding new features, improving the model's efficiency, or providing better documentation, your input is valuable. Please feel free to fork the repository, make changes, and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

