This project demonstrates object detection using the YOLOv8 model with a pre-recorded video file and also by real-time vide capture. It uses the `ultralytics` library for YOLO, `opencv-python` for handling video capture and display, and `cvzone` for additional OpenCV functionalities.

## Requirements

To run this project, you need to install the following Python packages:

- `ultralytics==8.2.0`
- `opencv-python==4.10.0`
- `cvzone==1.5.1`

You can install these dependencies using pip. The recommended way is to use the `requirements.txt` file provided.

## Installation

1. Clone the Repository

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
2.pip install -r requirements.txt


Usage
Download YOLO Weights

Make sure you have the YOLO weights file yolov8l.pt. If you don't have it, the script will automatically download it to the ../Yolo-Weights/ directory.

Prepare Your Video File

Place your video file in the ../Videos/ directory or modify the video_path variable in the script to point to your video file.

Run the Object Detection Script
3.python detect.py



License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
If you have any questions or issues, please open an issue on the GitHub repository.





### Notes:
- Replace `https://github.com/yourusername/your-repository.git` with the URL of your GitHub repository.
- Adjust paths (`video_path` and YOLO weights location) as necessary based on your directory structure.
- Ensure you provide a license if applicable and adjust the contact information as needed.

This `README.md` file provides a clear guide for setting up and running your YOLO object detection project.
