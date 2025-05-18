# Image of Interest - Raspberry Pi Person Detection System using PyTorch YOLOv8n Model

# Model.py Explanation

1. ObjectDetector Class
- This class is the core of your detection system, handling object detection with a Sony camera on Raspberry Pi.
- Uses YOLOv8n model for person detection.
- Implements image quality assessment to capture only high-quality frames.
- Tracks performance metrics including VRAM usage, detection time, and latency.

2. Initialization 
- Sets up the detection parameters
- Creates directories for saving images and models
- Initializes the YOLO model, specifically loading "yolov8n.pt"
- Configures the model to only detect people
- Sets up tracking arrays for performance metrics

3. Model Loading Functions
- _load_model: Loads the YOLOv8n model, downloading it if not present
- Handles configuring the model to specifically detect the "person" class

4. Image Quality Assessment Functions
- detect_blur: Measures image sharpness using Laplacian variance
- detect_stability: Compares consecutive frames to detect motion/stability
- assess_lighting: Evaluates image brightness and exposure quality
- capture_best_frame: Decides whether to capture an image based on quality metrics
  and saves it with performance metrics in the filename

5. Performance Tracking Functions
- get_vram_usage: Measures memory usage (VRAM or system memory on Raspberry Pi)
- get_system_resources: Collects CPU, RAM, and disk usage metrics
- print_performance_metrics: Generates a summary report of all tracked metrics
- Real-time tracking of object detection time, image capture time, FPS, and latency

6. Detection Visualization (plot_box)
- Draws bounding boxes and confidence scores on detected people
- Displays real-time performance metrics on the video feed

7. Main Detection Loop (__call__)
- Initializes the camera using Picamera2 for optimal performance with the IMX219 Sony camera
- Continuously processes frames for person detection
- Manages frame quality assessment and triggers image capture when quality thresholds are met
- Tracks and displays performance metrics in real-time
- Saves high-quality images of detected persons with performance data in the filename
- Provides a comprehensive performance summary when exiting

# Verification.py Explanation

1. Main Function (main)
- Orchestrates the verification process for all system components
- Collects results and provides a summary of passed/failed checks
- Helper Function (print_status)
- Formats verification messages with checkmarks/X marks for success/failure

2. Component Verification Functions
- verify_dependencies: Checks all required Python libraries are installed
- verify_paths: Ensures required directories exist or creates them
- verify_model: Validates the YOLO model is present and can detect people
- verify_camera: Tests camera access using Picamera2 
- verify_object_detector: Tests core functionality of the ObjectDetector class
- camera_diagnostic: Performs detailed camera diagnostics when issues are detected

# Main.py Explanation
- Main Function (main)
- Entry point for the application
- Checks for required dependencies (including psutil for memory tracking)
- Creates necessary directories
- Initializes the ObjectDetector with appropriate parameters
- Runs the detection loop in a try/except block for error handling

# Installation Scripts
- global_install.sh: System-level dependencies and configurations for Raspberry Pi
- venv_install.sh: Python virtual environment setup with all required packages
- Includes model conversion utilities for optimized performance on Raspberry Pi

# System Requirements

1. Hardware
- Raspberry Pi (tested on Raspberry Pi OS)
- Sony IMX219 Camera with 1920x1080 resolution
- USB connection between camera and Raspberry Pi

2. Software Dependencies
- Python 3.8+
- OpenCV 
- PyTorch
- Ultralytics (for YOLOv8)
- NumPy
- Picamera2
- psutil (for memory tracking)

# Getting Started

1. Setup
- Clone this repository to your Raspberry Pi
- Run `sudo bash global_install.sh` to install system dependencies
- Run `bash venv_install.sh` to set up the Python environment
- Ensure your Sony camera is connected and recognized by the system
- Run `python verification.py` to check if all components are properly set up

2. Running the System
- Once verification passes, run `python main.py` to start person detection
- Alternatively, use the provided run script: `./run.sh main.py`
- The system will:
  - Initialize the camera at 1920x1080 resolution
  - Load the YOLOv8n model
  - Start detecting people in the camera feed
  - Save high-quality images when people are detected
  - Display the camera feed with detection results and performance metrics

3. Output
- Detected persons will be saved to the `captured_images` directory
- Each image is named with performance metrics:
  `WMSV4AI_[Detection Time]_[FPS]_[Latency]_[VRAM]_[Capture Time].jpg`
- Performance metrics are displayed in real-time on the video feed
- A complete performance summary is provided when the application exits

4. Performance Metrics Tracked
- VRAM/Memory Consumption: Monitors memory usage during detection
- CPU Usage and Temperature: Tracks system resource utilization
- Object Detection Time: Measures time taken for YOLO model inference
- Image Capture Time: Tracks time required to save captured images
- Average FPS: Calculates frames processed per second
- Average Latency: Measures total processing time per frame
- Disk Usage: Monitors storage availability

5. Exiting
- Press 'q' to exit the application
- The system will gracefully shut down and display:
  - Total number of images captured
  - Complete performance metrics summary

6. Advanced Usage
- Use `python convert_model.py` to optimize the model for Raspberry Pi using NCNN
- Adjust detection parameters in `main.py` to customize sensitivity and performance