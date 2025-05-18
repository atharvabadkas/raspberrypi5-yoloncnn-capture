# Real-Time Object Image Capture: Implementation Strategy

## 1. Introduction

This document outlines the implementation strategy for an enhanced real-time image capture system. The primary goal is to reliably capture stable, non-blurry, and clear images of specific kitchen objects (e.g., trays, plates, hands) using a Raspberry Pi 5 equipped with an IMX219 camera module.

This project evolves from a previous system that utilized a PyTorch-based YOLOv8n model. The key improvements will focus on:
1.  **Upgraded Model Backend:** Transitioning to a YOLOv11n model in NCNN format for significantly improved CPU inference performance on the Raspberry Pi.
2.  **Advanced Image Quality Assessment:** Implementing a robust pipeline to analyze multiple frames and select the single best quality image of a detected object, minimizing motion blur and ensuring clarity.
3.  **Optimized System Architecture:** Redesigning the software architecture for modularity, maintainability, and real-time performance through a multi-threaded approach.

The core objective is to define a clear development path for building this improved system, ensuring that captured images are of high quality and suitable for downstream tasks.

## 2. Core System Architecture

The system will be redesigned with a modular and threaded architecture to maximize performance and maintainability, based on the research in `ncnn_improvements.md`.

### 2.1. Modular Design

The application will be broken down into the following key Python modules:

*   **`camera.py`**:
    *   Responsibilities: Manages all interactions with the `Picamera2` library.
    *   Features: Camera initialization, configuration of resolution (potentially dual streams: a lower-resolution, higher-FPS stream for detection, and a higher-resolution stream for image saving), frame rate, and other camera-specific parameters (e.g., shutter speed, ISO if manual control is desired). Provides a consistent API for frame acquisition.

*   **`base.py` and `yolo_ncnn.py`**:
    *   Responsibilities: Handles all aspects of object detection.
    *   Features: `base.py` provides an abstract `BaseDetector` class defining a common interface (`load_model()`, `detect()`). The `yolo_ncnn.py` implements NCNN-specific logic, including model loading from `.param` and `.bin` files, pre-processing input frames, running inference, and post-processing detections (e.g., applying NMS if not handled by the NCNN graph). Will include model warm-up routines.

*   **`image_quality.py`**:
    *   Responsibilities: Provides functions to assess various quality aspects of an image frame.
    *   Features: Will contain discrete functions for:
        *   Blur detection (e.g., Laplacian variance).
        *   Motion/Stability analysis (e.g., frame differencing).
        *   Lighting assessment (e.g., brightness, contrast, histogram entropy).
        *   (Optional) BRISQUE/NIQE score computation if integrated.

*   **`frame_selector.py`**:
    *   Responsibilities: Implements the core logic for selecting the best frame from a sequence.
    *   Features: Takes a buffer of candidate frames (with associated detection and quality metadata) and applies a configurable scoring/ranking algorithm to determine the optimal frame to save.

*   **`capture_thread.py`, `inference_thread.py`, and `save_thread.py`**:
    *   Responsibilities: Manages the individual thread functions for the pipeline.
    *   Features: Each file contains the target function for its respective thread, handling the appropriate section of the processing pipeline.

*   **`main.py`**:
    *   Responsibilities: The main entry point of the system.
    *   Features: Orchestrates the initialization of all modules (camera, detector, etc.), starts the processing threads, and handles user interrupts (e.g., Ctrl+C) for a clean shutdown.

*   **`config_manager.py`**:
    *   Responsibilities: Loads and provides access to system configurations.
    *   Features: Reads parameters from a configuration file (e.g., `config.yaml`) such as model paths, detection thresholds, quality metric weights, queue sizes, camera settings, etc.

*   **`assistance.py`**:
    *   Responsibilities: Contains common helper functions.
    *   Features: Logging setup, timestamp generation, file path manipulation, etc.

*   **`performance_metrics.py`**:
    *   Responsibilities: Tracks and reports various performance metrics for the system.
    *   Features: 
        *   Real-time monitoring of system resources: VRAM usage, RAM consumption, CPU utilization, disk usage, and temperature
        *   Tracking of operation timings: detection time, capture time, latency
        *   Calculation of averages: FPS, latency, temperature
        *   Generation of a comprehensive performance report when the system stops
        *   Integration with the filename generation system to include performance metrics in saved image filenames

*   **`installations.py` and `verifications.py`**:
    *   Responsibilities: Setup and validation of the environment.
    *   Features: `installations.py` checks/installs required dependencies, and `verifications.py` verifies that all components (camera, models, etc.) are working properly.

The folders will include:
*   **`models/`**: Storage for YOLOv11n models (both PyTorch and NCNN formats)
*   **`images/`**: Storage for captured images in an organized directory structure

### 2.2. Threaded Pipeline

A multi-threaded pipeline will be implemented to decouple I/O-bound and CPU-bound tasks, enabling parallel processing and improving real-time responsiveness:

*   **Capture Thread (Producer)**:
    *   Operates at high frequency, driven by the camera's capabilities.
    *   Continuously grabs frames from `camera.py`.
    *   Places raw frames (or minimal metadata if using shared memory/efficient passing) into a bounded `raw_frames_queue` for the Inference Thread. This queue will be small to ensure recent frames are processed.

*   **Inference Thread (Consumer/Producer)**:
    *   Consumes frames from `raw_frames_queue`.
    *   Performs object detection using the loaded detector from `base.py`/`yolo_ncnn.py` (YOLOv11n NCNN).
    *   If a target object is detected with sufficient initial confidence, this thread will package the frame, detection results (bounding boxes, classes, confidences), and a timestamp.
    *   This package is then placed into a `detected_frames_queue` for the Quality Assessment & Selection Thread.

*   **Quality Assessment & Selection Thread (Consumer/Producer)**:
    *   Consumes frames and detection data from `detected_frames_queue`.
    *   Maintains a short-term buffer (e.g., last 5-10 relevant frames, or frames captured within a ~0.5-second window).
    *   For each frame in its buffer, it computes detailed quality scores using `image_quality.py`.
    *   The `frame_selector.py` logic is invoked on this buffer to identify the single "best" frame according to the defined selection strategy (see Section 3).
    *   Once the best frame is identified (e.g., at the end of an "event" or when a sufficiently high-quality frame appears), it's passed to a `frames_to_save_queue`.

*   **Saving Thread (Consumer)**:
    *   Consumes the selected best frame from `frames_to_save_queue`.
    *   Handles the actual disk I/O: naming the file (with relevant metadata), compressing (if needed), and writing the image to the designated storage path. This isolates slow disk operations from the main detection loop.

### 2.3. Model Strategy

*   **Primary Model:** YOLOv11n in NCNN format. This is chosen for its optimized performance on ARM CPUs, offering a significant speed-up over PyTorch inference on the Raspberry Pi.

*   **Model Conversion:** The `convert_model.py` script (PyTorch -> ONNX -> NCNN) will be used to prepare the NCNN model files (`.param` and `.bin`). These will be stored in a designated `models/` directory.

*   **Model Loading & Warm-up:** The NCNN implementation will load the model once at initialization and perform a "warm-up" inference pass on a dummy input to prepare caches and ensure consistent performance for subsequent real frames.

*   **Fallback Mechanism (Optional but Recommended):** As outlined in `update_model.py`, retain the ability to load and use the original YOLOv11n PyTorch model as a fallback if the NCNN inference fails or if a direct comparison is needed during development. The detector implementation can be designed to switch to the fallback if NCNN initialization or inference encounters critical errors.

## 3. Best Frame Selection Strategy

The core of the image quality improvement lies in intelligently selecting the best possible frame when an object of interest is detected. This strategy will be based on the research from `image_quality.md` and `ncnn_improvements.md`.

### 3.1. Triggering "Capture Mode" / Evaluation Window

*   When the `Inference Thread` detects a target object (tray, plate, hand) with a preliminary confidence score exceeding a configurable threshold (e.g., `initial_detection_confidence_threshold`), it signals the start of an "evaluation window."
*   Frames containing these initial detections are passed to the `Quality Assessment & Selection Thread`.

### 3.2. Frame Buffering in Quality Assessment

*   The `Quality Assessment & Selection Thread` will maintain a small circular buffer. This buffer will store a limited number of recent frames (e.g., 5-10 frames, or frames captured within a short time window like 0.3-0.7 seconds) that have passed the initial detection.
*   Each entry in the buffer will store the frame data along with its detection metadata and subsequently calculated quality scores.

### 3.3. Comprehensive Quality Metrics Calculation

For each frame entering the evaluation buffer, the following metrics will be calculated using `image_quality.py`:

*   **Mandatory Metrics:**
    *   **Detection Confidence:** The confidence score for the primary object of interest from the YOLO model (already available from the Inference Thread).

    *   **Image Clarity (Sharpness):** Calculated using the variance of the Laplacian. A higher score indicates a sharper image. Requires calibration of a "good" threshold.

    *   **Frame Stability (Motion Analysis):** Calculated using the mean absolute difference between the current frame and the previous frame (or a small local window of frames). A lower difference indicates less motion and a more stable scene, reducing motion blur. Focus on ROI-based stability if feasible.

    *   **Lighting Quality:** Assessed by a combination of average brightness (should be within an acceptable mid-range) and histogram entropy (higher entropy often indicates better contrast and detail).

*   **Recommended Optional Metric (Performance Permitting):**
    *   **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) / NIQE (Natural Image Quality Evaluator):** If computational resources allow without significantly impacting the frame processing rate, integrating BRISQUE or NIQE could provide a more holistic, single-score assessment of overall image quality, capturing various distortions including blur and noise. Its impact will be carefully evaluated.

### 3.4. Scoring, Ranking, and Selection

*   **Normalization:** Each individual quality metric score will be normalized to a consistent range (e.g., 0 to 1) to allow for equitable comparison and combination.

*   **Composite Quality Score:** A composite score will be calculated for each frame in the buffer. A weighted sum approach is proposed:

    `TotalQualityScore = (w_clarity * ClarityScore) + (w_stability * StabilityScore) + (w_lighting * LightingScore) + (w_confidence * DetectionConfidenceScore) [+ (w_brisque * BRISQUEScore)]`
    The weights (`w_clarity`, `w_stability`, etc.) will be configurable and tuned empirically to prioritize the most critical aspects (e.g., sharpness and confidence might receive higher weights).

*   **Best Frame Tracking:** The `frame_selector.py` logic within the `Quality Assessment & Selection Thread` will continuously track the frame with the highest `TotalQualityScore` currently in its active buffer ("best frame so far").

*   **Decision Point (End of Event):**
    *   An "event" (e.g., a hand appearing, interacting, and then retracting) needs a defined end. This could be:
        *   The target object is no longer detected for a certain number of consecutive frames (e.g., M frames).
        *   A fixed time window after the initial detection has elapsed.
    *   Once the event is considered ended, the "best frame so far" is selected from the buffer and sent to the `frames_to_save_queue`.

*   **Cooldown Mechanism:** After a frame is saved, a cooldown period will be enforced for the same object class (or a similar object instance based on tracking heuristics like IoU, if implemented) to prevent capturing numerous redundant images of a static or slowly moving object.

## 4. Performance, Stability, and Optimization

Achieving reliable real-time performance on the Raspberry Pi 5 with NCNN requires careful attention to system resources and optimization techniques, drawing from `ncnn_improvements.md`.

### 4.1. NCNN Specific Optimizations

*   **Compilation & Configuration:** Ensure NCNN is compiled/used with NEON (ARM SIMD) optimizations enabled. Configure NCNN to use an appropriate number of threads (e.g., 2-4 threads for the Pi 5's quad-core CPU) for its internal operations to parallelize workloads like convolutions.

*   **Precision:** Investigate and utilize FP16 inference if available and if it provides a speedup without unacceptable accuracy loss for YOLOv11n.

*   **Model Design:** Stick to the "nano" version (YOLOv11n) as it's designed for edge devices. Avoid larger variants.

### 4.2. Efficient Frame Handling and Preprocessing

*   **Input Resolution:** For detection, use a smaller input resolution (e.g., 640x480 or similar, matching the model's expected input like 640x640 after aspect ratio preserving resize). `Picamera2`'s ability to provide a separate, lower-resolution "preview" stream concurrently with a high-resolution "still" stream should be leveraged. The low-res stream feeds detection; the high-res stream is queried only when the `Saving Thread` needs to save the chosen best frame.

*   **Minimize Data Copies:** Be mindful of NumPy array copying between threads. Use efficient mechanisms for passing frame data.


### 4.3. Pipeline Stability

*   **Bounded Queues:** All inter-thread queues (`raw_frames_queue`, `detected_frames_queue`, `frames_to_save_queue`) will be of a fixed, small size. This acts as a natural backpressure mechanism. If a downstream thread is slow, the queue will fill, causing the upstream thread to block or (preferably for `raw_frames_queue`) overwrite the oldest frame with the newest, ensuring the system always works on the most recent data and prevents memory bloat.

*   **Frame Skipping:** The bounded `raw_frames_queue` (potentially of size 1, always holding the latest frame) inherently implements frame skipping if the inference rate is lower than the camera capture rate.

### 4.4. Performance Monitoring and Metrics Tracking

*   **System Metrics Tracking:** Use `performance_metrics.py` with `psutil` to continuously monitor and store:
    * **Resource Usage:**
        * VRAM Consumption (memory used by the NCNN model)
        * RAM Consumption (overall system memory usage)
        * CPU Utilization (percentage across cores)
        * Disk Usage (storage space utilized and available)
        * CPU Temperature (current, average, and maximum)
    
    * **Metrics:**
        * Total Captures (count of images saved)
        * Detection Time (time taken for model inference per frame)
        * Capture Time (time taken to save images to disk)
        * Average FPS (frames processed per second)
        * Average Latency (total processing time per frame)

*   **Logging:** Log these metrics periodically during runtime and generate a comprehensive report when the system is stopped.

*   **Real-time Display:** Optionally show key performance indicators on a debug display or overlay on the video feed.

*   **Throttling/Alerts:** If CPU temperature exceeds critical thresholds, the system could log warnings or even intelligently reduce processing load to prevent hardware damage.

*   **Hardware Considerations:** Ensure the Raspberry Pi 5 has adequate cooling (heatsink and fan) for sustained operation.

### 4.5. Camera Settings Optimization

*   Work with `Picamera2` to find optimal settings for:
    *   **Resolution:** As discussed (dual streams).

    *   **Frame Rate:** Aim for a consistent FPS from the camera that balances the need for temporal resolution with the processing capacity of the Pi.

    *   **Shutter Speed & ISO:** In well-lit environments, explore manually setting a faster shutter speed to reduce motion blur, potentially compensating with ISO if needed. Auto-exposure will be the default.

    *   **Focus:** Ensure the camera module's focus is correctly set for the expected object distance.

## 5. Image Storage and Management

Efficient and organized storage of captured images is crucial.

*   **Asynchronous Saving:** The dedicated `Saving Thread` ensures that potentially slow disk write operations do not block the real-time detection and analysis pipeline.

*   **Descriptive Filenames with Performance Metrics:** Captured images will be saved with filenames that embed useful metadata and performance metrics. The filename format will be:

    `[timestamp]_rc[ram_consumption]_dt[detection_time]_ct[capture_time]_fps[fps]_lat[latency].jpg`
    
    Example: `20231028T153045_rc128MB_dt45ms_ct12ms_fps8.5_lat120ms.jpg`

    This naming convention embeds key performance metrics directly in the filename, making it easy to analyze system performance by examining the captured images.

*   **Organized Directory Structure:** Images will be stored in a structured manner. A possible structure:
    `images/YYYY-MM-DD/[detected_object_class_main]/[filename.jpg]`
    This allows easy filtering by date and primary object.

## 6. Development Workflow and Verification

A structured development workflow will be followed to ensure robustness and facilitate debugging.

### 6.1. Incremental Implementation Plan

1.  **Environment Setup:** Ensure all dependencies (Python, OpenCV, NCNN prerequisites, Ultralytics for conversion, Picamera2, psutil) are installed correctly on the Raspberry Pi.

2.  **Basic Threading & Camera Integration:**
    *   Implement the camera interface and capture thread to grab frames and display them.
    *   Establish the basic multi-threaded structure.

3.  **NCNN Model Integration:**
    *   Use `convert_model.py` to convert YOLOv11n to NCNN format.
    *   Implement the detector classes to load the NCNN model and perform inference on frames from the Capture Thread. Display bounding boxes.

4.  **Performance Metrics Implementation:**
    *   Develop the `performance_metrics.py` module with functions to track and report all required metrics.
    *   Integrate it with the main processing pipeline to continuously monitor performance.

5.  **Individual Quality Metrics Implementation:**
    *   Develop and test each function in `image_quality.py` (Laplacian variance, frame differencing, lighting) independently using sample images and live camera feed.

6.  **Frame Selector and Scoring Logic:**
    *   Implement `frame_selector.py` with the buffering, scoring (initially with placeholder weights), and ranking logic.
    *   Integrate the Quality Assessment & Selection Thread.

7.  **Saving Thread and File Management:**
    *   Implement the Saving Thread with the new filename format incorporating performance metrics.

8.  **Full Pipeline Integration and Testing:**
    *   Connect all threads and queues.
    *   Thoroughly test the end-to-end pipeline with various objects and motion patterns.
    *   Verify that performance metrics are being accurately recorded and included in filenames.

9.  **Parameter Tuning and Optimization:**
    *   Empirically tune the weights for the composite quality score, detection confidence thresholds, cooldown periods, and queue sizes.
    *   Profile the system to identify and address any performance bottlenecks.
    *   Test BRISQUE/NIQE integration if pursued.

### 6.2. Configuration Management

*   A central configuration file (e.g., `config.yaml` or `config.ini`) will be used to manage all tunable parameters (model paths, confidence thresholds, quality metric parameters and weights, queue sizes, camera settings, logging levels, etc.). This will be handled by `config_manager.py`.

### 6.3. Robust Error Handling and Logging

*   **Error Handling:** Implement comprehensive `try-except` blocks within each thread's main loop and critical function calls. Log errors extensively.
*   **Fallback:** The NCNN to PyTorch fallback for inference (from `update_model.py`) should be implemented in the object detector if NCNN fails.
*   **Logging:** Utilize Python's `logging` module. Configure different log levels (DEBUG, INFO, WARNING, ERROR). Log important events, decisions, queue statuses, and performance metrics.

### 6.4. Verification Scripts

*   The existing `verification.py` script will be expanded to:
    *   Verify all Python dependencies.
    *   Check for the presence and integrity of model files (NCNN `.param` and `.bin`).
    *   Test camera connectivity and frame capture via `Picamera2`.
    *   Perform a basic test inference with the NCNN model.
    *   Optionally, run sample calculations for each image quality metric.
    *   Ensure essential directories (for saving images, storing models) exist or can be created.
    *   Verify the performance metrics tracking is working correctly.

## 7. Documentation and Future Work

*   **Inline Code Documentation:** Code will be well-commented, explaining complex logic and class/function purposes.

*   **This Document:** `implementation.md` will be kept as a living document, updated if strategies change during development.

*   **Potential Future Enhancements (Post-Core Implementation):**
    *   More sophisticated object tracking (e.g., using Kalman filters or simple IoU tracking) to improve cooldown logic and handle object re-identification.
    *   Remote monitoring interface (e.g., a simple web UI to view live feed and status).
    *   Adaptive quality thresholds based on ambient conditions.
    *   Training/fine-tuning a custom YOLOv11n model specifically on the target kitchen objects for even better accuracy.
    *   Integration with a database for storing image metadata and performance metrics for long-term analysis.
    *   Expanded performance metrics dashboard showing historical performance trends.

This implementation strategy provides a comprehensive roadmap for developing a robust and high-performing real-time object image capture system with integrated performance monitoring. 