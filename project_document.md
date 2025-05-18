# Project Requirements Document: Real-Time Object Image Capture via Raspberry Pi 5 and YOLO

## Introduction

This project aims to **capture images of specific objects** (e.g. trays, plates, hands) from a live video stream using a Raspberry Pi 5 and YOLO object detection. The system will run on a Raspberry Pi 5 with a camera module, using state-of-the-art YOLO models to detect target objects in real time. When a target object is detected in the video frame, the system will automatically capture and store an image of that object. 

Key considerations include choosing the right hardware, model, and deployment strategy to meet real-time performance needs, as well as organising captured images for scalability. This document outlines the tools, technologies, and requirements for the project, including comparisons of on-device vs. cloud inference, real-time performance factors, YOLO model version trade-offs, detection strategies, and data management practices.

---

## 1. Tools and Technologies Used

**Raspberry Pi 5 (Hardware Platform):** The Raspberry Pi 5 is a single-board computer that will serve as the edge device for this project. It features a quad-core 2.4 GHz Arm Cortex-A76 CPU (significantly faster than the Pi 4’s 1.5 GHz Cortex-A72) and is rated at *2–3× the processing power of Raspberry Pi 4*. This improved processing capability is critical for running object detection locally. 

The Pi 5 comes with 4 GB or 8 GB of RAM (with up to 16 GB in newer variants), providing sufficient memory for running neural networks and buffering video frames. It also includes a VideoCore VII GPU (supporting OpenGL ES 3.1/Vulkan 1.2) primarily for graphics; however, in this project the CPU will handle the YOLO inference (unless specialized accelerators are added). The Pi 5’s hardware advantages (faster CPU, better I/O and camera interface, optional PCIe for accelerators) make it suitable for edge AI tasks that were previously borderline on earlier models.

---

**Camera Module:** An official Raspberry Pi camera (such as the Camera Module 3 or HQ Camera) will capture the live video feed. This camera attaches via the Pi’s CSI camera interface. It supports high-definition video (e.g. 1080p30 or 720p60) and features like adjustable focus and HDR (in newer modules). For this project, camera settings will be tuned to balance resolution and frame rate for detection. For example, using a 1280×720 resolution at 30 FPS can provide enough detail for YOLO to detect objects while keeping frame rates reasonable. If motion blur is a concern (see Section 4), the camera’s shutter speed and exposure settings can be adjusted (or even a global shutter camera module used) to capture sharper images of moving objects. Good lighting or an IR illumination (if using IR-sensitive camera) may be employed to improve image clarity.

---

**YOLO Object Detection Models:** *You Only Look Once (YOLO)* is a family of state-of-the-art real-time object detection models. YOLO will be the core algorithm detecting trays, plates, and hands in each frame. YOLO models are known for their speed and accuracy, performing detection in a single pass through a neural network. The project will leverage the latest YOLO version (detailed in Section 5), which offers an optimal balance of accuracy and efficiency for deployment on the Pi 5. 

YOLO’s one-stage detector design makes it well-suited for live video; it can process frames sequentially and output bounding boxes and class labels for any detected objects. We will use pre-trained YOLO weights (fine-tuned on our specific object classes if needed) so that the system can recognise the target objects out-of-the-box. Notably, **Ultralytics YOLOv8** – the newest YOLO release – is designed for versatility and improved performance, featuring an anchor-free detection head and other architectural upgrades. YOLOv8 generally provides higher accuracy (mAP) at comparable or better speed than its YOLOv5 predecessor, making it a strong candidate for this project.

---

**PyTorch (Deep Learning Framework):** PyTorch is the underlying machine learning framework that will run the YOLO model. As an open-source deep learning library, PyTorch provides the necessary tools to load the YOLO model, perform inference on each video frame, and utilize CPU (or GPU, if available) efficiently. The choice of PyTorch is driven by its compatibility with the Ultralytics YOLO implementation (which is PyTorch-based) and its active support on Linux ARM platforms. On Raspberry Pi 5, PyTorch will run in CPU mode (the Pi’s GPU is not CUDA-capable for general neural network use). 

We will ensure a 64-bit OS is used on the Pi so that PyTorch is fully supported and can leverage the Arm Cortex-A76 CPU and NEON vector instructions for acceleration. PyTorch also allows options like TorchScript or ONNX export and quantisation which could be utilised to optimize the model (e.g. converting the model to 8-bit integers to speed up inference). These optimisations can be important on the Pi; for instance, running an int8 quantised YOLOv8 model on Raspberry Pi 5 has been shown to achieve roughly **~6 FPS** for the smallest model, a notable improvement for edge performance.

---

**Ultralytics YOLO (Library):** The Ultralytics YOLO Python library (often simply used via the `ultralytics` package) provides a high-level interface to YOLO models such as YOLOv5 and YOLOv8. We will use this library to simplify model loading and inference. Ultralytics offers a convenient API (and CLI) to load a YOLO model by name (e.g. `YOLO('yolov8n.pt')`) and run detection on images or video frames with a single function call. It comes with pre-trained weights for YOLOv8 models and supports customization such as setting detection confidence thresholds or NMS (Non-Maximum Suppression) settings. Using the Ultralytics library ensures we have the “latest and greatest” YOLO implementation optimized by the community – including any performance tweaks, and it also supports training custom models if we decide to fine-tune on a custom dataset of trays/plates/hands. The library is built on PyTorch and will utilize PyTorch under the hood. Given that YOLOv8 is an Ultralytics product, this library is the recommended way to deploy it. We benefit from ongoing updates and community support, as YOLOv8 is actively maintained with frequent improvements.

---

**OpenCV (Open Source Computer Vision Library):** OpenCV will be used for handling the video feed and image processing tasks. Specifically, we will use OpenCV’s `VideoCapture` interface to grab frames from the Raspberry Pi camera in real time. OpenCV supports the Pi camera via the `cv2.VideoCapture(0)` (with appropriate backend or using `libcamerasrc` on newer OS if needed) to retrieve frames as NumPy arrays. Once a frame is captured, OpenCV can also be used for any pre-processing required by the YOLO model – for example, resizing the frame to the resolution expected by the model (e.g. 640×640 square for YOLO by default) and converting color channels if needed (YOLO expects images in RGB). After detection, OpenCV can draw bounding boxes on frames (for debugging/visualization) and also handle saving images to disk (using `cv2.imwrite`). Its efficient image encoding/decoding is helpful for capturing and compressing images without heavy CPU overhead. OpenCV’s real-time capabilities and support for various formats make it a crucial part of the pipeline connecting the camera to the neural network and then to storage.

---

**Operating Environment:** The Raspberry Pi 5 will run a Linux-based OS (likely Raspberry Pi OS 64-bit “Bookworm” or Ubuntu for Pi). The environment will have Python installed along with the above libraries (PyTorch, Ultralytics, OpenCV). Ensuring these are properly installed on ARM (potentially via pip wheels or conda) is part of the setup. We also assume network connectivity for initial setup (to install packages), but the system at runtime can operate offline if doing local inference. Optionally, if remote monitoring or control is needed, SSH or VNC could be used during development, but they are not core to the project’s functionality.

---

## 2. Programming Language and Libraries

**Python:** The primary programming language for the project is Python. Python is chosen for its ease of development and rich ecosystem of libraries (like PyTorch and OpenCV) that have Python bindings. On an embedded platform like Raspberry Pi, Python allows rapid prototyping and integration of hardware interfaces and AI models without needing low-level code. The logic for capturing frames, running detection, and saving images will be written in Python scripts. 

Python’s simple syntax and dynamic typing make it easy to adapt the code as requirements evolve (for instance, adjusting detection thresholds or adding new object classes to detect). While Python is not the fastest language, the heavy-lifting (neural network inference and image processing) is done in optimised C/C++ libraries under the hood (PyTorch, OpenCV), so Python mainly orchestrates these calls. Given the real-time requirements, we will structure the code to be efficient (e.g. using vectorised operations from NumPy/OpenCV and avoiding heavy per-frame Python processing beyond the necessary logic).

---

**PyTorch (Python Library):** As mentioned, PyTorch is the deep learning library powering the YOLO model. We will use the `torch` library in Python to perhaps perform steps like model loading (if not entirely handled by Ultralytics), tensor manipulation, and possibly optimisations like moving the model to `float16` precision for faster CPU inference. PyTorch’s API will allow us to take an input frame (as a NumPy array from OpenCV), convert it to a PyTorch tensor, and pass it to the model’s forward pass. The output (detections) will be a PyTorch tensor or a structured object that we parse for bounding box coordinates and class labels. 

One advantage of PyTorch is the support for model quantisation and the existence of **ONNX** (Open Neural Network Exchange) export. In future, if performance on Pi’s CPU is not sufficient, we could export the model to ONNX and run it via a lighter runtime or even use TensorFlow Lite as an alternative (as one Medium reference did to achieve 6 FPS by running an int8 TFLite model[medium.com](https://medium.com/@elvenkim1/how-do-we-deploy-yolov8-on-raspberry-pi-5-d1c8be981c16#:~:text=python%20main,debug)). However, initially we stick to PyTorch inference for simplicity. PyTorch also benefits from any Arm-specific acceleration (for example, the PyTorch build might utilize OpenBLAS or NEON instructions for matrix ops). Python’s `pip` can install PyTorch on Raspberry Pi (there are pre-built wheels for ARM64 in many cases), or we may use a build from source if needed.

---

**Ultralytics YOLO (Python Package):** We import this package (often by `import ultralytics` or using `from ultralytics import YOLO`). This library simplifies interacting with YOLO models. For example, after installation, one can load a model with `model = YOLO('yolov8n.pt')` and then do `results = model(frame_array)` to get detections. The library handles preprocessing (resizing, normalisation) internally, though we can control aspects like the input size. It also provides the model files; YOLOv8n (nano) weights are around 3–4 MB and will be downloaded if not present. We’ll leverage Ultralytics’ documentation and defaults for things like confidence threshold (e.g. default 0.25) and NMS IoU threshold (e.g. 0.7) to filter detections. 

This library also supports training (`model.train()`) and exporting (`model.export()`) if needed. In our use-case, we primarily use the inference capability. Because Ultralytics regularly updates this library, using it ensures we can easily upgrade to newer YOLO versions or get performance improvements without rewriting code. It’s an essential library tying together Python, PyTorch, and the YOLO model in a user-friendly way.

---

**OpenCV (cv2 Python Module):** We use `opencv-python` library in our Python code. The `cv2.VideoCapture` class will initialize the camera. For Raspberry Pi camera modules, OpenCV can work either through the V4L2 drivers or using the **libcamera** stack (depending on the OS). We might use a command like `cv2.VideoCapture(0, cv2.CAP_V4L2)` or an appropriate pipeline string to get frames at the desired resolution and frame rate. Once frames are captured, OpenCV’s array is in BGR color by default, which we will convert to RGB (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`) before feeding to YOLO, as YOLO models are typically trained on RGB images. After detection, if we want to draw the results for debugging, OpenCV’s drawing functions (`cv2.rectangle`, `cv2.putText`) are useful. Most importantly, OpenCV allows us to **save images** easily with `cv2.imwrite(filename, image)`, which we will use when a target object is detected to save that frame (or cropped object image) to disk. 

OpenCV is highly optimised in C++ and can handle the 30 FPS video capture without issue on Pi (it can buffer and retrieve frames in a separate thread internally). We must be mindful of the fact that if our processing (YOLO) is slower than the incoming frame rate, frames might queue up or get dropped; OpenCV has ways to grab the latest frame to avoid lag (by reinitialising capture or reading continuously). We will consider such details to ensure minimal latency (discussed in Section 4).

---

**Other Libraries:** A few standard Python libraries will also be used. For example, **NumPy** is implicitly used with OpenCV and PyTorch (for converting arrays). We may use **time** (to timestamp images or measure FPS), **OS**/**path lib** (to create directories for image storage), and potentially **logging** (to log events like “captured image X”). If multi-threading or concurrency is needed (for example, a thread to save images while another thread continues detection), we might utilize Python’s `threading` or `asyncio`. However, to keep things simple at first, a sequential pipeline can be implemented, and if performance is sufficient (which we’ll evaluate), we may not need multi-threading. Python’s ability to interface with hardware (e.g., if a GPIO sensor triggers the capture) is a plus, although that is beyond the current scope (the capture is continuous here, not on external trigger).

In summary, the software stack is Python-based, with **PyTorch + Ultralytics YOLO** for machine learning and **OpenCV** for video and image handling. This combination has been proven in many similar projects and provides a robust foundation for real-time object detection on the Raspberry Pi.

---

## 3. Local Inference on Pi vs Offloading to Server/Cloud (Comparison)

There are two deployment options for running the object detection: **locally on the Raspberry Pi 5** or **offloading the computation to a remote server/cloud**. Each approach has trade-offs in terms of speed (FPS), latency, accuracy (which model can be used), and ease of setup/maintenance. The table below summarises the comparison:

| **Aspect** | **Local Inference (On Raspberry Pi 5)** | **Remote Inference (Server/Cloud)** |
| --- | --- | --- |
| **Processing FPS** | ~5–10 FPS using a small YOLO model (e.g. YOLOv8 Nano) on the Pi 5’s CPU. In practice, expect single-digit FPS without accelerators (e.g. ~6 FPS was observed with an int8 YOLOv8n on Pi 5). Higher FPS might be possible with smaller frames or quantisation, but real-time 30 FPS is likely not reachable on CPU. | 30–60+ FPS using a powerful GPU server with a larger model. A modern GPU (e.g. on a cloud VM) can run YOLO models at dozens or even hundreds of FPS. The effective FPS seen by the system might be lower due to network transfer overhead (see Latency), but the raw inference speed is much higher. |
| **Inference Latency** | **Low computation latency** (~100–200 ms per frame on CPU for tiny model). No network delay since everything is on-device, so response is immediate once processing is done. Total end-to-end lag can be kept ~0.1–0.3 s. This is important for real-time capture; the Pi can react without internet dependency. | **Higher network-induced latency:** Even if the server processes a frame in, say, 10–20 ms, sending the frame to the server and receiving results can add 50–100+ ms depending on network speed. Overall latency might be 0.2–0.5 s (or more) per frame. This lag could cause missed captures if objects move quickly. Network latency and reliability become critical factors (especially on Wi-Fi or Internet). |
| **Detection Accuracy** | Limited by using smaller models due to Pi’s resource constraints. To maintain some FPS, we likely use YOLOv8n or YOLOv5n (the “nano” models) which have lower accuracy (e.g. ~37% mAP on COCO) compared to larger models. However, if our specific objects are simple and distinct, a small model fine-tuned on them can still achieve high accuracy. | Can use much larger, more accurate models on the server. For example, a server could run YOLOv8l or YOLOv8x, which have higher mAP (~53–54% on COCO). This could yield better detection of small or hard-to-see objects. The server could also ensemble multiple models or use heavier algorithms if needed. In essence, cloud inference can maximize accuracy by not being limited in model size or compute power. |
| **Ease of Setup & Maintenance** | **Self-contained setup:** Requires installing dependencies on the Pi 5 (which can be a bit involved, e.g. building PyTorch or using pre-compiled wheels). Once set up, everything runs on one device. Simpler architecture (no need to manage network protocols or separate server code). The Pi can be deployed in the field with only power and camera. Maintenance involves managing the software on the Pi (updates to model or code). It’s relatively straightforward to debug on a single device. | **Complex distributed setup:** Requires a server or cloud instance with GPU and environment configured for YOLO. You need to implement a communication method (e.g. sending frames via REST API, WebSocket, or MQTT). Ensuring low-latency streaming can be complex. There’s added dependency on network and server uptime. Maintenance includes server costs, updates, and monitoring network usage. Initial setup may be easier in terms of installing deep learning frameworks (since cloud GPUs come with pre-configured environments often), but integrating with the Pi camera stream adds complexity. |
| **Other Considerations** | No external data transfer – good for privacy if the images are sensitive, and no bandwidth cost. Limited by Pi’s hardware (CPU only unless adding external accelerators). If future requirements grow, Pi might become a bottleneck. Scaling to multiple cameras means multiple Pis (which scale linearly in cost). | Virtually unlimited scalability – you can scale up the server or use cloud auto-scaling for multiple camera feeds. However, requires reliable internet. Potential recurring costs for cloud GPU time or server maintenance. Also, images are transmitted over network which could raise security/privacy issues if not encrypted. |

**Summary:** Running inference locally on the Raspberry Pi 5 is **more self-sufficient and has minimal latency** (crucial for real-time responsiveness), but you must use a lightweight model which may sacrifice some accuracy. Offloading to a server can **boost accuracy and throughput** using more powerful hardware, but introduces network latency and complexity in setup. In this project, given the aim is to capture images in real time as objects appear, a local inference approach is preferred to avoid delays – we don’t want to miss a fast-moving hand or a tray that appears briefly due to network lag. However, we remain open to a hybrid: for instance, doing preliminary detection on Pi and then sending cropped images to a server for higher-precision analysis if needed. But for the primary goal of triggering image capture, the **edge (on-Pi) inference is recommended** for reliability (it will work even offline and with consistent latency).

---

## 4. Real-Time Performance Requirements and Factors

Capturing images in real time implies that our system needs to process video frames quickly enough to react to relevant events (object appearances) without noticeable delay. Here we analyse the performance requirements, particularly **lag (latency)** and **image quality factors** that could impact detection and capture.

**Latency and Lag:** End-to-end lag is the delay from an object appearing in front of the camera to the system saving an image of it. We aim to minimize this lag. Several components contribute to latency:

- **Camera capture latency:** The time to expose the frame and deliver pixel data. At 30 FPS, a new frame comes every ~33 ms. The Pi camera has negligible additional lag in transmitting the frame, especially using the CSI interface. We should use a continuous camera mode to avoid initialisation delays.
- **Inference latency:** The time for the YOLO model to process a frame on the Pi. As discussed, a small YOLO model might take ~100–200 ms per frame on CPU. This is the dominant lag component in local inference. It effectively limits us to ~5–10 FPS processing even if the camera supplies 30 FPS. We may **skip frames** (process every nth frame) to ensure we always act on recent frames rather than queueing old frames, to avoid compounding lag. If using quantisation or model optimization, we might reduce this to ~100 ms or below for each inference.
- **Processing and saving overhead:** Minor additional time to, say, draw boxes or encode and write an image to storage. Writing a full-resolution image to SD card can take a few milliseconds to tens of milliseconds, depending on I/O speed. We might perform saving in a separate thread or buffer it so as not to stall the next frame’s capture.
- **Total expected lag:** We target an overall detection and capture latency around **0.2–0.3 seconds** or better. This means if a hand appears, within a third of a second an image is captured. Such lag is usually acceptable for capturing a relatively static object (like a tray placed somewhere). If the objects are extremely fast (e.g. a hand waving quickly), a higher frame rate and lower latency might be needed, but given the Pi’s constraints, we assume objects are somewhat momentary stable targets (e.g. a hand reaching in and pausing, or a plate being shown). In any case, we want to minimize lag to reduce the chance of capturing after the object has moved or left the frame.

---

**Impact of Motion Blur:** Motion blur occurs when the object (or camera) moves during the exposure time of the frame, causing the object to appear smeared. This can severely affect detection – a blurred object may not be recognized by the YOLO model (which was trained on relatively sharp images), and even if detected, the captured image will be low quality. On the Raspberry Pi camera, motion blur is more pronounced in low-light conditions where the exposure time is longer. To combat motion blur:

- We will ensure **adequate lighting** so that the camera can use a short exposure for each frame. The camera’s auto-exposure should ideally choose a high shutter speed when possible. In outdoor or bright indoor conditions this is fine; in dim conditions we might need additional lights.
- We can manually set the camera’s shutter speed (exposure time) if needed. For example, using the Pi camera2 library or direct camera controls, set a max exposure of a few milliseconds to freeze motion (this may require raising ISO or adding lights).
- The **frame rate** chosen also ties to exposure – at 30 FPS the maximum exposure time is 33 ms, but at 60 FPS it’s ~16 ms, which inherently reduces motion blur. If blur is a big problem, we could attempt 60 FPS capture and drop resolution, but then the processing of 60 FPS is not possible with YOLO on Pi. So we might stick to 30 FPS capture but ensure the exposure is much shorter than 33 ms (with a correspondingly dark image that we brighten with gain or lights).
- Another solution is using the **Global Shutter Camera** for Raspberry Pi (which captures the whole frame instantaneously rather than line-by-line). The Pi Global Shutter Camera can help reduce *rolling shutter artefact's* and slightly help motion blur. However, even a global shutter will blur if the object moves during the exposure, so shutter speed is still key.

In summary, controlling motion blur is vital for image quality. Otherwise, captured images of a moving hand could be unusably blurry. We will prioritise short exposure times and good lighting to get crisp images that the model can detect and that are worth saving.

---

**Camera Settings and Image Quality:** Apart from motion blur, other camera settings and conditions affect detection:

- **Resolution:** A higher resolution provides more detail for the model to detect small objects, but also means more pixels to process. YOLO models typically take inputs like 640×640. We can capture at a higher res (e.g. 1280×720) but then downscale to 640×640 for detection. Downscaling is usually okay, but if we have very small objects, we might need a higher network input resolution (YOLO can be configured to 960×960 or 1280×1280, at cost of speed). We will likely use 640×640 network input to start, which is standard and a good trade-off. The camera could capture at 720p or 1080p; higher than needed just wastes processing. Also, processing a smaller ROI (region of interest) is faster – if the area of interest is known (say objects will always appear in a certain zone), one could crop the frame to that zone to reduce work.

---

- **Color and exposure consistency:** YOLO models are somewhat robust to different lighting, but extreme shadows, glare, or color tints might reduce accuracy. We should ensure the camera’s white balance and exposure are tuned to the environment. For example, if working under fluorescent lights, lock the white balance to avoid shifts. If there is a bright light source causing glare (say a metallic tray reflecting light), position the camera to minimize that.

---

- **Camera frame rate vs processing frame rate:** If the camera outputs frames faster than we process, we will have a backlog. One strategy is to always grab the latest frame (flush the buffer) when the model is ready for a new frame. This way, we don’t process stale frames that introduce additional lag. OpenCV doesn’t automatically do this buffering flush, but one can read in a loop and only process the most recent.

---

- **System load:** Running neural nets is CPU intensive. We must monitor Pi 5’s CPU temperature and potentially enable cooling (heatsink/fan) to avoid thermal throttling, which would slow down performance and increase lag. The Pi 5 is more powerful but can run hot under sustained load. Keeping the system cool ensures consistent frame processing times.

---

- **Real-time constraints:** If the application absolutely required real-time (e.g. in a feedback control system), one might say the system should operate at a minimum of e.g. 10 FPS with under 100 ms latency. For our purpose of image capture, we have *some* leeway (if a frame or two is missed, it’s not catastrophic, as long as we capture one good image of the object when it appears). Therefore, we define our real-time requirement as *“capture at least one clear image of the target object within its first second of appearance”*. The system meeting ~0.2–0.3 s latency and ~5–10 FPS processing will fulfil this.

---

To illustrate, imagine a use case: a hand enters the frame to place a plate on a table. Our camera (30 FPS) and YOLO detection (say 5 FPS effective) will perhaps process every 6th frame. If the hand is in view for 2 seconds, that’s ~10 frames to analyse. The first detection might trigger at 0.2–0.5 s after the hand appears, and we save that frame. If the hand moves quickly, we might capture it slightly later, but hopefully still in frame. Good lighting and short exposure will ensure that saved image is not blurry. Conversely, if we had a 1–2 second lag (as could happen with a slow model or network delay), the hand might have left by the time we save an image, or be blurred across multiple frames – which we clearly want to avoid. Thus, all these measures (fast model, local processing, camera tuning) are geared toward **minimising delay and maximising the chance of a clear capture.**

---

**Light Conditions:** Lighting deserves emphasis: low light can introduce **noise** (high ISO grain) and force longer exposures causing blur. The Pi camera modules have small sensors that aren’t great in low light. If our deployment is in a dim area, we should add illumination (even an LED light) to improve detection reliability. High dynamic range scenes (very bright and dark areas) might also confuse exposure – if our object is in a shadow but background is bright, the camera might underexpose the object. We could use exposure compensation or spot metering on the area of interest. The new Camera Module 3 supports HDR mode which can help in high dynamic range scenes, though that might reduce frame rate. We will test the system in the expected lighting conditions and adjust parameters accordingly to ensure consistent image quality for the YOLO model.

---

In summary, **real-time performance** is achieved by keeping the detection loop efficient and reducing any sources of lag. The Pi 5’s improved CPU helps, but we also manage frame acquisition and processing carefully. Simultaneously, we address image quality factors like motion blur and lighting so that the model has the best chance to detect the object quickly and correctly. By balancing these factors, the system will meet the real-time requirement of promptly capturing images when target objects appear.

---

## 5. Comparison of YOLO Versions (v5, v7, v8) – Performance vs Resource Efficiency

There are several versions of the YOLO model family available. We focus on YOLOv5, YOLOv7, and YOLOv8 – all of which are capable of our object detection task, but with differences in architecture, speed, and resource demands. Below, we compare these versions, especially in the context of running on a resource-constrained device like Raspberry Pi 5, and provide a recommendation on which to use.

**Performance vs Resource Trade-offs:** In choosing a YOLO version for the Pi, the main factors are model size, inference speed on CPU, and accuracy. To illustrate the trade-offs, consider the **smallest models** of each: YOLOv5n (1.9M params, ~28% mAP), YOLOv7-tiny (~6M params, 37% mAP), YOLOv8n (3.2M params, 37% mAP). YOLOv8n stands out by achieving the ~37% mAP of the larger tiny models while being in between their size. It’s about twice the params of YOLOv5n but delivers much higher accuracy. Meanwhile, YOLOv7-tiny and YOLOv8n have similar accuracy, but YOLOv8n is smaller and tested to be quite fast. 

---

**For edge devices, YOLOv8n currently offers the best accuracy-per-compute ratio.** On the other hand, if we look at **maximum accuracy**: YOLOv7 (X) and YOLOv8x are both around 52–54% mAP. YOLOv8x slightly edges out in accuracy (53.9 vs 52.9) but runs a bit slower than YOLOv7x in some benchmarks. However, these are far too slow for Pi CPU (would be <1 FPS), only relevant if using a GPU server. YOLOv5’s largest wasn’t as high in accuracy.

We can visualize the performance of these YOLO versions in terms of speed vs accuracy (from an example benchmark on Jetson Orin, which correlates with general efficiency):

![image.png](attachment:acd3e0b0-0cbd-4b83-bfb5-2364ad2d0c32:image.png)

In the above figure, higher and to the right is better (more accuracy, more FPS). We can see YOLOv8 (red) models form an upper-right frontier in many cases, meaning they are very competitive. YOLOv5 (blue) models tend to have lower accuracy at similar speeds. YOLOv7 (green) gave a boost in accuracy over YOLOv5 but YOLOv8 then matched or exceeded that.

**Recommendation:** For this project, **YOLOv8** is strongly recommended as the primary model family. Specifically, **YOLOv8-nano (yolov8n)** is a great starting point on Raspberry Pi 5 due to its tiny size and relatively high accuracy for that size. It will likely run at a few FPS on the Pi, which is acceptable for our needs, and can detect the objects of interest with reasonable accuracy. If we find that we need slightly more accuracy and can tolerate a bit less speed, we could try **YOLOv8-small (yolov8s)** – but it has ~3× more parameters and would be slower (possibly 2–3 FPS on Pi CPU). YOLOv5n or YOLOv5s are alternatives if, for example, YOLOv8 was not an option, but given Ultralytics supports YOLOv8 seamlessly, there’s no strong reason to stick with the older v5. YOLOv7-tiny could be used, but integration is more manual, and YOLOv8n can fill the same niche with easier integration.

---

We also consider that in the future, if we offload to a server, we can simply switch to a larger YOLOv8 model (like yolov8m or l) for higher accuracy, without changing code—another benefit of staying within the Ultralytics YOLOv8 ecosystem. YOLOv8’s versatility (support for tasks like segmentation or pose) is a bonus, though not needed now, it means we have the option to extend functionality (e.g. hand pose estimation) down the line with the same framework.

In conclusion, **use YOLOv8-nano for on-Pi detection**. It gives the best balance of performance and resource usage for Raspberry Pi 5, and the Ultralytics/PyTorch stack supports it well. YOLOv8’s proven improvements in both speed and accuracy make it the top choice for a fast and reliable object detection framework.

---

## 6. Single-Object vs Multi-Object Detection Strategy

Our system needs to handle scenarios where either a single target object or multiple target objects appear in the camera view. The strategy for detection and image capture will differ slightly based on whether we are focusing on one object at a time or multiple objects simultaneously:

- **Single-Object Detection Scenario:** In some cases, the user might only be interested in one specific object class at a time. For example, perhaps initially the project is only capturing images of trays (and not plates or hands). In such a scenario, we can simplify the pipeline:
    - **Model and class filtering:** We could either train the model on only that one class or use a multi-class model but filter out detections of other classes. Using a single-class model (by training YOLO on just trays) might improve detection accuracy slightly for that class and reduce model size, but given YOLO’s small models are already efficient, it might not be necessary to retrain – we can simply ignore other classes in code. The Ultralytics YOLO allows specifying class IDs to filter, ensuring we only react to “tray” detections.
    - **Logic:** Each frame, we look for a detection of the target object. If found and meets confidence threshold, we trigger image capture. We might also want to ensure we don’t capture duplicate images of the same instance. For example, if a tray stays in view for 50 frames, we probably should only capture it once. A strategy to handle this is to implement a **cooldown or tracking** – e.g. once a capture is done, wait some seconds or until the object leaves frame before capturing that class again. Alternatively, use simple object tracking (by bounding box position) to see that it’s the same tray and avoid recapturing it repeatedly.
    - **Framing the capture:** In single-object scenario, we might choose to **crop the image to the object** rather than save the entire frame. Since only one thing is of interest, cropping the tray out and saving just that can produce a focused dataset of tray images. However, cropping in real-time must be done carefully (with correct bounding box coordinates and some padding perhaps). If the requirement is to capture the whole scene with the object, then we keep the full frame.
    - **Performance:** Focusing on one class can slightly reduce overhead (less post-processing on other detections), but the model still examines the whole frame. We could restrict detection to a region of interest if, say, we know the tray will only appear in a certain area of the image – this can cut computations by cropping input frames to that area.
    
    ---
    
- **Multi-Object Detection Scenario:** Here, the system should detect and capture images of **multiple different object types**, possibly appearing together. For instance, a hand and a plate might appear in the same frame (a hand holding a plate), and we are interested in capturing images of both the hand and the plate.
    - **Model setup:** We will use a YOLO model trained on all relevant classes (trays, plates, hands – and any other we anticipate). YOLO is inherently multi-class: it can predict several classes in one pass. We ensure the model’s output includes labels for each class we care about. If using a pre-trained model, we may map our target objects to existing classes if appropriate (e.g. “hand” might be detected as a person or part of person – better would be custom training to detect hand as its own class; “plate” and “tray” likely need custom data unless the model was trained on similar objects). Assuming we fine-tune YOLO on these objects, it will detect all in one frame.
    
    ---
    
    - **Simultaneous detections:** When multiple objects are detected in one frame, we have to decide how to handle image capture. There are a couple of approaches:
        1. **Capture entire frame with annotations:** We could simply save the whole frame image and record that it contains, say, a tray and a plate (maybe by encoding the labels in the filename or a metadata file). This way one image might have multiple objects.
        2. **Capture individual objects (cropped out):** We could save a separate image for each detected object. For example, if a hand and a plate are in the frame, we output two image files: one cropped around the hand, one around the plate. This yields focused images per object, but we should ensure the cropping includes the whole object (maybe add a small margin). YOLO gives bounding boxes, which we can use to crop from the original high-resolution frame.
        3. **Hybrid:** Save the full frame for context and also crop each object as separate images for a more granular dataset.
        
        ---
        
    - **Handling overlaps and priority:** In multi-object frames, objects may overlap (e.g. a hand might occlude part of a plate). The YOLO model will do its best via NMS to give separate boxes. Our logic might need to consider if one object is essentially a part of another (is a hand holding a plate two captures or do we treat it as one event?). Likely, since the requirement is to capture images of each specific object, we treat them separately even if in the same scene.
    
    ---
    
    - **Avoiding duplication:** As with single-object, we don’t want to save the *same* object every frame. For example, if a plate is stationary on a table, we should perhaps capture it once, not 30 times. We can use simple heuristics: if an object (class + position) has been detected in consecutive frames, we skip capturing until it either moves significantly or disappears and reappears. Implementing an **object tracker** (like OpenCV’s MOSSE or CSRT tracker, or even using YOLO detections over time) can assign IDs to objects. However, a simpler approach is to check the IOU (intersection over union) of detected boxes with the last saved detection – if it’s very high (meaning it’s the same object in place), skip saving again. This prevents flooding with duplicates.
    
    ---
    
    - **Performance considerations:** Detecting multiple objects doesn’t inherently slow down YOLO (it processes the image in one go regardless of number of detections). But if we choose to crop and save each object, that adds a bit of overhead per detection (cropping and writing file). If many objects (say 5+) were in frame, writing 5 images could briefly stall the loop. We can mitigate this by queuing the save operations. In our context, likely it’s at most 2–3 objects (e.g. a hand, a plate, maybe a tray all at once).
    
    ---
    
    - **Scalability:** Designing for multi-object means our code should handle a dynamic list of detections each frame rather than a single detection. We will iterate through detection results and handle each. It’s straightforward with Ultralytics output (each result has a list of boxes with class IDs).
    
    ---
    

**Example strategy in practice:** Suppose the camera sees a tray and a plate on it, and a hand reaches in. The YOLO model (trained on all three classes) detects all three. Our system could then:

- Save an image of the **tray** (cropped to tray’s bounding box or the whole frame labeled “tray”).
- Save an image of the **plate**.
- Save an image of the **hand** (note: if the hand is just a part of a person, our model would specifically detect “hand” only if trained; we might need to train on hand images).
We should name these images distinctly (more on naming in next section). If the plate and tray are static and remain in the next frames, we should not save them again every frame. But if the hand moves to a new position or a new hand appears, capture that accordingly. In essence, multi-object support means being prepared to handle **multiple detections per frame** and managing the capture logic for each independently.

---

**Single vs Multi – Model Choice:** Another angle: one could use separate models for different objects (e.g. a specialized model for hands vs a different model for trays) and even run them in parallel if needed. However, on a Pi, running multiple models would be too heavy. It’s more efficient to have one model that handles all classes at once. The YOLO model’s capacity to handle multiple classes is beneficial here. So we will stick to a single model that knows all target classes.

**Thresholds per class:** In some cases, one class might be harder to detect than others, so confidence scores may differ. YOLO outputs a confidence per detection. We might choose a higher threshold for one class if it tends to produce false positives, and a lower threshold for another class if it’s harder to catch. The Ultralytics library allows setting a confidence threshold globally; customising per class might require filtering the results manually. This is an advanced tweak if needed to balance precision/recall per object type.

**Conclusion for strategy:** The system will be designed to **gracefully handle both single and multiple target objects**:

- If only one type is enabled, it will focus on that, simplifying output.
- If multiple types are expected, it will detect all in one pass, then for each detection trigger a capture (with de-duplication logic to avoid redundant images).
- This ensures we build a comprehensive image dataset of each object type, whether they appear alone or together.

---

**Conclusion:** This project brings together a powerful yet efficient hardware setup (Raspberry Pi 5 + camera) with advanced object detection software (YOLOv8 via PyTorch) to achieve real-time image capture of specific objects. We have detailed the tools and technologies to be used, weighed local vs cloud inference options, addressed real-time performance factors like lag and image quality, compared YOLO model versions to select the most appropriate one, and outlined strategies for both single-object focus and multiple-object handling. We also described how to manage and organize the captured images for future scalability. By following these requirements and guidelines, the implementation should result in a robust system capable of detecting target objects on-the-fly and archiving their images in an efficient manner. This sets the stage for building valuable datasets or feeding downstream processes (like analytics or alerts) with minimal human intervention, leveraging edge AI on the Raspberry Pi and the latest in object detection technology.