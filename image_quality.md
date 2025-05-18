# Optimizing Real-Time Frame Selection for Clear Waste Item Images

Capturing a clear, non-blurry image of a moving waste item on a Raspberry Pi 5 is challenging due to motion and limited processing power. The system operates on a 15–30 FPS video feed, and the current approach scores each frame by four metrics—detection confidence, frame stability, image clarity, and lighting—to decide if the frame is worth saving. In this report, we evaluate how effective these metrics are, suggest improvements and additional quality metrics, and propose a real-time decision pipeline for selecting the best frame. We also discuss the trade-offs between accuracy and computational feasibility on the Raspberry Pi 5, concluding with comparison tables for different metrics.

---

## Current Frame Quality Metrics and Enhancements

**1. Detection Confidence Score:** The detection confidence is obtained from the YOLO model’s output for the target object (in this case, the waste item or the person handling it). A higher confidence implies the model is more certain about the object’s presence and usually correlates with the object being clearly visible (e.g., not occluded or motion-blurred). This metric is effective in filtering out frames where the object is barely detectable or mostly out of frame. However, *confidence alone is not a direct measure of image clarity* – an image can have a high confidence detection even if slightly blurry, as long as the model still recognises the object. To enhance this metric:

- **Use Consecutive Detections:** Require a high confidence for several consecutive frames to ensure the object is steadily detected and likely fully in view (reduces chance of saving a momentary false positive or partial glimpse).
- **Incorporate Bounding Box Size:** Larger bounding boxes (to an extent) may indicate the object is closer and clearer in the frame. If the waste item appears larger (without being blurred), it might be a good frame to capture.
- **Class-Specific Thresholds:** Use a higher confidence threshold for saving an image than for detection; e.g., detect at 50% confidence but only capture a frame if confidence > 80%, ensuring only clear, unambiguous detections trigger a save.

---

**2. Frame Stability Score:** The current implementation uses the mean absolute difference between the current frame and the previous frame to estimate stability. A low difference (below a threshold) suggests the scene is not changing much, implying either the camera or object motion is minimal – a stable frame likely has less motion blur. This is a simple proxy for motion: if the waste item is moving fast, consecutive frames will differ a lot (high difference score); if it pauses or moves slowly, differences are small (low score). In real-time usage, this metric helps avoid saving frames during rapid motion. However, there are some nuances and potential improvements:

- **ROI-Based Stability:** Instead of the whole frame difference, consider computing difference only in the region of interest (ROI) around the detected object. This focuses the stability score on the waste item itself. For example, if the background is moving (camera shake or flicker) but the object is momentarily still, the ROI stability would catch that even if full-frame difference is higher.
- **Optical Flow for Motion:** For a more robust motion estimation, lightweight optical flow algorithms (e.g., block matching or Lucas-Kanade with few points) can estimate how fast and in what direction the object is moving. If the flow magnitude is high, the frame likely has motion blur. Optical flow is more computationally intensive than a simple frame diff, so the trade-off is complexity vs. precision.
- **Temporal Smoothing:** Use a short sliding window (e.g., 3–5 frames) to smooth the stability score. You could average the frame differences over a few frames to avoid reacting to a single anomalously stable frame in the middle of motion. Only if the average motion drops below a threshold for some duration do we consider the scene stable.

---

**3. Image Clarity (Sharpness) Score:** Currently, image clarity is assessed via a blur detection using the variance of the Laplacian of the image. This produces a **high score for sharp images** (lots of high-frequency detail) and a **low score for blurry images**. A threshold like 100 is used to decide if the image is sharp enough. In practice, this is an effective and fast measure: computing the Laplacian’s variance is a simple convolution and variance calculation, which is very feasible in real-time. However, a fixed threshold (e.g., 100) may not generalize to all scenarios; it depends on camera resolution, focus, and scene details. Enhancements for clarity scoring include:

- **Adaptive Thresholding:** Instead of a hard-coded 100, calibrate the blur threshold dynamically. For instance, during a startup calibration, capture a clearly focused image of a sample object to measure its Laplacian variance as the “100% sharp” reference. Set the threshold as a percentage of that reference. This accounts for the specific camera lens and sensor characteristics.
- **Multi-scale Clarity Check:** Apply the Laplacian variance on multiple scales of the image (downsampled versions) – similar to multi-scale SSIM concept – to ensure the image is uniformly sharp, not just containing a single sharp edge. This helps because an image might have one small in-focus region (high local variance) but the rest is motion-blurred.
- **Alternate Sharpness Metrics:** The Sobel operator (measuring gradient magnitude in X/Y directions) can also be used for a focus measure. There’s also the Brenner gradient or Tenengrad method – these are all in the same family of high-frequency content measures. They might not vastly outperform Laplacian variance, but if one shows more consistency with your camera’s kind of blur, it could be used. Overall, the Laplacian variance method has been found simple yet effective for scoring clarity, as long as the threshold is chosen wisely.

---

**4. Lighting Score:** The lighting metric currently combines average brightness and image entropy to evaluate exposure. It computes how close the frame’s overall brightness is to a mid-range (not too dark or bright) and the entropy of the brightness histogram to gauge contrast. This composite score ranges between 0 and 1 (higher means better lighting). In real time, this helps ensure the frame is neither underexposed (dim) nor overexposed (bleached out) and has a decent contrast distribution. Effective lighting is crucial for image quality and also aids the object recognition model. Possible improvements to the lighting metric:

- **Highlight/Shadows Check:** In addition to entropy, check if a significant portion of the image is in very high or very low intensity (e.g., count of pixels near 0 or 255). If yes, the image might have blown-out highlights or deep shadows that reduce useful detail. The current entropy measure partially captures this (a low entropy could mean most pixels are clustered at extremes).
- **Color Balance (White Balance):** If color information is important (e.g., identifying the type of waste by color), a metric for color balance could be introduced. However, since the detection is done in RGB and presumably the model accounts for color, adjusting white balance might be handled at the camera level rather than in scoring frames.
- **Adaptive Exposure Threshold:** Similar to blur, you might not want a fixed threshold (0.5 as in code) for all environments. If the system is deployed outdoors vs indoors, the expected “good” brightness varies. The system could learn the ambient lighting by sampling a few frames with no motion (assuming background) to set a baseline for normal exposure. Then lighting score can be the deviation from this baseline optimal exposure.

---

**Effectiveness of Current Metrics:** In summary, the four metrics complement each other. The **confidence score** ensures relevance of the frame (the object is present and recognized). The **stability score** and **clarity score** together address motion blur and focus issues – stability catches motion blur from movement, clarity catches general blur (out of focus or motion). The **lighting score** filters out frames that would be clear but too dark/bright to see details. The current approach requires all conditions to be met (confidence high, blur low, etc.) before capturing, which is a conservative strategy to only save high-quality frames. One enhancement to consider is a *weighted scoring system*: instead of strict pass/fail thresholds for each, compute a single composite quality score. For example, one can weight each normalised metric (after appropriate scaling) and sum them: `Quality = w_conf*Conf + w_blur*BlurScore – w_motion*MotionBlur + w_light*LightScore`. This way, a frame that is excellent in three metrics but just slightly under one threshold could still be selected if the overall score is highest. This would require careful tuning but could maximize the chance of getting the best shot.

---

## Additional Frame Quality Metrics for Real-Time Assessment

Beyond the current four metrics, there are established image quality assessment (IQA) metrics which can help judge frame quality:

- **SSIM (Structural Similarity Index):** SSIM is a full-reference metric, meaning it compares two images – typically used to compare a compressed image with the original to measure quality loss. In our real-time scenario, we lack a pristine "reference" image of the waste item for comparison. However, one creative way to use SSIM could be comparing a frame to the previous *good* frame as a reference, essentially checking how similar the structure is. A clear frame of the object should have similar structure to another clear frame, whereas a motion-blurred frame might diverge in structure (details smeared out). This is a bit contrived and may not be very reliable unless the object holds relatively still in two frames. SSIM is moderately fast to compute (it involves windowed means and variances over the image), but applying it in real-time on 720p frames on a Pi may be borderline. If used, it should be on a downscaled image or ROI. **In practice, SSIM might not be the first choice here due to the lack of a true reference image.**

---

- **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator):** BRISQUE is a no-reference IQA metric that uses a model trained on natural images with varying distortions and their human opinion scores. It extracts statistical features from the image (NSS features) and through a learned SVR (support vector regressor) model, outputs a “quality score.” Importantly, *lower BRISQUE scores indicate better quality*. BRISQUE is good at capturing various distortions like blur, noise, and compression artefacts in a single score. For our application, BRISQUE could directly rate each frame’s quality without needing a reference. It has been noted that **BRISQUE has very low computational complexity, making it suitable for real-time applications**. After an initial model load, computing BRISQUE on a 720p image might take on the order of a few tens of milliseconds, which could be acceptable on Raspberry Pi 5 if not done every single frame (or done on a smaller cropped region around the object). The downside is that BRISQUE’s model might not specifically focus on motion blur – it’s generic quality – but generally, a motion-blurred or very noisy image will indeed get a worse (higher) BRISQUE score than a clear one. Implementationally, OpenCV’s contrib. module has `cv2.quality.QualityBRISQUE_compute()` which can compute this score after loading a trained model. Using BRISQUE in the pipeline could give a more holistic quality measurement beyond our handcrafted metrics. It could potentially replace the combination of blur+lighting with one number. But we must test whether it’s fast enough on the Pi; perhaps computing it only on candidate frames (not every frame) is a strategy.

---

- **NIQE (Natural Image Quality Evaluator):** NIQE is another no-reference metric similar to BRISQUE but “opinion-unaware.” It doesn’t use subjective human ratings in its model; it’s trained only on pristine images and looks at how “natural” an image’s statistics are. NIQE also yields a quality score (again, typically lower means better quality). Its advantage is that it doesn’t need a distortion-specific training set; it can handle arbitrary distortions. However, the trade-off is that NIQE’s scores may not correlate with human perception as well as BRISQUE’s, since NIQE isn’t calibrated with subjective opinions. NIQE might flag an image as low-quality even if it’s subjectively fine but just not “natural” according to its model. In the context of a waste item being thrown, NIQE could be used similarly to BRISQUE to assess each frame. Computationally, NIQE is also designed to be efficient (similar feature computation without the SVR step). If BRISQUE and NIQE models are both available, one could test which correlates better with “blurry vs sharp” in our scenario. BRISQUE might edge out since motion blur is a common distortion humans would rate poorly, which BRISQUE would have learned.

---

- **PIQE (Perception-based Image Quality Evaluator):** PIQE is another no-reference metric that is opinion-unaware and does not require training. It works by dividing the image into blocks, detecting distorted blocks, and computing a quality score from the variance of those distortions. PIQE often performs similarly to NIQE and is also good for arbitrary distortions. However, PIQE is **less computationally efficient** than BRISQUE/NIQE– it does a more exhaustive analysis (since it can even output a local quality heat map). On a Raspberry Pi, PIQE might be too slow to run on each frame (perhaps feasible on smaller resolutions or if only run occasionally). Unless we specifically need to locate which part of the frame is low-quality (which might not be necessary if we just want the best frame), PIQE’s extra work isn’t crucial. We mainly need a single quality score per frame.

---

- **Other Metrics & Methods:** There are a few other approaches worth mentioning:
    - **Contrast and Noise Metrics:** A frame might not be motion blurred but could be noisy (especially in low light). Metrics like signal-to-noise ratio (SNR) or simply measuring noise level (e.g., using a filter to detect noise variance) could be useful if noise is a concern. The Raspberry Pi camera might introduce noise in low light; BRISQUE would catch that as reduced quality, but one could also explicitly ensure the chosen frame isn’t extremely noisy.
    - **Edge Density or Texture**: Counting the number of strong edges or texture details in the ROI can serve as a simpler proxy for clarity. A clear image of a complex object will have more detectable edges than a blurry one. This is somewhat related to Laplacian variance but could be done via Canny edge count or similar.
    - **Deep Learning based IQA (e.g., NIMA):** NIMA (Neural Image Assessment) is a learned model (usually a CNN) that outputs an aesthetic/quality score. While effective, running a CNN just for image quality on a Pi may be too slow (unless a tiny model or using hardware acceleration). There are also some specific blur detection CNNs, but again the cost might not justify it when simpler methods suffice.
    - **Multi-Frame Assessment:** Another strategy is to leverage multiple frames to assess quality. For example, one could compute the **Structural Similarity (SSIM)** or just a simple difference between a given frame and its adjacent frames – a sharp frame will be more similar to its neighbour's than a very blurry frame (since a blurry frame loses unique texture that sharp frames of the same scene would share). This is not a standard metric per se, but a heuristic: if frame N is sharp and frame N+1 is also sharp, they will look alike; if frame N is blurred due to fast motion, it might look significantly different from frame N+1 (which might capture the object at a slightly shifted position with clarity). This method would require careful tuning and might be unreliable if the object is constantly moving (then *all* frames differ a lot).

---

When considering these additional metrics on a low-power device, **BRISQUE and NIQE stand out** as promising options because they are designed to be computationally efficient and no-reference. Indeed, *no-reference metrics often outperform full-reference ones in correlating with human judgment for real-world images*– which suits our case where human-like judgment of “is this frame good enough?” is the goal. If implementation complexity is a concern, one could start with the existing four metrics and perhaps add **BRISQUE** as a fifth metric for a holistic score. If BRISQUE is too slow to run on every frame, one strategy is to run it only on frames that already passed the basic checks (confidence, stability, etc.), as a final gate to choose the best of the good frames.

---

## Real-Time Decision-Making Pipeline for Frame Selection

Designing a real-time pipeline requires balancing speed and the need to evaluate frames from a continuous stream. Below is a recommended decision-making logic to pick the best frame as the waste item is being thrown:

1. **Continuous Frame Acquisition & Detection:** The camera feed (15–30 FPS) is continuously read. Each frame goes through the object detection model (YOLO). This is the most expensive step, so it may effectively cap the frame rate. On Raspberry Pi 5, YOLOv8s or a similarly small model might achieve around 10–15 FPS at 720p with CPU-only inference (this is an estimate based on Pi 5’s specs and known performance; using the Pi’s GPU via OpenCL or a Coral accelerator could improve this). If no relevant object is detected in a frame, the frame is immediately discarded and not processed further for quality (this saves computations – only analyse frames that *might* contain the target).

---

1. **Triggering Frame Evaluation Window:** When a waste item (or the person about to throw it) is detected and confidence exceeds a threshold, the system should trigger a “capture mode” for the upcoming frames. In practice, this could mean: once detection confidence > X%, start buffering frames and computing their quality metrics. The idea is that as soon as the event of interest (throwing) starts, we anticipate the best frame will occur within the next second or two. We don’t know exactly which frame will be the best, so we temporarily evaluate a bunch of them.

---

1. **Maintain a Buffer of Recent Frames & Scores:** Keep a small circular buffer of the last N frames (for example, N=5 or N=10, which at ~15 FPS corresponds to about 0.3–0.7 seconds of video). For each frame in this buffer, compute the quality metrics of interest:
    - Confidence (already from detection).
    - Stability (inter-frame motion) – this could be relative to the previous frame in the buffer or some baseline frame from the start of the capture mode.
    - Clarity (Laplacian variance or other sharpness measure).
    - Lighting (exposure score).
    - Any additional metric like BRISQUE if feasible (maybe computed for every second frame or something to lighten load).
    Store these metrics alongside the frame in the buffer.
    
    ---
    
2. **Choose the Best Frame on the Fly:** Continuously determine which frame in the buffer currently has the highest overall “quality” based on the metrics. This could be done by a composite score as discussed, or a simple hierarchical check (like the current code uses thresholds to qualify a frame). A refined approach could be:
    - Among frames that meet minimum criteria (e.g., confidence > 0.5, blur score > threshold_low, etc.), pick the one with the highest sum of normalised scores.
    - Alternatively, if using thresholds strictly, capture the first frame that meets all the criteria. But this might be suboptimal if the first qualifying frame is okay and a later one is even better. So, a better method is to wait until the event is over and then pick the best.
    
    The system can update the “best frame so far” in real-time. For example: *Frame 12* is the best so far (scores highest); then *Frame 13* comes in, it’s even sharper and better lit, so now that becomes the best so far.
    
    ---
    
3. **Event End Detection:** We need to know when to stop looking for a better frame and finalize our choice. In the scenario of a tossed waste item, this could be when the object (or person) is no longer detected – meaning the throw is complete and the item likely fell into the bucket (out of view). Concretely, if we had detection in previous frames but for the last M frames there is no detection (e.g., person moved away or item disappeared), then the event has ended. We can then take whatever frame was marked “best” and save it. It’s important to have a slight delay to confirm the event ended to avoid cutting off too early. Another approach is a timer: once capture mode is triggered, run it for a fixed duration (say 2 seconds) then automatically pick the best frame in that window.

---

1. **Save and Reset:** Once the best frame is selected at the end of an event, save the image with an appropriate filename (timestamp, confidence, etc. as in the current implementation). Then clear the buffer and reset the “best frame” tracker in preparation for the next detection event. The system returns to only detection mode until the next throw is detected.

---

**Real-time Considerations:** The above pipeline ensures we don’t necessarily save multiple frames for one event – only the single best frame. It avoids saving duplicates (the current code might save multiple if several frames in a row meet criteria). By buffering, we also ensure that if a slightly blurred frame met the criteria but the next frame was even clearer, we don’t miss the opportunity to choose the latter. This approach is akin to how high-end cameras do **“best shot” selection by taking a burst and then picking the sharpest photo**. Here we simulate a burst by continuously capturing but decide after seeing a few outcomes.

The pipeline can be further optimised by adjusting N (buffer size) and how frequently metrics are computed:

- If the Pi is struggling, you could compute expensive metrics (like a full BRISQUE score) only on every few frames or only on the final selection phase.
- If detection is running on one thread/core, metrics computation can potentially run in parallel on another core given Pi 5’s quad-core CPU. For instance, as soon as a frame is detected and handed off, a separate thread can compute blur, lighting, etc., so that by the time the next frame is done detecting, the previous frame’s quality metrics are ready. This pipelining maximizes throughput.

By following this pipeline, the system will effectively implement a real-time “smart frame selector” that accounts for both the **temporal aspect** (monitoring frames over the duration of an event) and **multi-metric quality assessment** to choose the optimal shot.

---

## Accuracy vs. Computational Feasibility on Raspberry Pi 5

Running advanced frame selection on an embedded device like Raspberry Pi 5 requires careful balance. The Pi 5 is significantly more powerful than its predecessors (with a 2.4 GHz quad-core Cortex-A76 CPU and a VideoCore VII GPU), but it’s still nowhere near a desktop CPU or GPU in raw performance. Here we discuss trade-offs and optimisations:

- **Object Detection Cost:** The YOLO model inference is likely the most expensive operation per frame. A smaller model (YOLOv8n or YOLOv5n – nano versions) or using lower resolution input can reduce this cost. Pi 5’s CPU can handle some neural network load, but using the GPU via OpenCV DNN or TensorRT (if available for YOLO) could accelerate it. Another option is to use the Raspberry Pi’s camera hardware features – Pi cameras can do some motion detection onboard, but in our case we need actual object recognition, so that might not apply directly. If the detection step takes, say, 50–100 ms per frame, that inherently limits the frame rate to ~10 FPS even before quality metrics. This is a **trade-off between detection accuracy and speed**: a heavier model (more accurate at detecting the waste item) will slow down frame rate, potentially missing the moment of best clarity. Thus, a slightly less accurate but faster model might actually yield a better chance of capturing a clear frame at the right time.

---

- **Quality Metrics Cost:** Simpler metrics like Laplacian variance (blur) and frame differencing (stability) are extremely fast – on the order of a millisecond or less on Pi 5. Computing a histogram and entropy for lighting is also very fast (a few ms). These will not significantly affect performance if done for each frame. More complex metrics like BRISQUE/NIQE involve computing wavelet features and a model prediction; while designed to be efficient, they will take more time (tens of ms). Running BRISQUE on every frame in a 30 FPS stream might not be feasible (30 FPS means 33 ms per frame budget). **One way to mitigate this** is to run BRISQUE only when needed: for instance, once a frame passes basic blur/stability checks, then compute BRISQUE to get a refined score. Or run BRISQUE on one frame out of every 5 during the action – this can still guide the selection without overloading the CPU. The table below (in the next section) will summarize approximate speeds.

---

- **Memory and Storage:** Buffering frames (especially at 1280×720 resolution) consumes memory. However, 5–10 frames is not a big issue (10 frames of 720p RGB ~ 10 * 1280 * 720 * 3 bytes ≈ 27.6 MB, which Pi 5 can handle with its increased RAM). Storing images to disk (eMMC or SD card) for each capture is I/O load, but the captures are infrequent (only when a throw happens) so that’s fine. If writing many images, one might have to consider I/O blocking the pipeline, but presumably this is rare.

---

- **Parallelism:** Raspberry Pi 5 has four cores – we should utilize them. For example, core 1 can handle the main loop and detection, core 2 can handle computing the quality metrics in parallel, core 3 could handle writing images or any logging, etc. Python with GIL might not fully utilize cores unless using multiprocessing or releasing GIL in C extensions (OpenCV likely releases GIL in heavy computations). Alternatively, using OpenCV’s CUDA (if you have a supported GPU or accelerator) could offload some work. But assuming CPU only: splitting tasks into separate threads (with careful synchronisation around the frame buffer) can significantly increase throughput.

---

- **Accuracy Trade-offs:** Each added metric or more strict threshold improves the chance that the selected frame is truly high quality, but it also increases the chance of *not* selecting any frame if the criteria are too strict. For instance, one might miss saving a decent frame because it didn’t meet a new metric’s threshold. We must ensure the system still captures something for each event (since missing the event entirely is worse than capturing a slightly imperfect image). Therefore, when adding metrics, consider making the selection criteria “softer” in combination: e.g., allow a frame that is a bit under the ideal lighting if it is very sharp and high confidence, and nothing better came along. It might be useful to always store at least one frame per event, even if it’s below thresholds, as a fallback (perhaps mark it as lower quality). Accuracy in terms of capturing the *optimal* frame is hard to measure without ground truth, but one can test by recording a bunch of events and seeing if the chosen frame was indeed the best. If not, adjust the metric weights/thresholds or add new ones as needed.

---

- **Real-time Constraints:** The entire pipeline should ideally run faster than the frame rate to not introduce lag. If detection is 15 FPS and metrics add negligible overhead, we’re good. If we add something like BRISQUE that takes 40 ms, suddenly the effective processing time per frame might become 100 ms +, dropping frame rate to ~10 FPS or lower. If frames are coming in faster than processing, a backlog might occur. It may be acceptable to drop frames (processing only the latest frame and skipping some) as long as we don’t skip the critical moment. A strategy here: if the system is falling behind, it could automatically become more selective (e.g., only evaluate every other frame). The Pi 5’s extra horsepower gives some headroom, but it’s wise to monitor the CPU usage. Writing efficient code (avoiding Python loops where possible, leveraging vectorized OpenCV/Numpy operations as is done) will help keep things real-time.

---

To illustrate the computational impact, consider the rough time per frame for various components on Raspberry Pi 5:

*Approximate computational cost per frame for different operations on Raspberry Pi 5 (times are illustrative). Simpler operations like computing blur or stability are negligible compared to running a neural network detection or advanced IQA metrics like BRISQUE.*

As shown above, the detection (YOLOv8s, in blue) might take on the order of tens of milliseconds per frame, dominating the cycle. The current metrics (gray bars for confidence retrieval, stability via frame diff, clarity via Laplacian, lighting via histogram) add only a millisecond or two. BRISQUE/NIQE (orange bars) are heavier but still in perhaps the 20–30 ms range each. The Pi 5 can likely handle a couple of these extra 20 ms tasks on separate cores without dropping below an acceptable frame rate, but it emphasizes why we should be strategic in their use.

In terms of **accuracy vs feasibility**: if we pursue maximum accuracy (saving only the absolute best, using very stringent quality evaluation like deep IQA), we risk the system being too slow or missing frames. On the flip side, if we make it too simplistic and fast, we might capture suboptimal frames. The recommended pipeline and metrics aim for a sweet spot: use fast, informative metrics as a first pass (confidence, blur, etc.), and optionally use one heavier metric (like BRISQUE) sparingly to refine the decision. Raspberry Pi 5, with its improved CPU, should manage this as long as concurrency is used and we avoid unnecessary computation on frames that don’t matter (which we do by gating on detection).

---

## Comparative Analysis of Quality Metrics

Finally, we present a comparison of various frame quality metrics in terms of their **speed**, **reliability** (effectiveness at indicating a clear image), and **suitability for embedded use**. This will help in deciding which metrics to use on a Raspberry Pi 5 for real-time frame selection:

| **Metric** | **Type** | **Speed (per frame)** | **Reliability for Clarity** | **Embedded Suitability** |
| --- | --- | --- | --- | --- |
| **Detection Confidence** | Object detection score (model-based) | High cost (depends on model; e.g., ~50–100 ms on CPU for YOLOv8s) | Indirectly high – if confidence is high, object is likely well-visible (though not a direct blur measure) | **Moderate** – Requires running a neural net. Needed for trigger, but costly. Use smallest model possible or hardware acceleration. |
| **Frame Stability** | Temporal difference (no-reference) | **Very fast** (~0.5–1 ms) | Medium – detects motion, hence correlates with motion blur. A stable (low-motion) frame is often clearer. | **High** – Simple calculation, trivial for Pi. Should always use as it virtually has no cost. |
| **Image Clarity (Laplacian variance)** | Spatial sharpness (no-reference) | **Very fast** (~1 ms) | High – strong indicator of focus and blur. High variance = sharp image. Needs calibrated threshold. | **High** – Implemented easily via OpenCV. Excellent for embedded devices due to low cost. |
| **Lighting Score** | Brightness & entropy (no-reference) | Fast (~1–2 ms for histogram) | Medium – ensures image is well-exposed, which is necessary for clarity but not sufficient (doesn’t detect blur). | **High** – Simple to compute. Good to include to avoid extreme lighting conditions. |
| **SSIM** | Full-reference quality | Moderate (depends on size; e.g., tens of ms for 720p) | High if reference is truly sharp – can detect even subtle degradation. But not usable without a reference frame. | **Low** – Not directly applicable for live single-frame quality. Could be used between consecutive frames, but limited usefulness and added complexity on Pi. |
| **BRISQUE** | No-reference IQA (opinion-aware) | Moderate (~20–30 ms for 720p, after model load) | High – correlates well with human perception for many distortions (blur, noise, etc.). Outputs a single quality score (lower is better quality). | **Medium** – Needs loading a model (~~200 KB). Computation is efficient, but not trivial. Likely can be used in real-time on Pi 5 for occasional frames or lower resolutions. |
| **NIQE** | No-reference IQA (opinion-unaware) | Moderate (~20–30 ms similar to BRISQUE) | Medium/High – can flag unnatural blur/noise, but might be less aligned with human opinion. (Lower score = better) | **Medium** – No training needed which simplifies usage. Computational load similar to BRISQUE. Viable on Pi 5 with careful use. |
| **PIQE** | No-reference IQA (unsupervised) | Slow (potentially 50 ms+ for full image due to block-wise analysis)[mathworks.com](https://www.mathworks.com/help/images/image-quality-metrics.html#:~:text=The%20BRISQUE%20and%20the%20NIQE,a%20subjective%20human%20quality%20score) | Medium/High – identifies local distortions and gives overall score. Similar goal as NIQE. | **Low** – Less efficient[mathworks.com](https://www.mathworks.com/help/images/image-quality-metrics.html#:~:text=The%20BRISQUE%20and%20the%20NIQE,a%20subjective%20human%20quality%20score) for real-time use on embedded hardware. Could be too slow for per-frame analysis at high FPS. |
| **Edge/Texture Count** | No-reference (heuristic) | Fast (~1–5 ms depending on method) | Medium – a sharp image of a textured object will have more edges. However, not robust to different content (a clear sky vs a textured wall). | **High** – Simple to implement (e.g., Canny edge then count). Could augment blur detection. |
| **Deep Learning IQA (e.g., NIMA)** | Learned no-reference | Very Slow (100 ms+ without acceleration) | High – can predict aesthetic/quality scores like a human would. Would capture blur, poor lighting, etc., in one score. | **Low** – Heavy computationally for Pi. Not practical without a dedicated AI accelerator. |

**Key Takeaways from the Table:** Simpler metrics (Laplacian variance, histogram/lighting, frame diff) are **highly suitable for embedded use** due to their speed and ease of implementation, and they each address a specific aspect of image quality. Advanced metrics like BRISQUE and NIQE offer a more *holistic assessment* of image quality and still maintain reasonable speed, making them candidates to improve reliability of selecting the best frame – especially on the Raspberry Pi 5 which has just enough power to use them judiciously. Full-reference metrics like SSIM are not very applicable unless a reference is available, and heavy deep learning-based metrics are currently impractical on such hardware for real-time use.

---

In conclusion, a hybrid approach that uses the current four metrics for fast filtering and adds one of the no-reference IQA metrics (BRISQUE or NIQE) for fine-grained assessment could significantly improve the chances of capturing a clear, non-blurry image of the waste item. By structuring the frame selection pipeline to operate in real-time and understanding the computational limits, the Raspberry Pi 5 can be leveraged to its fullest potential – capturing the crucial moment when the waste item is clearly visible, just before it drops into the bin. The above strategies and comparisons provide a roadmap for refining the system to achieve reliable results within the constraints of embedded hardware.