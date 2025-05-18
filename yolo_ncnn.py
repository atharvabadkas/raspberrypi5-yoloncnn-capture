import os
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

try:
    import ncnn
except ImportError:
    ncnn = None
    logging.warning("NCNN Python module not available, some features will be disabled")

from base import BaseDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("yolo_ncnn")

class YoloVersion(Enum):
    YOLOV8 = 8

class YoloNcnnDetector(BaseDetector):
    
    def __init__(self, 
                 param_path: str = "models/yolov8n.param",
                 bin_path: str = "models/yolov8n.bin",
                 class_names_path: str = "models/coco.names",
                 input_width: int = 640,
                 input_height: int = 640,
                 conf_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 num_threads: int = 4,
                 use_fp16: bool = True,
                 use_gpu: bool = False):
        # Pass model_path to BaseDetector
        super().__init__(model_path=param_path, conf_threshold=conf_threshold)
        
        # Model parameters
        self.param_path = param_path
        self.bin_path = bin_path
        self.class_names_path = class_names_path
        
        # Input parameters
        self.input_width = input_width
        self.input_height = input_height
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # NCNN parameters
        self.num_threads = num_threads
        self.use_fp16 = use_fp16
        self.use_gpu = use_gpu
        
        # Model objects
        self.net = None
        self.model = None  # For compatibility with other code
        self.class_names = []
        self.num_classes = 0
        self.yolo_version = YoloVersion.YOLOV8
        
        # Input/output layer names - will be detected automatically
        self.input_name = None
        self.output_name = None
        
        # Load class names
        self._load_class_names()
        
        # Statistics
        self.stats.update({
            "inference_count": 0,
            "inference_times": [],
            "avg_inference_time": 0.0,
            "detection_count": 0,
            "last_inference_time": 0.0
        })
        
    def _load_class_names(self) -> bool:
        try:
            if not os.path.exists(self.class_names_path):
                logger.warning(f"Class names file not found: {self.class_names_path}")
                # Create default class names (80 classes for COCO)
                self.class_names = [f"class{i}" for i in range(80)]
                self.num_classes = len(self.class_names)
                return True
                
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
                
            self.num_classes = len(self.class_names)
            logger.info(f"Loaded {self.num_classes} class names from {self.class_names_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            # Create default class names (80 classes for COCO)
            self.class_names = [f"class{i}" for i in range(80)]
            self.num_classes = len(self.class_names)
            return False
    
    @staticmethod
    def load_class_names(class_file: str) -> List[str]:
        try:
            if not os.path.exists(class_file):
                return [f"class{i}" for i in range(80)]
                
            with open(class_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
                
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            return [f"class{i}" for i in range(80)]
    
    def _detect_layer_names(self) -> bool:
        try:
            # Common input layer names
            input_names = ["in0", "images", "input.1", "input", "x.1", "x", "inputs"]
            
            # Common output layer names for YOLOv8
            output_names = ["out0", "output0", "output", "outputs", "370", "output.1", "448"]
            
            # Create a test extractor
            ex = self.net.create_extractor()
            
            # Try to find valid input name
            valid_input = None
            for name in input_names:
                try:
                    # Create a dummy input
                    dummy_input = np.zeros((3, self.input_height, self.input_width), dtype=np.float32)
                    ret = ex.input(name, ncnn.Mat(dummy_input))
                    if ret == 0:
                        valid_input = name
                        logger.info(f"Found valid input layer name: {name}")
                        break
                except Exception as e:
                    logger.debug(f"Input layer {name} failed: {e}")
                    continue
            
            if valid_input is None:
                # Fall back to the first layer
                try:
                    # Get blob names if available
                    blobs = self.net.input_names()
                    if blobs and len(blobs) > 0:
                        valid_input = blobs[0]
                        logger.info(f"Using first input layer: {valid_input}")
                    else:
                        # Default to 'in0'
                        valid_input = "in0"
                        logger.warning(f"Could not find valid input layer name, defaulting to {valid_input}")
                except:
                    valid_input = "in0"
                    logger.warning(f"Could not find valid input layer name, defaulting to {valid_input}")
                
            # Try to find valid output name
            valid_output = None
            for name in output_names:
                try:
                    out = ncnn.Mat()
                    ret = ex.extract(name, out)
                    if ret == 0:
                        valid_output = name
                        logger.info(f"Found valid output layer name: {name}")
                        break
                except Exception as e:
                    logger.debug(f"Output layer {name} failed: {e}")
                    continue
            
            if valid_output is None:
                # Fall back to the last layer
                try:
                    # Get blob names if available
                    blobs = self.net.output_names()
                    if blobs and len(blobs) > 0:
                        valid_output = blobs[0]
                        logger.info(f"Using first output layer: {valid_output}")
                    else:
                        # Default to 'out0'
                        valid_output = "out0"
                        logger.warning(f"Could not find valid output layer name, defaulting to {valid_output}")
                except:
                    valid_output = "out0"
                    logger.warning(f"Could not find valid output layer name, defaulting to {valid_output}")
            
            self.input_name = valid_input
            self.output_name = valid_output
            
            logger.info(f"Using input layer: {self.input_name}, output layer: {self.output_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error detecting layer names: {e}")
            # Default values
            self.input_name = "in0"
            self.output_name = "out0"
            logger.warning(f"Using default layer names: input={self.input_name}, output={self.output_name}")
            return False
    
    def load_model(self) -> bool:
        try:
            # Check if NCNN is available
            if ncnn is None:
                logger.error("NCNN Python module not available")
                return False
                
            # Check if model files exist
            if not os.path.exists(self.param_path):
                logger.error(f"Param file not found: {self.param_path}")
                return False
                
            if not os.path.exists(self.bin_path):
                logger.error(f"Bin file not found: {self.bin_path}")
                return False
                
            # Create NCNN net
            self.net = ncnn.Net()
            
            # Set NCNN options
            self.net.opt.use_vulkan_compute = self.use_gpu
            self.net.opt.use_fp16_packed = self.use_fp16
            self.net.opt.use_fp16_storage = self.use_fp16
            self.net.opt.use_fp16_arithmetic = self.use_fp16
            self.net.opt.num_threads = self.num_threads
            
            # Load model files
            logger.info(f"Loading NCNN model: {self.param_path}, {self.bin_path}")
            try:
                ret_param = self.net.load_param(self.param_path)
                if ret_param != 0:
                    logger.error(f"Failed to load param file: {self.param_path}, error code: {ret_param}")
                    return False
                    
                ret_bin = self.net.load_model(self.bin_path)
                if ret_bin != 0:
                    logger.error(f"Failed to load bin file: {self.bin_path}, error code: {ret_bin}")
                    return False
            except Exception as e:
                logger.error(f"Error loading model files: {e}")
                return False
            
            # Detect input and output layer names
            self._detect_layer_names()
            
            logger.info(f"Model loaded successfully: {self.param_path}, {self.bin_path}")
            logger.info(f"Model input size: {self.input_width}x{self.input_height}")
            logger.info(f"Using NCNN threads: {self.num_threads}, FP16: {self.use_fp16}, GPU: {self.use_gpu}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self.net is None:
            logger.error("Model not loaded")
            return []
            
        # Check if frame is valid
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.error("Invalid frame provided")
            return []
            
        try:
            # Record start time
            start_time = time.time()
            
            # Get original dimensions
            original_height, original_width = frame.shape[:2]
            
            # Preprocess the image (resize, normalize)
            input_tensor, scale_factor, pad = self._preprocess(frame)
            
            # Create NCNN extractor
            ex = self.net.create_extractor()
            
            # Set number of threads for this inference
            ex.set_num_threads(self.num_threads)
            
            # Set input tensor
            try:
                ret = ex.input(self.input_name, ncnn.Mat(input_tensor))
                if ret != 0:
                    logger.error(f"Failed to set input tensor: {ret}")
                    return []
            except Exception as e:
                logger.error(f"Exception setting input tensor: {e}")
                return []
                
            # Run inference and get output
            try:
                out = ncnn.Mat()
                ret = ex.extract(self.output_name, out)
                if ret != 0:
                    logger.error(f"Failed to extract output: {ret}")
                    return []
            except Exception as e:
                logger.error(f"Exception extracting output: {e}")
                return []
            
            # Postprocess detections
            detections = self._postprocess(out, (original_height, original_width), scale_factor, pad)
            
            # Apply NMS to filter overlapping detections
            detections = self._apply_nms(detections)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(inference_time, len(detections))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}", exc_info=True)
            return []
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Calculate scaling factor and padding
        scale = min(self.input_width / original_width, self.input_height / original_height)
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)
        
        # Calculate padding
        pad_x = (self.input_width - scaled_width) // 2
        pad_y = (self.input_height - scaled_height) // 2
        
        # Resize image more efficiently
        if scale != 1.0:
            # Use INTER_LINEAR for downscaling (faster), INTER_CUBIC for upscaling (better quality)
            interpolation = cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)
        else:
            resized = frame
        
        # Create padded image
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+scaled_height, pad_x:pad_x+scaled_width] = resized
        
        # Convert to NCNN input format (RGB, normalized, CHW)
        blob = padded.copy()
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC to CHW
        
        return blob, (scale, scale), (pad_x, pad_y)
    
    def _postprocess(self, 
                     out: "ncnn.Mat", 
                     original_shape: Tuple[int, int], 
                     scale_factor: Tuple[float, float], 
                     pad: Tuple[float, float]) -> List[Dict]:
        original_height, original_width = original_shape
        
        # Convert ncnn.Mat to numpy array
        out_shape = out.dims
        if out_shape == 2:
            # Shape: [num_dets, 5+num_classes]
            num_dets = out.h
            net_out = np.array(out).reshape(num_dets, -1)
        elif out_shape == 3:
            # Shape: [1, num_dets, 5+num_classes]
            num_dets = out.h
            net_out = np.array(out).reshape(out.c, num_dets, -1)[0]
        else:
            # Unknown shape, try to guess based on output size
            net_out = np.array(out).reshape(-1, self.num_classes + 5)
            num_dets = net_out.shape[0]
        
        # Parse detections
        detections = []
        
        for i in range(num_dets):
            detection = net_out[i]
            
            # Parse values
            x_center, y_center, width, height = detection[0:4]
            class_scores = detection[4:4+self.num_classes]
            confidence = 1.0  # Will be adjusted below
            
            # Skip low confidence detections using self.conf_threshold
            if confidence < self.conf_threshold:
                continue
                
            # Get class ID and adjusted confidence
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            # For YOLOv8, the final confidence is the class confidence
            if self.yolo_version == YoloVersion.YOLOV8:
                confidence = class_confidence
                
            # Skip low confidence detections using self.conf_threshold
            if confidence < self.conf_threshold:
                continue
                
            # Calculate bounding box coordinates in original image
            # 1. Remove padding
            x_center = (x_center - pad[0]) / scale_factor[0]
            y_center = (y_center - pad[1]) / scale_factor[1]
            width = width / scale_factor[0]
            height = height / scale_factor[1]
            
            # 2. Convert to top-left, bottom-right format
            x1 = max(0, x_center - width / 2)
            y1 = max(0, y_center - height / 2)
            x2 = min(original_width, x_center + width / 2)
            y2 = min(original_height, y_center + height / 2)
            
            # 3. Handle boundaries
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))
            
            # Skip if box has zero area
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class{class_id}"
            
            # Add detection
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_name
            })
        
        # Apply non-maximum suppression
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        # Group detections by class
        detections_by_class = {}
        for det in detections:
            class_id = det["class_id"]
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append(det)
            
        # Apply NMS to each class
        filtered_detections = []
        for class_id, dets in detections_by_class.items():
            # Sort by confidence (highest first)
            dets.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Apply NMS
            keep = []
            while len(dets) > 0:
                # Keep the detection with highest confidence
                keep.append(dets[0])
                
                # Remove detections with high IoU
                remaining = []
                for det in dets[1:]:
                    if self._calculate_iou(dets[0]["bbox"], det["bbox"]) < self.nms_threshold:
                        remaining.append(det)
                dets = remaining
                
            # Add kept detections to result
            filtered_detections.extend(keep)
            
        return filtered_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        # Convert to [x1, y1, x2, y2] format if needed
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area <= 0:
            return 0.0
        iou = intersection_area / union_area
        
        return iou
    
    def _update_stats(self, inference_time: float, num_detections: int) -> None:
        self.stats["inference_count"] += 1
        self.stats["last_inference_time"] = inference_time
        self.stats["detection_count"] += num_detections
        
        # Keep track of inference times (last 100)
        self.stats["inference_times"].append(inference_time)
        if len(self.stats["inference_times"]) > 100:
            self.stats["inference_times"] = self.stats["inference_times"][-100:]
            
        # Calculate average inference time
        self.stats["avg_inference_time"] = sum(self.stats["inference_times"]) / len(self.stats["inference_times"])
    
    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        
        # Add model info
        stats.update({
            "model_path": self.param_path,
            "model_type": self.yolo_version.name if self.yolo_version else "Unknown",
            "input_shape": f"{self.input_width}x{self.input_height}",
            "num_classes": self.num_classes,
            "num_threads": self.num_threads,
            "use_fp16": self.use_fp16,
            "use_gpu": self.use_gpu and getattr(self.net, "opt", None) and getattr(self.net.opt, "use_vulkan_compute", False)
        })
        
        # Add inference FPS
        if stats["inference_count"] > 0 and stats["avg_inference_time"] > 0:
            stats["inference_fps"] = 1.0 / stats["avg_inference_time"]
        else:
            stats["inference_fps"] = 0.0
            
        return stats
    
    def set_parameters(self, 
                       conf_threshold: Optional[float] = None,
                       nms_threshold: Optional[float] = None,
                       num_threads: Optional[int] = None,
                       use_fp16: Optional[bool] = None,
                       use_gpu: Optional[bool] = None) -> None:
        # Update parameters if provided
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
            
        if num_threads is not None:
            self.num_threads = num_threads
            if self.net:
                self.net.opt.num_threads = num_threads
                
        if use_fp16 is not None:
            self.use_fp16 = use_fp16
            if self.net:
                self.net.opt.use_fp16_arithmetic = use_fp16
                
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if self.net and ncnn and ncnn.get_gpu_count() > 0:
                self.net.opt.use_vulkan_compute = use_gpu
                
        logger.info(f"Updated detector parameters: conf={self.conf_threshold}, "
                   f"nms={self.nms_threshold}, threads={self.num_threads}, "
                   f"fp16={self.use_fp16}, gpu={self.use_gpu}")
                   
    def is_model_loaded(self) -> bool:
        return self.net is not None


# Test function to run detection on an image
def test_detector(image_path: str = None):
    # Create a test image if path not provided
    if image_path is None or not os.path.exists(image_path):
        # Create a test pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a rectangle and circle
        cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.circle(img, (450, 250), 100, (0, 0, 255), -1)
        
        # Add text
        cv2.putText(img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Load image from file
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image from {image_path}")
            return
    
    # Create detector
    model_found = False
    detector = None
    
    # Check for YOLOv8n model
    param_path = "models/yolov8n.param"
    bin_path = "models/yolov8n.bin"
    
    if os.path.exists(param_path) and os.path.exists(bin_path):
        print(f"Using model: {param_path} and {bin_path}")
        detector = YoloNcnnDetector(param_path=param_path, bin_path=bin_path)
        model_found = True
    
    if not model_found:
        print("YOLOv8n model files not found. Please run download_model.py and convert_model.py first.")
        return
    
    # Load model
    print("Loading model...")
    if not detector.load_model():
        print("Error loading model")
        return
    
    # Run detection
    print("Running detection...")
    start_time = time.time()
    detections = detector.detect(img)
    inference_time = time.time() - start_time
    
    print(f"Detection completed in {inference_time:.3f} seconds")
    print(f"Found {len(detections)} objects")
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
        conf = det["confidence"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        
        # Generate a color based on class_id
        color_id = class_id % 20  # Cycle through 20 different colors
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
            (64, 0, 64), (0, 64, 64), (192, 192, 192), (128, 128, 128)
        ]
        color = colors[color_id]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1])
        
        # Draw label background
        cv2.rectangle(
            img, 
            (x1, y1_label - label_size[1] - baseline), 
            (x1 + label_size[0], y1_label),
            color, 
            cv2.FILLED
        )
        
        # Draw text
        cv2.putText(
            img,
            label,
            (x1, y1_label - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            1
        )
    
    # Add inference time
    cv2.putText(
        img,
        f"Inference: {inference_time*1000:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    
    # Save result
    result_path = "test_detection.jpg"
    cv2.imwrite(result_path, img)
    print(f"Result saved to {result_path}")
    
    # Print statistics
    stats = detector.get_stats()
    print("Detector statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run test if script is executed directly
    import sys
    
    # Get image path from command line if provided
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run test
    test_detector(image_path)
