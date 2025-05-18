import os
import time
import abc
import logging
import threading
from enum import Enum
import queue
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define thread states
class ThreadState(Enum):
    INITIALIZING = 0
    READY = 1
    RUNNING = 2
    PAUSED = 3
    STOPPING = 4
    STOPPED = 5
    ERROR = 6

# Define frame quality states
class FrameQuality(Enum):
    UNKNOWN = 0
    POOR = 1
    ACCEPTABLE = 2
    GOOD = 3
    EXCELLENT = 4

# Frame data container
class FrameData:
    def __init__(self, frame: np.ndarray, timestamp: float, frame_id: int):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.metadata = {}
        self.quality_score = 0.0
        self.quality_assessment = FrameQuality.UNKNOWN
        self.detection_results = []
    
    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
    
    def set_quality(self, score: float, assessment: FrameQuality) -> None:
        """Set quality score and assessment"""
        self.quality_score = score
        self.quality_assessment = assessment
    
    def set_detection_results(self, results: List[Dict]) -> None:
        """Set detection results"""
        self.detection_results = results
    
    def get_filename(self, base_dir: str, prefix: str = "") -> str:
        """Generate a filename for saving this frame"""
        quality_str = self.quality_assessment.name.lower()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.timestamp))
        ms = int((self.timestamp - int(self.timestamp)) * 1000)
        
        filename = f"{prefix}_{timestamp_str}_{ms:03d}_{quality_str}_{self.quality_score:.2f}.jpg"
        return os.path.join(base_dir, filename)

# Abstract base class for threads
class BaseThread(abc.ABC):    
    def __init__(self, name: str, max_queue_size: int = 10):

        self.name = name
        self.logger = logging.getLogger(f"thread.{name}")
        self.state = ThreadState.INITIALIZING
        self.thread = None
        self.stop_event = threading.Event()
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queues = []
        self.stats = {
            "processed_frames": 0,
            "dropped_frames": 0,
            "processing_times": [],
            "queue_times": [],
            "start_time": None,
            "stop_time": None,
        }
    
    def set_output_queue(self, output_queue: queue.Queue) -> None:
        self.output_queues.append(output_queue)
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        try:
            self.input_queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            self.stats["dropped_frames"] += 1
            self.logger.warning("Input queue full, dropping frame")
            return False
    
    def send_to_outputs(self, item: Any) -> None:
        for q in self.output_queues:
            try:
                q.put(item, block=False)
            except queue.Full:
                self.logger.warning(f"Output queue full, dropping frame")
    
    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning("Thread already running")
            return
            
        self.stop_event.clear()
        self.state = ThreadState.READY
        self.stats["start_time"] = time.time()
        self.thread = threading.Thread(target=self._run, name=self.name)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"Thread {self.name} started")
    
    def stop(self) -> None:
        if self.thread is None or not self.thread.is_alive():
            self.logger.warning("Thread not running")
            return
            
        self.logger.info(f"Stopping thread {self.name}")
        self.state = ThreadState.STOPPING
        self.stop_event.set()
    
    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)
            if not self.thread.is_alive():
                self.state = ThreadState.STOPPED
                self.stats["stop_time"] = time.time()
                self.logger.info(f"Thread {self.name} stopped")
    
    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
            stats["max_processing_time"] = max(stats["processing_times"])
            stats["min_processing_time"] = min(stats["processing_times"])
        if stats["queue_times"]:
            stats["avg_queue_time"] = sum(stats["queue_times"]) / len(stats["queue_times"])
        
        # Calculate frames per second
        if stats["start_time"] is not None:
            if stats["stop_time"] is not None:
                elapsed = stats["stop_time"] - stats["start_time"]
            else:
                elapsed = time.time() - stats["start_time"]
                
            if elapsed > 0:
                stats["fps"] = stats["processed_frames"] / elapsed
            
        return stats
    
    @abc.abstractmethod
    def _process_item(self, item: Any) -> Optional[Any]:
        pass
    
    def _run(self) -> None:
        self.state = ThreadState.RUNNING
        self.logger.info(f"Thread {self.name} running")
        
        while not self.stop_event.is_set():
            try:
                # Get item from queue with timeout to check stop_event regularly
                try:
                    queue_start = time.time()
                    item = self.input_queue.get(timeout=0.1)
                    queue_time = time.time() - queue_start
                    self.stats["queue_times"].append(queue_time)
                except queue.Empty:
                    continue
                
                # Process the item
                try:
                    process_start = time.time()
                    result = self._process_item(item)
                    process_time = time.time() - process_start
                    
                    # Update stats
                    self.stats["processed_frames"] += 1
                    self.stats["processing_times"].append(process_time)
                    
                    # Send result to output queues if valid
                    if result is not None:
                        self.send_to_outputs(result)
                        
                except Exception as e:
                    self.logger.error(f"Error processing item: {e}", exc_info=True)
                    self.state = ThreadState.ERROR
                finally:
                    # Mark the task as done
                    self.input_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in thread loop: {e}", exc_info=True)
                self.state = ThreadState.ERROR
                
        self.state = ThreadState.STOPPED
        self.logger.info(f"Thread {self.name} exiting")

    def _send_frame(self, frame_data: Any) -> bool:
        if not self.output_queues:
            self.logger.warning("No output queues configured")
            return False
        
        sent = False
        for i, output_queue in enumerate(self.output_queues):
            try:
                # Try to send with a reasonable timeout
                output_queue.put(frame_data, block=True, timeout=0.5)
                sent = True
                self.logger.debug(f"Sent frame to output queue {i}")
            except queue.Full:
                self.logger.warning(f"Output queue {i} full, dropping frame")
                self.stats["dropped_frames"] += 1
        
        return sent

# Abstract base class for detectors
class BaseDetector(abc.ABC):
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.logger = logging.getLogger(f"detector.{self.__class__.__name__}")
        self.input_shape = None
        self.class_names = []
        self.stats = {
            "inference_times": [],
            "pre_times": [],
            "post_times": [],
        }
    
    @abc.abstractmethod
    def load_model(self) -> bool:
        pass
    
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        pass
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        stats = {}
        if self.stats["inference_times"]:
            stats["avg_inference_time"] = sum(self.stats["inference_times"]) / len(self.stats["inference_times"])
            stats["max_inference_time"] = max(self.stats["inference_times"])
            stats["min_inference_time"] = min(self.stats["inference_times"])
        if self.stats["pre_times"]:
            stats["avg_pre_time"] = sum(self.stats["pre_times"]) / len(self.stats["pre_times"])
        if self.stats["post_times"]:
            stats["avg_post_time"] = sum(self.stats["post_times"]) / len(self.stats["post_times"])
        
        return stats

# Abstract base class for frame processors
class BaseFrameProcessor(abc.ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"processor.{name}")
    
    @abc.abstractmethod
    def process(self, frame_data: FrameData) -> FrameData:
        pass

# Configuration manager function
def load_config(config_file: str) -> Dict:
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file {config_file}: {e}")
        return {}

# Utility functions
def create_directory(directory: str) -> bool:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except Exception as e:
            logging.error(f"Error creating directory {directory}: {e}")
            return False
    return True

def get_timestamp() -> float:
    return time.time()

def calculate_fps(frame_count: int, elapsed_time: float) -> float:
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0

def resize_with_aspect_ratio(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        
    return cv2.resize(image, dim)
