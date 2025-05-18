import time
import logging
import threading
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum

from base import FrameData, get_timestamp

# Set up logging
logger = logging.getLogger("camera")

class CameraStatus(Enum):
    CLOSED = 0
    INITIALIZED = 1
    STARTING = 2
    RUNNING = 3
    ERROR = 4
    STOPPING = 5

class CameraConfig:
    
    def __init__(self, 
                 width: int = 1920, 
                 height: int = 1080, 
                 fps: int = 30,
                 format: str = "RGB888",
                 rotation: int = 0,
                 camera_id: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        self.rotation = rotation
        self.camera_id = camera_id
        
    def to_dict(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.format,
            "rotation": self.rotation,
            "camera_id": self.camera_id
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CameraConfig':
        return cls(
            width=config_dict.get("width", 1920),
            height=config_dict.get("height", 1080),
            fps=config_dict.get("fps", 30),
            format=config_dict.get("format", "RGB888"),
            rotation=config_dict.get("rotation", 0),
            camera_id=config_dict.get("camera_id", 0)
        )

class Camera:
    
    def __init__(self, config: Union[CameraConfig, Dict] = None):
        # Convert dictionary to CameraConfig if needed
        if isinstance(config, dict):
            self.config = CameraConfig.from_dict(config)
        else:
            self.config = config or CameraConfig()
            
        self.status = CameraStatus.CLOSED
        self.frame_count = 0
        self.camera = None
        self.preview_config = None
        self.capture_config = None
        self.camera_controls = {}
        self.lock = threading.RLock()
        self.start_time = None
        self.stop_time = None
        self.last_frame = None
        self.last_error = None
        self.error_count = 0
        
    def initialize(self) -> bool:
        with self.lock:
            if self.status != CameraStatus.CLOSED:
                logger.warning("Camera already initialized")
                return True
                
            try:
                # Import PiCamera2 here to avoid import errors if not available
                from picamera2 import Picamera2
                
                # Get camera info
                cameras = Picamera2.global_camera_info()
                if not cameras:
                    self.last_error = "No cameras detected"
                    logger.error(self.last_error)
                    self.status = CameraStatus.ERROR
                    return False
                    
                # Check if requested camera ID is available
                if self.config.camera_id >= len(cameras):
                    self.last_error = f"Camera ID {self.config.camera_id} not available. Found {len(cameras)} cameras."
                    logger.error(self.last_error)
                    self.status = CameraStatus.ERROR
                    return False
                
                # Create camera object
                self.camera = Picamera2(self.config.camera_id)
                
                # Configure camera
                self._configure_camera()
                
                self.status = CameraStatus.INITIALIZED
                logger.info(f"Camera initialized with config: {self.config.to_dict()}")
                return True
                
            except Exception as e:
                self.last_error = f"Failed to initialize camera: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                self.status = CameraStatus.ERROR
                return False
                
    def _configure_camera(self) -> None:
        if not self.camera:
            return
            
        try:
            # Configure camera streams
            self.preview_config = self.camera.create_preview_configuration(
                main={"size": (self.config.width, self.config.height), 
                      "format": self.config.format}
            )
            
            self.capture_config = self.camera.create_still_configuration(
                main={"size": (self.config.width, self.config.height), 
                      "format": self.config.format}
            )
            
            # Set the preview configuration - this is used for video streaming
            self.camera.configure(self.preview_config)
            
            # Set camera controls if needed
            if self.camera_controls:
                self.camera.set_controls(self.camera_controls)
                
        except Exception as e:
            self.last_error = f"Failed to configure camera: {str(e)}"
            logger.error(self.last_error, exc_info=True)
            self.status = CameraStatus.ERROR
    
    def start(self) -> bool:
        with self.lock:
            if self.status == CameraStatus.RUNNING:
                logger.warning("Camera already running")
                return True
                
            if self.status == CameraStatus.CLOSED:
                if not self.initialize():
                    return False
                    
            try:
                self.status = CameraStatus.STARTING
                
                # Start the camera
                self.camera.start()
                
                # Reset counters
                self.frame_count = 0
                self.start_time = time.time()
                self.status = CameraStatus.RUNNING
                
                logger.info("Camera started")
                return True
                
            except Exception as e:
                self.last_error = f"Failed to start camera: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                self.status = CameraStatus.ERROR
                return False
    
    def stop(self) -> bool:
        with self.lock:
            if self.status not in [CameraStatus.RUNNING, CameraStatus.STARTING]:
                logger.warning(f"Camera not running (status: {self.status.name})")
                return True
                
            try:
                self.status = CameraStatus.STOPPING
                
                # Stop the camera
                self.camera.stop()
                
                self.stop_time = time.time()
                self.status = CameraStatus.INITIALIZED
                
                logger.info("Camera stopped")
                return True
                
            except Exception as e:
                self.last_error = f"Failed to stop camera: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                self.status = CameraStatus.ERROR
                return False
    
    def close(self) -> bool:
        with self.lock:
            if self.status == CameraStatus.RUNNING:
                self.stop()
                
            try:
                if self.camera:
                    self.camera.close()
                    
                self.camera = None
                self.status = CameraStatus.CLOSED
                
                logger.info("Camera closed")
                return True
                
            except Exception as e:
                self.last_error = f"Failed to close camera: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                self.status = CameraStatus.ERROR
                return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.status != CameraStatus.RUNNING:
                logger.warning(f"Cannot capture frame, camera not running (status: {self.status.name})")
                return None
                
            try:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Update counters
                self.frame_count += 1
                self.last_frame = frame
                
                return frame
                
            except Exception as e:
                self.last_error = f"Failed to capture frame: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                self.status = CameraStatus.ERROR
                return None
    
    def capture_frame_data(self) -> Optional[FrameData]:
        frame = self.capture_frame()
        if frame is not None:
            timestamp = get_timestamp()
            frame_data = FrameData(frame, timestamp, self.frame_count)
            
            # Add camera metadata
            frame_data.add_metadata("camera_config", self.config.to_dict())
            frame_data.add_metadata("frame_count", self.frame_count)
            
            return frame_data
        return None
    
    def get_fps(self) -> float:
        if self.start_time is None:
            return 0.0
            
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.frame_count > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def get_info(self) -> Dict:
        info = {
            "status": self.status.name,
            "config": self.config.to_dict(),
            "frame_count": self.frame_count,
            "fps": self.get_fps(),
            "running_time": None
        }
        
        if self.start_time is not None:
            if self.stop_time is not None:
                info["running_time"] = self.stop_time - self.start_time
            else:
                info["running_time"] = time.time() - self.start_time
                
        if self.last_error:
            info["last_error"] = self.last_error
            
        return info
    
    def set_camera_control(self, control: str, value: Any) -> bool:
        with self.lock:
            try:
                self.camera_controls[control] = value
                
                if self.camera and self.status != CameraStatus.CLOSED:
                    self.camera.set_controls({control: value})
                    
                logger.info(f"Camera control set: {control} = {value}")
                return True
                
            except Exception as e:
                self.last_error = f"Failed to set camera control {control}: {str(e)}"
                logger.error(self.last_error, exc_info=True)
                return False
    
    def get_available_cameras(self) -> List[Dict]:
        try:
            from picamera2 import Picamera2
            return Picamera2.global_camera_info()
        except Exception as e:
            logger.error(f"Failed to get available cameras: {str(e)}")
            return []
    
    def __enter__(self):
        self.initialize()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.close()


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.default_camera = None
        
    def get_camera(self, camera_id: int = 0) -> Optional[Camera]:
        return self.cameras.get(camera_id)
    
    def get_camera_list(self) -> List[int]:
        return list(self.cameras.keys())
    
    def add_camera(self, camera_id: int, config: Union[CameraConfig, Dict] = None) -> Camera:
        if camera_id in self.cameras:
            logger.warning(f"Camera {camera_id} already exists, will be replaced")
        
        # If config is a dictionary, ensure camera_id is set
        if isinstance(config, dict) and "camera_id" not in config:
            config["camera_id"] = camera_id
            
        # If config is None, create a default CameraConfig with the provided camera_id
        camera = Camera(config if config is not None else CameraConfig(camera_id=camera_id))
        self.cameras[camera_id] = camera
        
        # Set as default if it's the first camera
        if self.default_camera is None:
            self.default_camera = camera_id
            
        return camera
    
    def remove_camera(self, camera_id: int) -> bool:
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not found")
            return False
            
        camera = self.cameras[camera_id]
        camera.close()
        del self.cameras[camera_id]
        
        # Update default camera if needed
        if self.default_camera == camera_id:
            if self.cameras:
                self.default_camera = next(iter(self.cameras.keys()))
            else:
                self.default_camera = None
                
        return True
    
    def start_all(self) -> bool:
        success = True
        for camera_id, camera in self.cameras.items():
            if not camera.start():
                logger.error(f"Failed to start camera {camera_id}")
                success = False
                
        return success
    
    def stop_all(self) -> bool:
        success = True
        for camera_id, camera in self.cameras.items():
            if not camera.stop():
                logger.error(f"Failed to stop camera {camera_id}")
                success = False
                
        return success
    
    def close_all(self) -> bool:
        success = True
        for camera_id, camera in self.cameras.items():
            if not camera.close():
                logger.error(f"Failed to close camera {camera_id}")
                success = False
                
        return success
    
    def get_default_camera(self) -> Optional[Camera]:
        if self.default_camera is not None:
            return self.cameras.get(self.default_camera)
        return None
    
    def set_default_camera(self, camera_id: int) -> bool:
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not found")
            return False
            
        self.default_camera = camera_id
        return True
    
    def get_camera_info(self) -> Dict:
        info = {
            "default_camera": self.default_camera,
            "camera_count": len(self.cameras),
            "cameras": {}
        }
        
        for camera_id, camera in self.cameras.items():
            info["cameras"][camera_id] = camera.get_info()
            
        return info
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


def get_picamera2_version() -> str:
    try:
        import picamera2
        return picamera2.__version__
    except (ImportError, AttributeError):
        return "PiCamera2 not installed or version not available"


def list_cameras() -> List[Dict]:
    try:
        from picamera2 import Picamera2
        return Picamera2.global_camera_info()
    except Exception as e:
        logger.error(f"Failed to list cameras: {str(e)}")
        return []
