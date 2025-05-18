import time
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Any

from base import BaseThread, FrameData, ThreadState
from camera import Camera, CameraStatus, CameraConfig

class CaptureThread(BaseThread):
    def __init__(self, camera: Camera, max_queue_size: int = 10, frame_rate_limit: float = 0):
        super().__init__("CaptureThread", max_queue_size)
        self.camera = camera
        self.frame_rate_limit = frame_rate_limit
        self.min_frame_time = 1.0 / frame_rate_limit if frame_rate_limit > 0 else 0
        self.latest_frame = None
        self.latest_frame_time = 0
        self.dropped_frames_counter = 0
        self.frame_lock = threading.Lock()
        
        # Additional statistics
        self.stats.update({
            "captured_frames": 0,
            "camera_errors": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "camera_fps": 0.0,
        })
        
    def _process_item(self, item: Any) -> Optional[Any]:
        # This thread is primarily a producer, not a consumer
        # In a typical setup, nothing should be put in its input queue
        self.logger.warning("CaptureThread received an item in its input queue (unexpected)")
        return None
    
    def _run(self) -> None:
        self.state = ThreadState.RUNNING
        self.logger.info(f"Thread {self.name} running")
        
        # Ensure camera is initialized and running
        if not self._ensure_camera_ready():
            self.state = ThreadState.ERROR
            self.logger.error("Failed to initialize camera, thread stopping")
            return
        
        last_frame_time = time.time()
        
        # Main capture loop
        while not self.stop_event.is_set():
            try:
                # Rate limiting
                if self.min_frame_time > 0:
                    elapsed = time.time() - last_frame_time
                    if elapsed < self.min_frame_time:
                        # Sleep for the remaining time to match the target frame rate
                        time.sleep(self.min_frame_time - elapsed)
                
                # Capture start time for performance measurement
                process_start = time.time()
                
                # Check if camera is still running
                if self.camera.status != CameraStatus.RUNNING:
                    self.stats["camera_errors"] += 1
                    self.logger.warning("Camera not running, attempting recovery")
                    
                    if not self._recover_camera():
                        # If recovery failed, sleep briefly to avoid busy loop
                        time.sleep(0.5)
                        continue
                
                # Capture frame using camera
                frame_data = self.camera.capture_frame_data()
                
                if frame_data is None:
                    self.stats["camera_errors"] += 1
                    self.logger.warning("Failed to capture frame, camera returned None")
                    continue
                
                # Update statistics
                self.stats["captured_frames"] += 1
                self.stats["camera_fps"] = self.camera.get_fps()
                process_time = time.time() - process_start
                self.stats["processing_times"].append(process_time)
                
                # Store latest frame
                with self.frame_lock:
                    self.latest_frame = frame_data
                    self.latest_frame_time = time.time()
                
                # Send frame to output queues
                for q in self.output_queues:
                    try:
                        # Non-blocking put - if queue is full, drop the frame
                        if q.full():
                            self.dropped_frames_counter += 1
                            self.stats["dropped_frames"] += 1
                            # Log less frequently to avoid log spam
                            if self.dropped_frames_counter % 30 == 1:
                                self.logger.warning(f"Output queue full, dropping frames ({self.dropped_frames_counter} dropped)")
                        else:
                            q.put(frame_data, block=False)
                            self.stats["processed_frames"] += 1
                            # Reset counter when successfully sending a frame
                            if self.dropped_frames_counter > 0:
                                self.logger.info(f"Queue no longer full, resumed sending frames after {self.dropped_frames_counter} dropped frames")
                                self.dropped_frames_counter = 0
                    except Exception as e:
                        self.logger.error(f"Error sending frame to queue: {e}")
                
                # Update last frame time for rate limiting
                last_frame_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}", exc_info=True)
                self.state = ThreadState.ERROR
                
                # Try to recover if possible
                if not self._recover_camera():
                    # If recovery fails consistently, exit the thread
                    consecutive_errors = self.stats.get("consecutive_errors", 0) + 1
                    self.stats["consecutive_errors"] = consecutive_errors
                    
                    if consecutive_errors > 10:
                        self.logger.error("Too many consecutive errors, stopping thread")
                        break
                        
                    # Sleep briefly to avoid busy loop
                    time.sleep(0.5)
                else:
                    # Reset consecutive errors counter
                    self.stats["consecutive_errors"] = 0
        
        # Clean up camera when thread stops
        self._cleanup_camera()
        self.state = ThreadState.STOPPED
        self.logger.info(f"Thread {self.name} exiting")
    
    def _ensure_camera_ready(self) -> bool:
        try:
            if self.camera.status == CameraStatus.CLOSED:
                self.logger.info("Initializing camera")
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize camera")
                    return False
            
            if self.camera.status == CameraStatus.INITIALIZED:
                self.logger.info("Starting camera")
                if not self.camera.start():
                    self.logger.error("Failed to start camera")
                    return False
                    
            return self.camera.status == CameraStatus.RUNNING
            
        except Exception as e:
            self.logger.error(f"Error ensuring camera is ready: {e}", exc_info=True)
            return False
    
    def _recover_camera(self) -> bool:
        self.stats["recovery_attempts"] += 1
        
        try:
            self.logger.info("Attempting to recover camera")
            
            # Try to stop and close the camera first
            try:
                if self.camera.status != CameraStatus.CLOSED:
                    self.camera.stop()
                    self.camera.close()
            except Exception as e:
                self.logger.warning(f"Error during camera cleanup: {e}")
            
            # Reinitialize and start the camera
            if self.camera.initialize() and self.camera.start():
                self.logger.info("Camera recovery successful")
                self.stats["successful_recoveries"] += 1
                return True
            else:
                self.logger.error("Camera recovery failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during camera recovery: {e}", exc_info=True)
            return False
    
    def _cleanup_camera(self) -> None:
        try:
            if self.camera.status == CameraStatus.RUNNING:
                self.logger.info("Stopping camera")
                self.camera.stop()
                
            self.logger.info("Closing camera")
            self.camera.close()
            
        except Exception as e:
            self.logger.error(f"Error during camera cleanup: {e}", exc_info=True)
    
    def get_latest_frame(self) -> Optional[FrameData]:
        with self.frame_lock:
            return self.latest_frame
    
    def get_stats(self) -> Dict:
        stats = super().get_stats()
        
        # Add camera-specific stats
        if self.camera:
            camera_info = self.camera.get_info()
            stats.update({
                "camera_status": camera_info.get("status", "Unknown"),
                "camera_frame_count": camera_info.get("frame_count", 0),
                "camera_fps": camera_info.get("fps", 0.0),
            })
            
        stats["frame_rate_limit"] = self.frame_rate_limit
        stats["latest_frame_age"] = time.time() - self.latest_frame_time if self.latest_frame_time > 0 else None
        
        return stats

    def set_frame_rate(self, frame_rate: float) -> None:
        if frame_rate < 0:
            frame_rate = 0
            
        old_rate = self.frame_rate_limit
        self.frame_rate_limit = frame_rate
        self.min_frame_time = 1.0 / frame_rate if frame_rate > 0 else 0
        
        self.logger.info(f"Frame rate adjusted from {old_rate:.1f} to {frame_rate:.1f} FPS")
    
    def get_frame_rate(self) -> float:
        return self.frame_rate_limit