#!/usr/bin/env python3
"""
Thread Manager for WMSV4AI
Coordinates thread lifecycle and provides monitoring
"""

import os
import time
import logging
import threading
import queue
import signal
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from base import BaseThread, ThreadState, FrameData, FrameQuality
from capture_thread import CaptureThread
from inference_thread import InferenceThread, DetectionParameters
from quality_thread import QualityThread, QualityParameters
from save_thread import SaveThread, SaveParameters
from camera import Camera, CameraManager, CameraStatus
from yolo_ncnn import YoloNcnnDetector
from image_quality import ImageQualityProcessor
from performance_metrics import get_metrics_instance
from image_storage import get_storage_instance

class SystemState(Enum):
    """System state enumeration"""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    PAUSED = 5

class ThreadManager:
    """Manages and coordinates all threads in the system"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize thread manager
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("thread_manager")
        self.config = config or {}
        self.state = SystemState.STOPPED
        self.threads = {}
        self.queues = {}
        self.camera_manager = None
        self.camera = None
        self.detector = None
        self.quality_processor = None
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.error_lock = threading.Lock()
        self.thread_errors = {}
        self.components_initialized = False
        
        # Statistics tracking
        self.stats = {
            "start_time": None,
            "stop_time": None,
            "uptime": 0,
            "thread_stats": {},
            "queue_stats": {},
            "error_count": 0,
            "restart_count": 0,
            "last_error": None,
        }
        
    def initialize_components(self) -> bool:
        """
        Initialize system components (camera, detector, etc.)
        
        Returns:
            True if initialization was successful
        """
        try:
            # Initialize camera
            self.logger.info("Initializing camera...")
            self._init_camera()
            
            # Initialize detector
            self.logger.info("Initializing detector...")
            self._init_detector()
            
            # Initialize quality processor
            self.logger.info("Initializing quality processor...")
            self._init_quality_processor()
            
            # Create queues
            self.logger.info("Creating queues...")
            self._create_queues()
            
            # Create threads
            self.logger.info("Creating threads...")
            self._create_threads()
            
            self.components_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}", exc_info=True)
            self.state = SystemState.ERROR
            self.stats["last_error"] = str(e)
            self.stats["error_count"] += 1
            return False
    
    def _init_camera(self) -> None:
        """Initialize camera and camera manager"""
        # Get camera configuration
        camera_config = self.config.get("camera", {})
        
        # Create camera manager
        self.camera_manager = CameraManager()
        
        # Add and get camera - pass the camera_config dictionary directly
        # The Camera class now handles dictionary conversion internally
        camera_id = camera_config.get("camera_id", 0)
        self.camera = self.camera_manager.add_camera(camera_id, camera_config)
        
        # Test camera
        if not self.camera.initialize():
            raise RuntimeError(f"Failed to initialize camera: {self.camera.last_error}")
    
    def _init_detector(self) -> None:
        """Initialize object detector"""
        # Get detector configuration
        detector_config = self.config.get("detector", {})
        
        # Create detector
        param_path = detector_config.get("model_param_path", "models/yolov8n.param")
        bin_path = detector_config.get("model_bin_path", "models/yolov8n.bin")
        class_file = detector_config.get("class_file", "models/coco.names")
        conf_threshold = detector_config.get("conf_threshold", 0.25)
        nms_threshold = detector_config.get("nms_threshold", 0.45)
        input_width = detector_config.get("input_width", 640)
        input_height = detector_config.get("input_height", 640)
        num_threads = detector_config.get("num_threads", 4)
        use_fp16 = detector_config.get("use_fp16", True)
        
        self.detector = YoloNcnnDetector(
            param_path=param_path,
            bin_path=bin_path,
            class_names_path=class_file,
            input_width=input_width,
            input_height=input_height,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            num_threads=num_threads,
            use_fp16=use_fp16
        )
        
        # Load model
        if not self.detector.load_model():
            raise RuntimeError("Failed to load detector model")
    
    def _init_quality_processor(self) -> None:
        """Initialize quality processor"""
        # Get quality configuration
        quality_config = self.config.get("image_quality", {})
        
        # Create quality processor
        self.quality_processor = ImageQualityProcessor(quality_config)
    
    def _create_queues(self) -> None:
        """Create communication queues between threads"""
        # Get queue sizes from config
        queue_config = self.config.get("system", {})
        base_queue_size = queue_config.get("max_queue_size", 20)
        
        # Create queues with specific sizes for different stages
        # Detection frames need a much larger queue as they're the current bottleneck
        # in the system causing "Output queue full" errors
        self.queues = {
            "raw_frames": queue.Queue(maxsize=base_queue_size),
            "detected_frames": queue.Queue(maxsize=base_queue_size * 4),  # Increased from 2x to 4x
            "selected_frames": queue.Queue(maxsize=base_queue_size * 4)   # Keep this large for saving
        }
    
    def _create_threads(self) -> None:
        """Create thread objects"""
        # Create Capture Thread
        capture_thread = CaptureThread(
            camera=self.camera,
            max_queue_size=self.config.get("system", {}).get("max_queue_size", 10),
            frame_rate_limit=self.config.get("capture", {}).get("frame_rate_limit", 0)
        )
        capture_thread.set_output_queue(self.queues["raw_frames"])
        self.threads["capture"] = capture_thread
        
        # Create Inference Thread
        detection_params = DetectionParameters(
            main_conf_threshold=self.config.get("detector", {}).get("conf_threshold", 0.25),
            secondary_conf_threshold=self.config.get("detector", {}).get("secondary_conf_threshold", 0.5),
            target_classes=self.config.get("detector", {}).get("target_classes"),
            max_detections=self.config.get("detector", {}).get("max_detections", 20),
            min_detection_area=self.config.get("detector", {}).get("min_detection_area", 100)
        )
        
        inference_thread = InferenceThread(
            detector=self.detector,
            detection_params=detection_params,
            max_queue_size=self.config.get("system", {}).get("max_queue_size", 10),
            enable_fallback=self.config.get("detector", {}).get("enable_fallback", True)
        )
        inference_thread.input_queue = self.queues["raw_frames"]
        inference_thread.set_output_queue(self.queues["detected_frames"])
        self.threads["inference"] = inference_thread
        
        # Create Quality Thread
        quality_params = QualityParameters(
            buffer_size=self.config.get("frame_selector", {}).get("buffer_size", 5),
            min_quality_score=self.config.get("frame_selector", {}).get("min_quality_score", 0.6),
            min_detection_score=self.config.get("frame_selector", {}).get("min_detection_score", 0.5),
            quality_weight=self.config.get("frame_selector", {}).get("quality_weight", 0.7),
            detection_weight=self.config.get("frame_selector", {}).get("detection_weight", 0.3),
            selection_interval=self.config.get("frame_selector", {}).get("selection_interval", 1.0),
            selection_cooldown=self.config.get("frame_selector", {}).get("selection_cooldown", 0.5),
            auto_select_best=self.config.get("frame_selector", {}).get("auto_select_best", True)
        )
        
        quality_thread = QualityThread(
            quality_processor=self.quality_processor,
            params=quality_params,
            max_queue_size=self.config.get("system", {}).get("max_queue_size", 10)
        )
        quality_thread.input_queue = self.queues["detected_frames"]
        quality_thread.set_output_queue(self.queues["selected_frames"])
        self.threads["quality"] = quality_thread
        
        # Get image storage instance
        storage = get_storage_instance(self.config)
        
        # Create Save Thread
        save_params = SaveParameters(
            base_dir=self.config.get("system", {}).get("save_dir", "images"),
            use_date_subdirs=self.config.get("image_storage", {}).get("storage_mode", "by_date") == "by_date",
            use_class_subdirs=self.config.get("image_storage", {}).get("storage_mode", "by_date") == "by_class",
            include_metadata=self.config.get("image_storage", {}).get("save_metadata", True),
            save_quality=self.config.get("image_storage", {}).get("jpeg_quality", 95),
            draw_detections=self.config.get("save", {}).get("draw_detections", True),
            save_original=self.config.get("save", {}).get("save_original", True),
            save_metadata_file=self.config.get("image_storage", {}).get("save_metadata", True),
            filename_template=self.config.get("save", {}).get("filename_template", 
                                              "{timestamp}_{frame_id}_{quality:.2f}_{class_name}"),
            storage_instance=storage
        )
        
        save_thread = SaveThread(
            params=save_params,
            max_queue_size=self.config.get("system", {}).get("max_queue_size", 20) * 2  # Larger queue for saving
        )
        save_thread.input_queue = self.queues["selected_frames"]
        self.threads["save"] = save_thread
    
    def start(self) -> bool:
        """
        Start all threads
        
        Returns:
            True if all threads started successfully
        """
        if not self.components_initialized:
            self.logger.error("Components not initialized")
            return False
        
        if self.state == SystemState.RUNNING:
            self.logger.warning("System already running")
            return True
        
        try:
            self.logger.info("Starting system...")
            self.state = SystemState.STARTING
            self.stats["start_time"] = time.time()
            self.stop_event.clear()
            
            # Start threads in reverse dependency order
            
            # Start save thread first
            self.logger.info("Starting save thread")
            self.threads["save"].start()
            time.sleep(0.1)  # Small delay to ensure startup
            
            # Start quality thread
            self.logger.info("Starting quality thread")
            self.threads["quality"].start() 
            time.sleep(0.1)  # Small delay to ensure startup
            
            # Start inference thread
            self.logger.info("Starting inference thread")
            self.threads["inference"].start()
            time.sleep(0.1)  # Small delay to ensure startup
            
            # Start capture thread (which starts the camera) last
            self.logger.info("Starting capture thread")
            self.threads["capture"].start()
            time.sleep(0.1)  # Small delay to ensure startup
            
            # Start monitoring thread
            self._start_monitor_thread()
            
            # Verify that all threads are running
            for thread_name, thread in self.threads.items():
                if thread.state != ThreadState.RUNNING:
                    self.logger.error(f"Thread {thread_name} failed to start. State: {thread.state.name}")
                    self.state = SystemState.ERROR
                    return False
            
            self.state = SystemState.RUNNING
            self.logger.info("System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}", exc_info=True)
            self.state = SystemState.ERROR
            self.stats["last_error"] = str(e)
            self.stats["error_count"] += 1
            return False
    
    def stop(self) -> bool:
        """
        Stop all threads
        
        Returns:
            True if all threads were stopped successfully
        """
        if self.state not in [SystemState.RUNNING, SystemState.ERROR, SystemState.PAUSED]:
            self.logger.warning(f"System not running (state: {self.state.name})")
            return True
            
        self.logger.info("Stopping system...")
        self.state = SystemState.STOPPING
        self.stop_event.set()
        
        # Stop monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.info("Stopping monitor thread")
            self.monitor_thread.join(timeout=2.0)
            
        # Stop threads in order (capture -> inference -> quality -> save)
        threads_to_stop = ["capture", "inference", "quality", "save"]
        
        for thread_name in threads_to_stop:
            thread = self.threads.get(thread_name)
            if thread:
                self.logger.info(f"Stopping {thread_name} thread")
                thread.stop()
        
        # Join threads to wait for them to stop
        for thread_name in threads_to_stop:
            thread = self.threads.get(thread_name)
            if thread:
                self.logger.info(f"Joining {thread_name} thread")
                thread.join(timeout=5.0)
                
                # Check if thread stopped successfully
                if thread.thread and thread.thread.is_alive():
                    self.logger.warning(f"{thread_name} thread did not stop within timeout")
        
        # Close camera
        if self.camera:
            self.logger.info("Closing camera")
            self.camera.close()
            
        self.state = SystemState.STOPPED
        self.stats["stop_time"] = time.time()
        if self.stats["start_time"]:
            self.stats["uptime"] += self.stats["stop_time"] - self.stats["start_time"]
            
        self.logger.info("System stopped successfully")
        return True
    
    def pause(self) -> bool:
        """
        Pause capture thread
        
        Returns:
            True if capture thread was paused successfully
        """
        if self.state != SystemState.RUNNING:
            self.logger.warning(f"System not running (state: {self.state.name})")
            return False
            
        # Only pause the capture thread
        capture_thread = self.threads.get("capture")
        if capture_thread:
            self.logger.info("Pausing capture thread")
            # We don't have a pause method, so stop the capture thread
            capture_thread.stop()
            capture_thread.join(timeout=2.0)
            
        self.state = SystemState.PAUSED
        self.logger.info("System paused")
        return True
    
    def resume(self) -> bool:
        """
        Resume capture thread
        
        Returns:
            True if capture thread was resumed successfully
        """
        if self.state != SystemState.PAUSED:
            self.logger.warning(f"System not paused (state: {self.state.name})")
            return False
            
        # Restart the capture thread
        capture_thread = self.threads.get("capture")
        if capture_thread:
            self.logger.info("Resuming capture thread")
            # Create a new capture thread
            capture_thread = CaptureThread(
                camera=self.camera,
                max_queue_size=self.config.get("system", {}).get("max_queue_size", 10),
                frame_rate_limit=self.config.get("capture", {}).get("frame_rate_limit", 0)
            )
            capture_thread.set_output_queue(self.queues["raw_frames"])
            self.threads["capture"] = capture_thread
            capture_thread.start()
            
        self.state = SystemState.RUNNING
        self.logger.info("System resumed")
        return True
    
    def _stop_all_threads(self) -> None:
        """Stop all threads"""
        for thread_name, thread in self.threads.items():
            if thread.state != ThreadState.STOPPED:
                self.logger.info(f"Stopping {thread_name} thread")
                thread.stop()
                thread.join(timeout=2.0)
    
    def _start_monitor_thread(self) -> None:
        """Start thread monitoring loop"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Monitor thread already running")
            return
            
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ThreadMonitor"
        )
        self.monitor_thread.start()
    
    def _monitor_loop(self) -> None:
        """Monitoring thread loop"""
        self.logger.info("Monitor thread started")
        last_stats_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Check thread health
                self._check_thread_health()
                
                # Update statistics
                self._update_stats()
                
                # Monitor queue sizes and throttle if needed
                self._monitor_queues()
                
                # Log periodic statistics
                current_time = time.time()
                if current_time - last_stats_time >= 10.0:  # Log every 10 seconds
                    self._log_system_stats()
                    last_stats_time = current_time
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(1.0)  # Sleep longer on error
                
        self.logger.info("Monitor thread stopping")
    
    def _monitor_queues(self) -> None:
        """Monitor queue sizes and apply throttling if needed"""
        if not self.queues:
            return
            
        try:
            # Check queue utilization
            raw_queue_size = self.queues["raw_frames"].qsize()
            raw_queue_capacity = self.queues["raw_frames"].maxsize
            
            detection_queue_size = self.queues["detected_frames"].qsize()
            detection_queue_capacity = self.queues["detected_frames"].maxsize
            
            selected_queue_size = self.queues["selected_frames"].qsize()
            selected_queue_capacity = self.queues["selected_frames"].maxsize
            
            # Calculate utilization percentages
            raw_util = (raw_queue_size / raw_queue_capacity) * 100 if raw_queue_capacity > 0 else 0
            detection_util = (detection_queue_size / detection_queue_capacity) * 100 if detection_queue_capacity > 0 else 0
            selected_util = (selected_queue_size / selected_queue_capacity) * 100 if selected_queue_capacity > 0 else 0
            
            # Update stats
            self.stats["queue_utilization"] = {
                "raw_frames": raw_util,
                "detected_frames": detection_util,
                "selected_frames": selected_util
            }
            
            # Check if any queue is getting too full
            critical_threshold = 90.0  # 90% utilization
            high_threshold = 75.0  # 75% utilization
            
            # Clear blockages - if detected queue has frames but selected queue is empty for too long
            starvation_detected = detection_queue_size > 5 and selected_queue_size == 0
            elapsed_since_last_save = time.time() - self.threads["save"].last_save_time if hasattr(self.threads["save"], "last_save_time") else 9999
            starvation_time = 5.0  # 5 seconds with no images saved
            
            # If starvation is detected, trigger frame selection or try to unblock the system
            if starvation_detected and elapsed_since_last_save > starvation_time:
                self.logger.warning(f"🚨 Detected pipeline blockage: detection_queue={detection_queue_size}, selected_queue={selected_queue_size}, time since last save={elapsed_since_last_save:.1f}s")
                
                # Try to trigger selection in the quality thread
                if self.threads["quality"].state == ThreadState.RUNNING:
                    self.logger.info("Manually triggering frame selection to unblock pipeline")
                    result = self.threads["quality"].trigger_selection()
                    if result:
                        self.logger.info("Manual frame selection triggered successfully")
                    else:
                        self.logger.warning("Failed to trigger manual frame selection")
                        
                # If still blocked, clear some frames from the detection queue
                if detection_util > 80.0 and elapsed_since_last_save > 10.0:
                    self.logger.warning(f"Emergency clearing of detection queue (utilization: {detection_util:.1f}%)")
                    try:
                        # Clear half of the frames in the detection queue
                        frames_to_clear = detection_queue_size // 2
                        for _ in range(frames_to_clear):
                            if not self.queues["detected_frames"].empty():
                                _ = self.queues["detected_frames"].get_nowait()
                        self.logger.info(f"Cleared {frames_to_clear} frames from detection queue")
                    except Exception as e:
                        self.logger.error(f"Error clearing detection queue: {e}")
            
            # Apply throttling based on queue utilization
            if raw_util > critical_threshold:
                # Critical throttling - capture thread
                if hasattr(self.threads["capture"], "set_frame_rate"):
                    current_rate = self.threads["capture"].get_frame_rate()
                    if current_rate > 2.0:  # Don't go below 2 FPS
                        new_rate = max(2.0, current_rate * 0.7)  # Reduce by 30%
                        self.threads["capture"].set_frame_rate(new_rate)
                        self.logger.warning(f"Critical throttling: Reduced frame rate to {new_rate:.1f} FPS (raw queue: {raw_util:.1f}%)")
                        
            elif raw_util > high_threshold:
                # High throttling - capture thread
                if hasattr(self.threads["capture"], "set_frame_rate"):
                    current_rate = self.threads["capture"].get_frame_rate()
                    if current_rate > 5.0:  # Don't go below 5 FPS
                        new_rate = max(5.0, current_rate * 0.85)  # Reduce by 15%
                        self.threads["capture"].set_frame_rate(new_rate)
                        self.logger.info(f"High throttling: Reduced frame rate to {new_rate:.1f} FPS (raw queue: {raw_util:.1f}%)")
            
            # If queues are mostly empty, we can increase frame rate
            elif raw_util < 20.0 and detection_util < 30.0:
                if hasattr(self.threads["capture"], "set_frame_rate"):
                    current_rate = self.threads["capture"].get_frame_rate()
                    config_fps = self.config.get("camera", {}).get("fps", 30)
                    # Increase rate if below config, but not too aggressively
                    if current_rate < config_fps:
                        new_rate = min(config_fps, current_rate * 1.1)  # Increase by 10%
                        self.threads["capture"].set_frame_rate(new_rate)
                        self.logger.info(f"Increasing frame rate to {new_rate:.1f} FPS (raw queue: {raw_util:.1f}%)")
            
            # Log queue status periodically based on utilization
            if any(util > high_threshold for util in [raw_util, detection_util, selected_util]):
                self.logger.warning(f"Queue utilization: raw_frames: {raw_queue_size}/{raw_queue_capacity} ({raw_util:.1f}%), "
                              f"detected_frames: {detection_queue_size}/{detection_queue_capacity} ({detection_util:.1f}%), "
                              f"selected_frames: {selected_queue_size}/{selected_queue_capacity} ({selected_util:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"Error monitoring queues: {e}", exc_info=True)
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
    
    def _log_system_stats(self) -> None:
        """Log system statistics"""
        try:
            # Get queue stats
            queue_stats = {}
            for name, q in self.queues.items():
                try:
                    size = q.qsize()
                    capacity = q.maxsize
                    utilization = size / capacity if capacity > 0 else 0
                    queue_stats[name] = {
                        "size": size,
                        "capacity": capacity,
                        "utilization": utilization
                    }
                except Exception:
                    pass
            
            # Log queue utilization
            self.logger.info("Queue utilization: " + 
                          ", ".join([f"{name}: {stats['size']}/{stats['capacity']} ({stats['utilization']:.1%})" 
                                    for name, stats in queue_stats.items()]))
            
            # Log thread stats
            thread_stats = {}
            total_processed = 0
            total_dropped = 0
            
            for name, thread in self.threads.items():
                stats = thread.get_stats()
                thread_stats[name] = {
                    "processed": stats.get("processed_frames", 0),
                    "dropped": stats.get("dropped_frames", 0),
                    "state": thread.state.name if hasattr(thread, "state") else "UNKNOWN"
                }
                total_processed += stats.get("processed_frames", 0)
                total_dropped += stats.get("dropped_frames", 0)
            
            # Log thread stats summary
            self.logger.info(f"System throughput: {total_processed} frames processed, {total_dropped} dropped")
            
            # Get performance metrics if available
            try:
                metrics = get_metrics_instance()
                system_metrics = metrics.get_latest_metrics()
                
                if system_metrics:
                    self.logger.info(f"System metrics: CPU: {system_metrics.get('cpu_percent', 0):.1f}%, " +
                                  f"Memory: {system_metrics.get('memory_percent', 0):.1f}%, " +
                                  f"Temperature: {system_metrics.get('temperature', 0):.1f}°C")
            except Exception:
                pass
                
        except Exception as e:
            self.logger.error(f"Error logging system stats: {e}")
    
    def _check_thread_health(self) -> None:
        """Check health of all threads"""
        for thread_name, thread in self.threads.items():
            if thread.state == ThreadState.ERROR:
                with self.error_lock:
                    self.thread_errors[thread_name] = f"Thread {thread_name} in error state"
                    self.stats["error_count"] += 1
                    self.stats["last_error"] = f"Thread {thread_name} in error state"
                
                self.logger.error(f"Thread {thread_name} in error state")
                
                # Attempt to recover depending on configuration
                if self.config.get("system", {}).get("auto_restart_on_error", True):
                    self._attempt_recovery(thread_name)
    
    def _attempt_recovery(self, thread_name: str) -> None:
        """
        Attempt to recover a failed thread
        
        Args:
            thread_name: Name of the thread to recover
        """
        self.logger.info(f"Attempting to recover {thread_name} thread")
        
        # If capture thread fails, restart the camera
        if thread_name == "capture":
            self._recover_capture_thread()
        # If other threads fail, just restart them
        else:
            thread = self.threads.get(thread_name)
            if thread:
                # Stop the thread
                thread.stop()
                thread.join(timeout=2.0)
                
                # Restart the thread
                thread.start()
                
                self.stats["restart_count"] += 1
    
    def _recover_capture_thread(self) -> None:
        """Recover capture thread by restarting the camera"""
        # Stop and close camera
        self.camera.stop()
        self.camera.close()
        
        # Reinitialize camera
        if self.camera.initialize() and self.camera.start():
            # Create a new capture thread
            capture_thread = CaptureThread(
                camera=self.camera,
                max_queue_size=self.config.get("system", {}).get("max_queue_size", 10),
                frame_rate_limit=self.config.get("capture", {}).get("frame_rate_limit", 0)
            )
            capture_thread.set_output_queue(self.queues["raw_frames"])
            
            # Stop the old thread
            old_thread = self.threads.get("capture")
            if old_thread:
                old_thread.stop()
                old_thread.join(timeout=2.0)
                
            # Replace with new thread
            self.threads["capture"] = capture_thread
            capture_thread.start()
            
            self.stats["restart_count"] += 1
            self.logger.info("Capture thread recovered successfully")
        else:
            self.logger.error("Failed to recover capture thread")
    
    def _update_stats(self) -> None:
        """Update internal statistics"""
        # Update thread stats
        for thread_name, thread in self.threads.items():
            thread_stats = thread.get_stats()
            self.stats["thread_stats"][thread_name] = thread_stats
            
            # Update performance metrics
            if self.config.get("metrics", {}).get("enabled", True):
                metrics = get_metrics_instance()
                metrics.update_thread_metrics(thread_name, thread_stats)
        
        # Update queue stats
        for queue_name, q in self.queues.items():
            self.stats["queue_stats"][queue_name] = {
                "qsize": q.qsize(),
                "maxsize": q.maxsize
            }
            
        # Update detector stats
        if self.detector:
            detector_stats = self.detector.get_stats()
            self.stats["detector_stats"] = detector_stats
            
            # Update performance metrics for detector
            if self.config.get("metrics", {}).get("enabled", True):
                metrics = get_metrics_instance()
                metrics.update_detector_metrics(detector_stats)
                
        # Update camera stats
        if self.camera:
            camera_stats = {
                "frame_count": self.camera.frame_count,
                "last_error": self.camera.last_error,
                "is_open": self.camera.status != CameraStatus.CLOSED,
                "width": self.config.get("camera", {}).get("width", 1920),
                "height": self.config.get("camera", {}).get("height", 1080),
                "fps": self.camera.get_fps()
            }
            self.stats["camera_stats"] = camera_stats
            
            # Update performance metrics for camera
            if self.config.get("metrics", {}).get("enabled", True):
                metrics = get_metrics_instance()
                metrics.update_camera_metrics(camera_stats)
                
        # Calculate overall system FPS
        if "capture" in self.stats["thread_stats"]:
            capture_stats = self.stats["thread_stats"]["capture"]
            if "fps" in capture_stats:
                self.stats["system_fps"] = capture_stats["fps"]
    
    def get_stats(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            Dictionary with system statistics
        """
        # Update stats one more time to ensure they're current
        self._update_stats()
        
        # Calculate uptime
        if self.stats["start_time"]:
            if self.state == SystemState.RUNNING:
                current_uptime = time.time() - self.stats["start_time"]
            else:
                current_uptime = 0
                
            uptime = self.stats["uptime"] + current_uptime
        else:
            uptime = 0
            
        # Create stats summary
        stats_summary = {
            "state": self.state.name,
            "uptime": uptime,
            "error_count": self.stats["error_count"],
            "restart_count": self.stats["restart_count"],
            "last_error": self.stats["last_error"],
            "threads": {},
            "queues": {}
        }
        
        # Add summary stats for each thread
        for thread_name, thread_stats in self.stats["thread_stats"].items():
            stats_summary["threads"][thread_name] = {
                "state": self.threads[thread_name].state.name,
                "processed_frames": thread_stats.get("processed_frames", 0),
                "dropped_frames": thread_stats.get("dropped_frames", 0),
                "fps": thread_stats.get("fps", 0.0)
            }
            
        # Add queue stats
        for queue_name, queue_stats in self.stats["queue_stats"].items():
            stats_summary["queues"][queue_name] = queue_stats
            
        return stats_summary
    
    def get_detailed_stats(self) -> Dict:
        """
        Get detailed system statistics
        
        Returns:
            Dictionary with detailed system statistics
        """
        return self.stats
    
    def trigger_frame_selection(self) -> bool:
        """
        Manually trigger frame selection
        
        Returns:
            True if frame selection was triggered
        """
        quality_thread = self.threads.get("quality")
        if quality_thread:
            return quality_thread.trigger_selection()
        return False
    
    def update_config(self, config: Dict) -> bool:
        """
        Update configuration
        
        Args:
            config: New configuration dictionary
            
        Returns:
            True if configuration was updated successfully
        """
        # Store old config
        old_config = self.config
        
        # Update config
        self.config.update(config)
        
        # Update thread parameters
        self._update_thread_parameters()
        
        self.logger.info("Configuration updated")
        return True
    
    def _update_thread_parameters(self) -> None:
        """Update thread parameters from config"""
        # Update capture thread parameters if it exists
        capture_thread = self.threads.get("capture")
        if capture_thread:
            # Not much to update for capture thread
            pass
            
        # Update inference thread parameters if it exists
        inference_thread = self.threads.get("inference")
        if inference_thread:
            detection_params = DetectionParameters(
                main_conf_threshold=self.config.get("detector", {}).get("conf_threshold", 0.25),
                secondary_conf_threshold=self.config.get("detector", {}).get("secondary_conf_threshold", 0.5),
                target_classes=self.config.get("detector", {}).get("target_classes"),
                max_detections=self.config.get("detector", {}).get("max_detections", 20),
                min_detection_area=self.config.get("detector", {}).get("min_detection_area", 100)
            )
            inference_thread.set_detection_parameters(detection_params)
            
        # Update quality thread parameters if it exists
        quality_thread = self.threads.get("quality")
        if quality_thread:
            quality_params = QualityParameters(
                buffer_size=self.config.get("frame_selector", {}).get("buffer_size", 5),
                min_quality_score=self.config.get("frame_selector", {}).get("min_quality_score", 0.6),
                min_detection_score=self.config.get("frame_selector", {}).get("min_detection_score", 0.5),
                quality_weight=self.config.get("frame_selector", {}).get("quality_weight", 0.7),
                detection_weight=self.config.get("frame_selector", {}).get("detection_weight", 0.3),
                selection_interval=self.config.get("frame_selector", {}).get("selection_interval", 1.0),
                selection_cooldown=self.config.get("frame_selector", {}).get("selection_cooldown", 0.5),
                auto_select_best=self.config.get("frame_selector", {}).get("auto_select_best", True)
            )
            quality_thread.set_parameters(quality_params)
            
        # Update save thread parameters if it exists
        save_thread = self.threads.get("save")
        if save_thread:
            save_params = SaveParameters(
                base_dir=self.config.get("system", {}).get("save_dir", "images"),
                use_date_subdirs=self.config.get("image_storage", {}).get("storage_mode", "by_date") == "by_date",
                use_class_subdirs=self.config.get("image_storage", {}).get("storage_mode", "by_date") == "by_class",
                include_metadata=self.config.get("image_storage", {}).get("save_metadata", True),
                save_quality=self.config.get("image_storage", {}).get("jpeg_quality", 95),
                draw_detections=self.config.get("save", {}).get("draw_detections", True),
                save_original=self.config.get("save", {}).get("save_original", True),
                save_metadata_file=self.config.get("image_storage", {}).get("save_metadata", True),
                filename_template=self.config.get("save", {}).get("filename_template", 
                                                  "{timestamp}_{frame_id}_{quality:.2f}_{class_name}"),
                storage_instance=get_storage_instance(self.config)
            )
            save_thread.set_parameters(save_params)
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize_components()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop() 