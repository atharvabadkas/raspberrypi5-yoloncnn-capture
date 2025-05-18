import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union

from base import BaseThread, FrameData, ThreadState
from yolo_ncnn import YoloNcnnDetector, BaseDetector

class DetectionParameters:    
    def __init__(self, 
                 main_conf_threshold: float = 0.25,
                 secondary_conf_threshold: float = 0.5,
                 target_classes: List[int] = None,
                 max_detections: int = 20,
                 min_detection_area: int = 100):
        
        self.main_conf_threshold = main_conf_threshold
        self.secondary_conf_threshold = secondary_conf_threshold
        self.target_classes = target_classes
        self.max_detections = max_detections
        self.min_detection_area = min_detection_area
        
    def to_dict(self) -> Dict:
        return {
            "main_conf_threshold": self.main_conf_threshold,
            "secondary_conf_threshold": self.secondary_conf_threshold,
            "target_classes": self.target_classes,
            "max_detections": self.max_detections,
            "min_detection_area": self.min_detection_area
        }

# Alias for DetectionParameters to maintain compatibility with verification tests
InferenceParameters = DetectionParameters

class InferenceThread(BaseThread):
    
    def __init__(self, 
                 detector: BaseDetector,
                 detection_params: DetectionParameters = None,
                 max_queue_size: int = 10,
                 enable_fallback: bool = True):
        
        super().__init__("InferenceThread", max_queue_size)
        self.detector = detector
        self.detection_params = detection_params or DetectionParameters()
        self.enable_fallback = enable_fallback
        self.fallback_detector = None
        self.fallback_mode = False
        self.detections_with_classes = {}  # Tracks detection counts by class
        self.consecutive_empty_frames = 0
        self.consecutive_error_frames = 0
        
        # Statistics tracking
        self.stats.update({
            "detection_count": 0,
            "detection_errors": 0,
            "frames_with_detection": 0,
            "frames_without_detection": 0,
            "fallback_activations": 0,
            "avg_inference_time": 0.0,
            "detection_counts_by_class": {},
            "detection_areas": [],
            "avg_confidence": 0.0,
            "error_count": 0,
            "total_processing_time": 0.0
        })
        
        # Ensure model is loaded
        if not self._ensure_detector_ready():
            self.logger.error("Failed to initialize detector")
    
    def _process_item(self, item: Any) -> Optional[Any]:
        if item is None or not isinstance(item, FrameData):
            self.logger.warning(f"Invalid item type: {type(item)}")
            return None
        
        frame_data = item
        
        try:
            # Run detection on the frame
            start_time = time.time()
            
            # Process the frame through the YOLO detector
            processed_frame = self._process_frame(frame_data)
            process_time = time.time() - start_time
            
            # Track timing statistics
            if not hasattr(self, "processing_times"):
                self.stats["processing_times"] = []
                
            if len(self.stats["processing_times"]) > 100:
                self.stats["processing_times"] = self.stats["processing_times"][-100:]
            self.stats["processing_times"].append(process_time)
            
            # If processing failed, skip this frame
            if not processed_frame:
                return None
                
            # Check for output queue fullness - if queue is nearing capacity, be more selective
            if hasattr(self, "output_queue") and self.output_queue:
                try:
                    queue_size = self.output_queue.qsize()
                    queue_capacity = self.output_queue.maxsize
                    queue_utilization = queue_size / queue_capacity if queue_capacity > 0 else 0
                    
                    # If queue is almost full, only allow person/hand detections through
                    if queue_utilization > 0.8:
                        has_priority_object = False
                        
                        if processed_frame.detection_results:
                            for detection in processed_frame.detection_results:
                                class_id = detection.get('class_id', None)
                                class_name = detection.get('class_name', '').lower()
                                confidence = detection.get('confidence', 0.0)
                                
                                if ((class_id == 0 or 'person' in class_name) and confidence > 0.4) or 'hand' in class_name:
                                    has_priority_object = True
                                    break
                        
                        # Filter aggressively when queue nearing capacity
                        if not has_priority_object:
                            self.logger.debug(f"Queue at {queue_utilization:.1%} capacity - filtering non-priority frame {frame_data.frame_id}")
                            return None
                except Exception as e:
                    self.logger.warning(f"Error checking queue utilization: {e}")
            
            # Filter based on detection content
            if processed_frame.detection_results:
                # Always forward frames with detections of interest (person, hand, plate, tray)
                has_person = False
                has_object_of_interest = False
                objects_of_interest = ['hand', 'plate', 'tray', 'bowl', 'cup', 'dining table', 'food']
                
                for detection in processed_frame.detection_results:
                    class_name = detection.get('class_name', '').lower()
                    class_id = detection.get('class_id', -1)
                    confidence = detection.get('confidence', 0)
                    
                    # Person is highest priority
                    if class_id == 0 or 'person' in class_name:
                        has_person = True
                    
                    # Check if any object of interest
                    if any(obj in class_name for obj in objects_of_interest):
                        has_object_of_interest = True
                
                # Forward based on priorities and frame counting
                if has_person:
                    # For person detections, only forward every Nth frame unless high confidence
                    # This reduces duplicate similar frames while ensuring we get good samples
                    if not hasattr(self, "person_frame_counter"):
                        self.person_frame_counter = 0
                    
                    # Forward frame if it's time or high confidence
                    high_confidence = any(d['confidence'] > 0.7 for d in processed_frame.detection_results if d.get('class_id', -1) == 0)
                    
                    if high_confidence or self.person_frame_counter % 3 == 0:
                        # Reset our counters for other frame types
                        if hasattr(self, "empty_frame_counter"):
                            self.empty_frame_counter = 0
                            
                        # Add metadata about the decision
                        processed_frame.add_metadata("forwarded_reason", "person_detected")
                        self.person_frame_counter += 1
                        return processed_frame
                    else:
                        # Skip this frame to avoid duplicates
                        self.person_frame_counter += 1
                        return None
                
                elif has_object_of_interest:
                    # Always forward frames with objects of interest
                    processed_frame.add_metadata("forwarded_reason", "object_of_interest")
                    return processed_frame
                    
                else:
                    # For other objects, forward but at lower priority
                    # Limit frequency of these frames
                    if not hasattr(self, "other_object_frame_counter"):
                        self.other_object_frame_counter = 0
                        
                    if self.other_object_frame_counter % 4 == 0:
                        processed_frame.add_metadata("forwarded_reason", "other_object")
                        self.other_object_frame_counter += 1
                        return processed_frame
                    else:
                        self.other_object_frame_counter += 1
                        return None
            else:
                # For empty frames, only forward occasionally for quality check
                if not hasattr(self, "empty_frame_counter"):
                    self.empty_frame_counter = 0
                
                # Forward only every 10th empty frame to avoid wasting resources
                if self.empty_frame_counter >= 10:
                    self.empty_frame_counter = 0
                    processed_frame.add_metadata("forwarded_reason", "periodic_empty")
                    return processed_frame
                else:
                    self.empty_frame_counter += 1
                    return None
                
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            self.stats["detection_errors"] += 1
            self.consecutive_error_frames += 1
            
            # Try fallback if enabled and not already in fallback mode
            if self.enable_fallback and not self.fallback_mode:
                if self._activate_fallback():
                    # Retry with fallback detector
                    self.logger.info("Retrying with fallback detector")
                    return self._process_item(frame_data)
            
            # If too many consecutive errors, stop thread
            if self.consecutive_error_frames > 10:
                self.logger.error("Too many consecutive detection errors, marking thread as error state")
                self.state = ThreadState.ERROR
            
            # Return frame without detections
            return None
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        # Apply confidence threshold
        filtered = [
            d for d in detections 
            if d["confidence"] >= self.detection_params.main_conf_threshold
        ]
        
        # Filter by target classes if specified
        if self.detection_params.target_classes:
            filtered = [
                d for d in filtered 
                if d["class_id"] in self.detection_params.target_classes
            ]
            
        # Calculate detection area and filter by minimum area
        for d in filtered:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            d["area"] = area
            
        filtered = [
            d for d in filtered 
            if d["area"] >= self.detection_params.min_detection_area
        ]
        
        # Sort by confidence (highest first) and limit number of detections
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        
        if len(filtered) > self.detection_params.max_detections:
            filtered = filtered[:self.detection_params.max_detections]
            
        return filtered
    
    def _update_detection_stats(self, detections: List[Dict]) -> None:
        # Update detection counts
        detection_count = len(detections)
        self.stats["detection_count"] += detection_count
        
        # Track consecutive empty frames
        if detection_count == 0:
            self.consecutive_empty_frames += 1
            self.stats["frames_without_detection"] += 1
        else:
            self.consecutive_empty_frames = 0
            self.stats["frames_with_detection"] += 1
            
        # Reset error counter if detection succeeded
        self.consecutive_error_frames = 0
        
        # Update detection counts by class
        class_counts = {}
        for d in detections:
            class_id = d["class_id"]
            class_name = d.get("class_name", str(class_id))
            
            # Update counts in current batch
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Update all-time counts
            if class_name not in self.stats["detection_counts_by_class"]:
                self.stats["detection_counts_by_class"][class_name] = 0
            self.stats["detection_counts_by_class"][class_name] += 1
            
        # Store current batch for quick access
        self.detections_with_classes = class_counts
        
        # Track detection areas
        if detections:
            areas = [d["area"] for d in detections]
            self.stats["detection_areas"].extend(areas)
            
            # Keep only the most recent 1000 areas to avoid memory growth
            if len(self.stats["detection_areas"]) > 1000:
                self.stats["detection_areas"] = self.stats["detection_areas"][-1000:]
        
        # Track average confidence
        if detections:
            confidences = [d["confidence"] for d in detections]
            avg_conf = sum(confidences) / len(confidences)
            
            # Rolling average for confidence
            if "confidence_count" not in self.stats:
                self.stats["confidence_count"] = 0
                self.stats["avg_confidence"] = 0.0
                
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * self.stats["confidence_count"] + avg_conf) / 
                (self.stats["confidence_count"] + 1)
            )
            self.stats["confidence_count"] += 1
    
    def _get_avg_inference_time(self) -> float:
        if self.fallback_mode and self.fallback_detector:
            # Handle YOLO fallback detector which doesn't have get_stats method
            if hasattr(self.fallback_detector, 'get_stats'):
                stats = self.fallback_detector.get_stats()
                if "avg_inference_time" in stats:
                    return stats["avg_inference_time"] * 1000  # Convert to ms
            # Use our internal timing stats for YOLO
            if "inference_times" in self.stats and self.stats["inference_times"]:
                avg_time = sum(self.stats["inference_times"]) / len(self.stats["inference_times"])
                return avg_time * 1000  # Convert to ms
        else:
            # Use primary detector stats
            stats = self.detector.get_stats()
            if "avg_inference_time" in stats:
                return stats["avg_inference_time"] * 1000  # Convert to ms
            elif "inference_times" in self.detector.stats and self.detector.stats["inference_times"]:
                avg_time = sum(self.detector.stats["inference_times"]) / len(self.detector.stats["inference_times"])
                return avg_time * 1000  # Convert to ms
        
        # Default value if no stats available
        return 0.0
    
    def _ensure_detector_ready(self) -> bool:
        try:
            # Initialize primary detector if needed
            if not hasattr(self.detector, "model") or self.detector.model is None:
                self.logger.info("Loading detector model")
                if not self.detector.load_model():
                    self.logger.error("Failed to load detector model")
                    return False
            
            # Initialize fallback detector if enabled
            if self.enable_fallback and self.fallback_detector is None:
                try:
                    from ultralytics import YOLO
                    
                    # Assume YOLO model path is in similar location but with .pt extension
                    if hasattr(self.detector, "model_path"):
                        pt_path = self.detector.model_path.replace(".param", ".pt")
                        if not pt_path.endswith(".pt"):
                            pt_path += ".pt"
                            
                        self.logger.info(f"Loading fallback PyTorch model from {pt_path}")
                        try:
                            self.fallback_detector = YOLO(pt_path)
                            self.logger.info("Fallback detector loaded successfully")
                        except Exception as e:
                            self.logger.warning(f"Failed to load fallback model: {e}")
                except ImportError:
                    self.logger.warning("Ultralytics package not available, fallback disabled")
                    self.enable_fallback = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring detector is ready: {e}", exc_info=True)
            return False
    
    def _activate_fallback(self) -> bool:
        if not self.enable_fallback or self.fallback_detector is None:
            return False
            
        try:
            self.logger.warning("Activating fallback detector")
            self.fallback_mode = True
            self.stats["fallback_activations"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating fallback: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict:
        stats = super().get_stats()
        
        # Add detector-specific stats
        try:
            detector_stats = self.detector.get_stats()
            
            for key, value in detector_stats.items():
                stats[f"detector_{key}"] = value
        except Exception as e:
            self.logger.warning(f"Error getting detector stats: {e}")
        
        # Add high-level detection stats
        if self.stats["frames_with_detection"] > 0:
            stats["detection_rate"] = (
                self.stats["frames_with_detection"] / 
                (self.stats["frames_with_detection"] + self.stats["frames_without_detection"])
            )
        else:
            stats["detection_rate"] = 0.0
        
        # Calculate average detection area
        if self.stats["detection_areas"]:
            stats["avg_detection_area"] = sum(self.stats["detection_areas"]) / len(self.stats["detection_areas"])
        else:
            stats["avg_detection_area"] = 0.0
        
        # Inference-specific stats
        stats["in_fallback_mode"] = self.fallback_mode
        
        # Safely get average inference time
        try:
            stats["avg_inference_time_ms"] = self._get_avg_inference_time()
        except Exception as e:
            self.logger.warning(f"Error getting inference time: {e}")
            stats["avg_inference_time_ms"] = 0.0
        
        return stats
    
    def set_detection_parameters(self, params: DetectionParameters) -> None:
        self.detection_params = params
        self.logger.info(f"Updated detection parameters: {params.to_dict()}")
        
    def has_detected_class(self, class_id: Union[int, str]) -> bool:
        if isinstance(class_id, int):
            # Convert to string for dictionary lookup
            class_id = str(class_id)
            
        return class_id in self.detections_with_classes and self.detections_with_classes[class_id] > 0

    def _process_frame(self, frame_data: FrameData) -> Optional[FrameData]:
        if frame_data is None:
            return None
        
        try:
            start_time = time.time()
            
            # Perform detection - use try/except to handle possible errors
            try:
                detections = self.detector.detect(frame_data.frame)
            except Exception as e:
                self.logger.error(f"Error in detector.detect(): {e}")
                self.stats["error_count"] += 1
                self.consecutive_error_frames += 1
                
                # Try fallback if enabled and not already in fallback mode
                if self.enable_fallback and not self.fallback_mode and self.fallback_detector is not None:
                    if self._activate_fallback():
                        # Try with fallback
                        try:
                            detections = self.fallback_detector.detect(frame_data.frame)
                        except Exception as e2:
                            self.logger.error(f"Error in fallback detector: {e2}")
                            return frame_data  # Return original frame without detections
                    else:
                        return frame_data  # Return original frame without detections
                else:
                    return frame_data  # Return original frame without detections
            
            # If no detections and fallback is enabled, try with PyTorch model
            if not detections and self.enable_fallback and self.fallback_detector:
                self.logger.info(f"No detections with NCNN model, trying fallback PyTorch model (frame {frame_data.frame_id})")
                try:
                    detections = self.fallback_detector.detect(frame_data.frame)
                except Exception as e:
                    self.logger.error(f"Error in fallback detector: {e}")
                    return frame_data  # Return original frame without detections
            
            # Process the detections
            filtered_detections = []
            person_detected = False
            
            for det in detections:
                # Ensure detection has all required fields
                if isinstance(det, dict):
                    detection = det
                else:
                    # Convert from ultralytics format if needed
                    x1, y1, x2, y2 = map(int, [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])
                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(det.confidence),
                        "class_id": int(det.class_id) if hasattr(det, 'class_id') else 0,
                        "class_name": det.class_name if hasattr(det, 'class_name') else "person"
                    }
                
                # Check if this is a person (class_id 0 in COCO)
                if detection.get("class_id") == 0 or detection.get("class_name", "").lower() == "person":
                    person_detected = True
                    # Boost confidence for persons to prioritize them
                    detection["confidence"] = min(detection["confidence"] * 1.2, 1.0)
                
                # Calculate area for filtering
                x1, y1, x2, y2 = detection["bbox"]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                detection["area"] = area
                
                # Filter by minimum size if needed
                if area >= self.detection_params.min_detection_area:
                    filtered_detections.append(detection)
            
            # Sort by confidence and limit number
            filtered_detections.sort(key=lambda x: x["confidence"], reverse=True)
            if self.detection_params.max_detections > 0:
                filtered_detections = filtered_detections[:self.detection_params.max_detections]
            
            # Add detections to frame data
            frame_data.detection_results = filtered_detections
            
            # Calculate detection score (average confidence)
            if filtered_detections:
                frame_data.detection_score = sum(d["confidence"] for d in filtered_detections) / len(filtered_detections)
                # Boost score if a person is detected
                if person_detected:
                    frame_data.detection_score = min(frame_data.detection_score * 1.5, 1.0)
            else:
                frame_data.detection_score = 0.0
            
            # Update stats
            self.stats["processed_frames"] += 1
            processing_time = time.time() - start_time
            self.stats["last_processing_time"] = processing_time
            self.stats["total_processing_time"] += processing_time
            
            # Calculate average processing time
            if self.stats["processed_frames"] > 0:
                self.stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["processed_frames"]
            
            # Print detection information
            if filtered_detections:
                detection_classes = [d.get("class_name", str(d.get("class_id", "unknown"))) for d in filtered_detections]
                detection_confidences = [f"{d['confidence']:.2f}" for d in filtered_detections]
                
                self.logger.info(f"Frame {frame_data.frame_id}: Detected {len(filtered_detections)} objects: " + 
                              f"{', '.join([f'{cls}({conf})' for cls, conf in zip(detection_classes, detection_confidences)])}" +
                              f" in {processing_time*1000:.1f}ms")
                
                # Log separately if person detected for easy tracking
                if person_detected:
                    self.logger.info(f"🚨 Person detected in frame {frame_data.frame_id} 🚨")
            else:
                self.logger.debug(f"Frame {frame_data.frame_id}: No detections in {processing_time*1000:.1f}ms")
            
            return frame_data
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
            return None
