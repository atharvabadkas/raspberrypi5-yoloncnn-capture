#!/usr/bin/env python3
"""
Quality Assessment Thread for WMSV4AI
Implements frame quality assessment and selection in a separate thread
"""

import time
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Deque
from collections import deque
from dataclasses import dataclass

from base import BaseThread, FrameData, FrameQuality, ThreadState
from image_quality import ImageQualityProcessor

@dataclass
class FrameScore:
    """Container for frame scoring data"""
    frame_data: FrameData
    quality_score: float
    detection_score: float
    combined_score: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "quality_score": self.quality_score,
            "detection_score": self.detection_score,
            "combined_score": self.combined_score,
            "timestamp": self.timestamp,
            "frame_id": self.frame_data.frame_id
        }


class QualityParameters:
    """Parameters for quality assessment and selection"""
    
    def __init__(self, 
                 buffer_size: int = 5,
                 min_quality_score: float = 0.6,
                 min_detection_score: float = 0.5,
                 quality_weight: float = 0.7,
                 detection_weight: float = 0.3,
                 selection_interval: float = 1.0,
                 selection_cooldown: float = 0.5,
                 auto_select_best: bool = True):
        """
        Initialize quality parameters
        
        Args:
            buffer_size: Size of the frame buffer
            min_quality_score: Minimum quality score for a good frame
            min_detection_score: Minimum detection score for a good frame
            quality_weight: Weight for quality score in combined score
            detection_weight: Weight for detection score in combined score
            selection_interval: Minimum time between frame selections
            selection_cooldown: Cooldown period after selecting a frame
            auto_select_best: Whether to automatically select the best frame
        """
        self.buffer_size = buffer_size
        self.min_quality_score = min_quality_score
        self.min_detection_score = min_detection_score
        self.quality_weight = quality_weight
        self.detection_weight = detection_weight
        self.selection_interval = selection_interval
        self.selection_cooldown = selection_cooldown
        self.auto_select_best = auto_select_best
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "buffer_size": self.buffer_size,
            "min_quality_score": self.min_quality_score,
            "min_detection_score": self.min_detection_score,
            "quality_weight": self.quality_weight,
            "detection_weight": self.detection_weight,
            "selection_interval": self.selection_interval,
            "selection_cooldown": self.selection_cooldown,
            "auto_select_best": self.auto_select_best
        }


class QualityThread(BaseThread):
    """Thread for assessing frame quality and selecting the best frames"""
    
    def __init__(self, 
                 quality_processor: ImageQualityProcessor,
                 params: QualityParameters = None,
                 max_queue_size: int = 10):
        """
        Initialize quality thread
        
        Args:
            quality_processor: Image quality processor
            params: Quality parameters
            max_queue_size: Maximum size of the input queue
        """
        super().__init__("QualityThread", max_queue_size)
        self.quality_processor = quality_processor
        self.params = params or QualityParameters()
        
        # Frame buffer for storing recent frames
        self.frame_buffer: Deque[FrameScore] = deque(maxlen=self.params.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Tracking state
        self.last_selection_time = 0.0
        self.cooldown_until = 0.0
        self.selection_event = threading.Event()
        self.selection_active = False
        self.best_frame: Optional[FrameScore] = None
        
        # Tracking objects by class
        self.tracked_objects = {}
        self.last_detection_by_class = {}
        
        # Statistics
        self.stats.update({
            "frames_assessed": 0,
            "frames_selected": 0,
            "frames_rejected": 0,
            "avg_quality_score": 0.0,
            "avg_detection_score": 0.0,
            "avg_combined_score": 0.0,
            "auto_selections": 0,
            "manual_selections": 0,
            "selection_triggers": {},
            "buffer_utilization": 0.0
        })
    
    def _process_item(self, item: Any) -> Optional[Any]:
        """
        Process a frame from the input queue
        
        Args:
            item: Frame data from input queue
            
        Returns:
            Selected frame data if a frame was selected, None otherwise
        """
        if item is None or not isinstance(item, FrameData):
            self.logger.warning(f"Invalid item type: {type(item)}")
            return None
        
        frame_data = item
        
        try:
            # Fast path - check for person detection first before doing quality assessment
            has_person = False
            has_hand = False
            person_confidence = 0.0
            
            if hasattr(frame_data, 'detection_results') and frame_data.detection_results:
                for detection in frame_data.detection_results:
                    class_id = detection.get('class_id', None)
                    class_name = detection.get('class_name', '').lower()
                    confidence = detection.get('confidence', 0.0)
                    
                    if class_id == 0 or 'person' in class_name:
                        has_person = True
                        person_confidence = max(person_confidence, confidence)
                        # Log person detection
                        self.logger.info(f"👤 Person detected in frame {frame_data.frame_id} with confidence {confidence}!")
                        
                    elif 'hand' in class_name:
                        has_hand = True
                        # Log hand detection
                        self.logger.info(f"👋 Hand detected in frame {frame_data.frame_id} with confidence {confidence}!")
            
            # HIGH PRIORITY PATH: For person/hand detections, prioritize saving these frames
            # This is our main goal - detect people/hands and save those images
            if has_person or has_hand:
                # For high-priority objects, process quality for better metadata
                if frame_data.quality_assessment == FrameQuality.UNKNOWN:
                    frame_data = self.quality_processor.process(frame_data)
                
                # Mark it as selected and pass it to the save thread
                self.stats["frames_selected"] += 1
                self.stats["auto_selections"] += 1
                
                # Update state to prevent rapid reselection
                current_time = time.time()
                self.last_selection_time = current_time
                self.cooldown_until = current_time + self.params.selection_cooldown
                
                # Log priority selection with debug info
                object_type = 'person' if has_person else 'hand'
                self.logger.info(f"✅ Priority selection of frame {frame_data.frame_id} containing {object_type}")
                self.logger.debug(f"Frame details: ID={frame_data.frame_id}, Quality={frame_data.quality_score:.2f}, Size={frame_data.frame.shape}")
                
                # MUST return the frame_data to pass it to the next thread
                return frame_data
            
            # NORMAL PATH: Process frame quality for non-high-priority frames
            if frame_data.quality_assessment == FrameQuality.UNKNOWN:
                frame_data = self.quality_processor.process(frame_data)
            
            # Score the frame
            frame_score = self._score_frame(frame_data)
            
            # Update statistics
            self._update_stats(frame_score)
            
            # Skip low-quality frames if buffer is getting full to reduce processing load
            buffer_threshold = self.params.buffer_size * 0.7
            min_score_threshold = self.params.min_quality_score
            
            with self.buffer_lock:
                buffer_size = len(self.frame_buffer)
                
                # Increase score threshold when buffer is filling up
                if buffer_size >= buffer_threshold:
                    min_score_threshold += 0.2
                
                # Don't add poor quality frames to buffer
                if frame_score.combined_score < min_score_threshold:
                    self.stats["frames_rejected"] += 1
                    return None
                
                # Add frame to buffer
                self.frame_buffer.append(frame_score)
                
                # Check if it's time to select a frame
                current_time = time.time()
                selection_due = (
                    current_time - self.last_selection_time >= self.params.selection_interval and
                    current_time >= self.cooldown_until and
                    buffer_size > 0
                )
                
                # Select if time for auto selection or manual trigger
                if self.params.auto_select_best and selection_due:
                    selected = self._select_best_frame()
                    if selected:
                        self.last_selection_time = current_time
                        self.cooldown_until = current_time + self.params.selection_cooldown
                        self.stats["auto_selections"] += 1
                        self.stats["frames_selected"] += 1
                        self.logger.info(f"Auto-selected frame {selected.frame_data.frame_id} with score {selected.combined_score:.3f}")
                        return selected.frame_data
                
                # Check for manual selection
                if self.selection_event.is_set():
                    self.selection_event.clear()
                    if buffer_size > 0:
                        selected = self._select_best_frame()
                        if selected:
                            self.last_selection_time = current_time
                            self.cooldown_until = current_time + self.params.selection_cooldown
                            self.stats["manual_selections"] += 1
                            self.stats["frames_selected"] += 1
                            self.logger.info(f"Manually selected frame {selected.frame_data.frame_id}")
                            return selected.frame_data
            
            # No frame was selected
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            self.stats["frames_rejected"] += 1
            self.stats["error_count"] = self.stats.get("error_count", 0) + 1
            return None
    
    def _score_frame(self, frame_data: FrameData) -> FrameScore:
        """
        Score a frame based on quality and detection metrics
        
        Args:
            frame_data: Frame data to score
            
        Returns:
            FrameScore object with scoring results
        """
        # Get quality score from frame_data
        quality_score = frame_data.quality_score
        
        # Calculate detection score based on detections
        detection_score = self._calculate_detection_score(frame_data)
        
        # Calculate combined score
        combined_score = (
            quality_score * self.params.quality_weight + 
            detection_score * self.params.detection_weight
        )
        
        # Create and return frame score
        return FrameScore(
            frame_data=frame_data,
            quality_score=quality_score,
            detection_score=detection_score,
            combined_score=combined_score,
            timestamp=time.time()
        )
    
    def _calculate_detection_score(self, frame_data: FrameData) -> float:
        """
        Calculate detection score based on detection results
        
        Args:
            frame_data: Frame data with detection results
            
        Returns:
            Detection score (0.0 to 1.0)
        """
        if not frame_data.detection_results:
            return 0.0
        
        # Calculate average confidence
        confidences = [d["confidence"] for d in frame_data.detection_results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate average relative area
        # (normalized to frame size)
        frame_area = frame_data.frame.shape[0] * frame_data.frame.shape[1]
        areas = []
        for d in frame_data.detection_results:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            rel_area = area / frame_area
            areas.append(min(rel_area, 1.0))  # Cap at 1.0
            
        # Small bonus for centered objects
        centering_scores = []
        frame_w = frame_data.frame.shape[1]
        frame_h = frame_data.frame.shape[0]
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2
        
        for d in frame_data.detection_results:
            x1, y1, x2, y2 = d["bbox"]
            # Calculate center of detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate normalized distance from frame center
            dist_x = abs(center_x - frame_center_x) / frame_w
            dist_y = abs(center_y - frame_center_y) / frame_h
            dist = (dist_x**2 + dist_y**2)**0.5
            
            # Convert to score (closer to center = higher score)
            centering_score = max(0.0, 1.0 - dist)
            centering_scores.append(centering_score)
        
        # Weight the different factors
        if areas:
            avg_area = sum(areas) / len(areas)
        else:
            avg_area = 0.0
            
        if centering_scores:
            avg_centering = sum(centering_scores) / len(centering_scores)
        else:
            avg_centering = 0.0
        
        # Combined detection score
        # 60% confidence, 30% area, 10% centering
        detection_score = (
            avg_confidence * 0.6 +
            avg_area * 0.3 +
            avg_centering * 0.1
        )
        
        return min(detection_score, 1.0)  # Cap at 1.0
    
    def _select_best_frame(self) -> Optional[FrameScore]:
        """
        Select the best frame from the buffer based on quality score
        
        Returns:
            Best frame from buffer or None if buffer empty
        """
        with self.buffer_lock:
            if not self.frame_buffer:
                return None
            
            # Sort frames by quality score
            best_frames = sorted(self.frame_buffer, key=lambda x: x.quality_score, reverse=True)
            
            # Take the best frame
            best_frame = best_frames[0]
            
            # Remove best frame from buffer
            self.frame_buffer.remove(best_frame)
            
            # Log selection
            self.logger.info(f"Selected best frame {best_frame.frame_data.frame_id} with quality score: {best_frame.quality_score:.2f}")
            if best_frame.frame_data.detection_results:
                class_names = [d["class_name"] for d in best_frame.frame_data.detection_results]
                class_counts = {}
                for name in class_names:
                    class_counts[name] = class_counts.get(name, 0) + 1
                class_str = ", ".join([f"{name}({count})" for name, count in class_counts.items()])
                self.logger.info(f"Frame contains {len(best_frame.frame_data.detection_results)} detections: {class_str}")
            
            return best_frame
    
    def _update_stats(self, frame_score: FrameScore) -> None:
        """
        Update statistics with new frame score
        
        Args:
            frame_score: Frame score to update statistics with
        """
        self.stats["frames_assessed"] += 1
        
        # Update rolling averages
        if "score_count" not in self.stats:
            self.stats["score_count"] = 0
            
        count = self.stats["score_count"]
        new_count = count + 1
        
        # Update average scores
        self.stats["avg_quality_score"] = (
            (self.stats["avg_quality_score"] * count + frame_score.quality_score) / new_count
        )
        self.stats["avg_detection_score"] = (
            (self.stats["avg_detection_score"] * count + frame_score.detection_score) / new_count
        )
        self.stats["avg_combined_score"] = (
            (self.stats["avg_combined_score"] * count + frame_score.combined_score) / new_count
        )
        
        self.stats["score_count"] = new_count
        
        # Update buffer utilization
        self.stats["buffer_utilization"] = len(self.frame_buffer) / self.params.buffer_size
        
        # Track detections by class
        detections = frame_score.frame_data.detection_results
        detected_classes = set()
        
        for detection in detections:
            class_id = detection.get("class_id")
            class_name = detection.get("class_name", str(class_id))
            detected_classes.add(class_name)
            
            # Update last detection time for this class
            self.last_detection_by_class[class_name] = time.time()
            
            # Update trigger counts
            if "selection_triggers" not in self.stats:
                self.stats["selection_triggers"] = {}
                
            if class_name not in self.stats["selection_triggers"]:
                self.stats["selection_triggers"][class_name] = 0
    
    def trigger_selection(self) -> bool:
        """
        Manually trigger frame selection
        
        Returns:
            True if selection was triggered
        """
        if self.state != ThreadState.RUNNING:
            return False
            
        current_time = time.time()
        if current_time < self.cooldown_until:
            self.logger.debug("Selection in cooldown period")
            return False
            
        self.selection_event.set()
        return True
    
    def get_best_frame(self) -> Optional[FrameScore]:
        """
        Get the current best frame
        
        Returns:
            Current best frame or None
        """
        with self.buffer_lock:
            return self.best_frame
    
    def get_buffer_frames(self) -> List[FrameScore]:
        """
        Get a list of all frames currently in the buffer
        
        Returns:
            List of frame scores
        """
        with self.buffer_lock:
            return list(self.frame_buffer)
    
    def set_parameters(self, params: QualityParameters) -> None:
        """
        Update quality parameters
        
        Args:
            params: New quality parameters
        """
        with self.buffer_lock:
            # Check if buffer size changed
            if params.buffer_size != self.params.buffer_size:
                # Create new buffer with new size
                new_buffer = deque(self.frame_buffer, maxlen=params.buffer_size)
                self.frame_buffer = new_buffer
                
            self.params = params
            self.logger.info(f"Updated quality parameters: {params.to_dict()}")
    
    def get_stats(self) -> Dict:
        """
        Get thread statistics
        
        Returns:
            Dictionary with thread statistics
        """
        stats = super().get_stats()
        
        # Add quality-specific stats
        stats.update({
            "buffer_size": self.params.buffer_size,
            "buffer_count": len(self.frame_buffer),
            "last_selection_time": self.last_selection_time,
            "cooldown_until": self.cooldown_until,
            "in_cooldown": time.time() < self.cooldown_until,
            "quality_params": self.params.to_dict()
        })
        
        # Add buffer stats
        if self.frame_buffer:
            buffer_quality_scores = [f.quality_score for f in self.frame_buffer]
            buffer_detection_scores = [f.detection_score for f in self.frame_buffer]
            buffer_combined_scores = [f.combined_score for f in self.frame_buffer]
            
            stats.update({
                "buffer_avg_quality": sum(buffer_quality_scores) / len(buffer_quality_scores),
                "buffer_avg_detection": sum(buffer_detection_scores) / len(buffer_detection_scores),
                "buffer_avg_combined": sum(buffer_combined_scores) / len(buffer_combined_scores),
                "buffer_max_combined": max(buffer_combined_scores),
                "buffer_age": time.time() - min(f.timestamp for f in self.frame_buffer)
            })
            
        # Add best frame stats
        if self.best_frame:
            stats["best_frame"] = self.best_frame.to_dict()
            
        return stats 

    def run(self) -> None:
        """Thread main loop"""
        self.logger.info(f"Thread {self.name} running")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Update thread state
        self.state = ThreadState.RUNNING
        
        # Log thread start
        self.logger.info(f"Thread {self.name} started")
        
        # Last selection time
        last_selection_time = time.time() - self.params.selection_interval
        
        try:
            while not self.stop_event.is_set():
                # Process frames
                frame_data = self._get_frame()
                
                if frame_data:
                    # Process frame quality
                    processed_frame = self._process_frame(frame_data)
                    self.logger.debug(f"Processed frame {frame_data.frame_id} with quality score {frame_data.quality_score:.2f}")
                    
                    if processed_frame:
                        # Add to buffer
                        with self.buffer_lock:
                            self.frame_buffer.append(processed_frame)
                            
                        # Check if it's time for selection
                        current_time = time.time()
                        
                        # If person is detected, force selection regardless of timer
                        has_person = False
                        if hasattr(processed_frame, 'detection_results') and processed_frame.detection_results:
                            for detection in processed_frame.detection_results:
                                if detection.get('class_id') == 0 or detection.get('class_name', '').lower() == 'person':
                                    has_person = True
                                    self.logger.info(f"Person detected in frame {processed_frame.frame_id}!")
                                    break
                        
                        # Select if it's time or if a person is detected
                        if has_person or (current_time - last_selection_time >= self.params.selection_interval):
                            # Select best frame
                            self.logger.info(f"Frame selection triggered - person detected: {has_person}")
                            selected_frame = self._select_best_frame()
                            
                            if selected_frame:
                                # Send to output queue
                                if self._send_frame(selected_frame.frame_data):
                                    self.logger.info(f"Selected and sent frame {selected_frame.frame_data.frame_id} to save queue")
                                    # Update last selection time
                                    last_selection_time = current_time
                                    # Update stats
                                    self.stats["frames_selected"] += 1
                                else:
                                    self.logger.warning("Output queue full, dropping selected frame")
                                    self.stats["dropped_frames"] += 1
                                    
        except Exception as e:
            self.logger.error(f"Error in thread {self.name}: {e}", exc_info=True)
            self.state = ThreadState.ERROR
            self.stats["last_error"] = str(e)
            self.stats["error_count"] += 1
            
        # Thread stopping
        self.state = ThreadState.STOPPED
        
        self.logger.info(f"Thread {self.name} stopped") 