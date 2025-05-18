#!/usr/bin/env python3
"""
Save Thread for WMSV4AI
Implements frame saving in a separate thread
"""

import os
import time
import logging
import threading
import datetime
import cv2
import numpy as np
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from base import BaseThread, FrameData, ThreadState
from image_storage import get_storage_instance

class SaveParameters:
    """Parameters for frame saving"""
    
    def __init__(self, 
                 base_dir: str = "images",
                 use_date_subdirs: bool = True,
                 use_class_subdirs: bool = True,
                 include_metadata: bool = True,
                 save_quality: int = 95,
                 draw_detections: bool = True,
                 save_original: bool = True,
                 save_metadata_file: bool = True,
                 filename_template: str = "{timestamp}_{frame_id}_{quality:.2f}_{class_name}",
                 storage_instance = None):
        """
        Initialize save parameters
        
        Args:
            base_dir: Base directory for saving frames
            use_date_subdirs: Whether to use date subdirectories (YYYY-MM-DD)
            use_class_subdirs: Whether to use class subdirectories
            include_metadata: Whether to include metadata in filename
            save_quality: JPEG save quality (1-100)
            draw_detections: Whether to draw detection boxes on saved images
            save_original: Whether to save original image without annotations
            save_metadata_file: Whether to save metadata as JSON file
            filename_template: Template for filename generation
            storage_instance: Optional ImageStorage instance
        """
        self.base_dir = base_dir
        self.use_date_subdirs = use_date_subdirs
        self.use_class_subdirs = use_class_subdirs
        self.include_metadata = include_metadata
        self.save_quality = save_quality
        self.draw_detections = draw_detections
        self.save_original = save_original
        self.save_metadata_file = save_metadata_file
        self.filename_template = filename_template
        self.storage_instance = storage_instance
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "base_dir": self.base_dir,
            "use_date_subdirs": self.use_date_subdirs,
            "use_class_subdirs": self.use_class_subdirs,
            "include_metadata": self.include_metadata,
            "save_quality": self.save_quality,
            "draw_detections": self.draw_detections,
            "save_original": self.save_original,
            "save_metadata_file": self.save_metadata_file,
            "filename_template": self.filename_template
        }


class SaveThread(BaseThread):
    """Thread for saving frames to disk"""
    
    def __init__(self, params: SaveParameters = None, max_queue_size: int = 20):
        """
        Initialize save thread
        
        Args:
            params: Save parameters
            max_queue_size: Maximum size of the input queue
        """
        super().__init__("SaveThread", max_queue_size)
        self.params = params or SaveParameters()
        self.last_save_time = 0.0
        self.last_saved_path = None
        self.save_lock = threading.Lock()
        
        # Get storage instance
        self.storage = self.params.storage_instance or get_storage_instance()
        
        # Statistics
        self.stats.update({
            "frames_saved": 0,
            "bytes_saved": 0,
            "save_errors": 0,
            "avg_save_time": 0.0,
            "classes_saved": {},
            "total_detections_saved": 0
        })
        
        # Ensure base directory exists
        self._ensure_base_dir()
    
    def _process_item(self, item: Any) -> Optional[Any]:
        """
        Process a frame from the input queue
        
        Args:
            item: Frame data from input queue
            
        Returns:
            Path to saved file or None if save failed
        """
        if item is None or not isinstance(item, FrameData):
            self.logger.warning(f"Invalid item type: {type(item)}")
            return None
        
        frame_data = item
        
        self.logger.info(f"📷 Received frame {frame_data.frame_id} for saving")
        
        try:
            # Check if frame has detection results before saving
            has_detections = hasattr(frame_data, 'detection_results') and frame_data.detection_results
            if has_detections:
                # Log detections info
                detection_classes = []
                for det in frame_data.detection_results:
                    class_name = det.get('class_name', str(det.get('class_id', 'unknown')))
                    confidence = det.get('confidence', 0.0)
                    detection_classes.append(f"{class_name}({confidence:.2f})")
                
                self.logger.info(f"Frame {frame_data.frame_id} has {len(frame_data.detection_results)} detections: {', '.join(detection_classes)}")
            
            # Process the frame using our method
            save_start_time = time.time()
            self._process_frame(frame_data)
            save_time = time.time() - save_start_time
            
            # Update stats
            self.stats["frames_saved"] += 1
            self.stats["avg_save_time"] = (self.stats["avg_save_time"] * (self.stats["frames_saved"] - 1) + save_time) / self.stats["frames_saved"]
            self.last_save_time = time.time()
            
            self.logger.info(f"✅ Successfully saved frame {frame_data.frame_id} in {save_time:.2f}s")
            
            # Return success indicator
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving frame {frame_data.frame_id}: {e}", exc_info=True)
            self.stats["save_errors"] += 1
            self.stats["last_error"] = str(e)
            return None
    
    def _prepare_frame_for_storage(self, frame_data: FrameData) -> None:
        """
        Prepare frame for storage system by adding annotations if needed
        
        Args:
            frame_data: Frame data to prepare
        """
        # If draw_detections is enabled and there are detections, add annotated frame to metadata
        if self.params.draw_detections and frame_data.detection_results:
            annotated_frame = self._draw_detections(frame_data)
            frame_data.add_metadata("annotated_frame", annotated_frame)
            
        # If save_original is enabled, ensure the original frame is preserved
        if self.params.save_original:
            frame_data.add_metadata("original_frame", frame_data.frame.copy())
    
    def _save_frame(self, frame_data: FrameData) -> str:
        """
        Save a frame to disk (legacy method)
        
        Args:
            frame_data: Frame data to save
            
        Returns:
            Path to saved file
        """
        # Get directory path
        dir_path = self._get_save_directory(frame_data)
        
        # Ensure directory exists
        os.makedirs(dir_path, exist_ok=True)
        
        # Generate filename
        base_filename = self._generate_filename(frame_data)
        
        # Save paths
        save_paths = []
        
        # Save original image if enabled
        if self.params.save_original:
            original_path = os.path.join(dir_path, f"{base_filename}_orig.jpg")
            cv2.imwrite(original_path, frame_data.frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
            save_paths.append(original_path)
            
        # Save image with detections if enabled
        if self.params.draw_detections and frame_data.detection_results:
            # Draw detections on a copy of the frame
            annotated_frame = self._draw_detections(frame_data)
            annotated_path = os.path.join(dir_path, f"{base_filename}.jpg")
            cv2.imwrite(annotated_path, annotated_frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
            save_paths.append(annotated_path)
        elif not self.params.save_original:
            # If not drawing detections and not saving original, save the frame as is
            default_path = os.path.join(dir_path, f"{base_filename}.jpg")
            cv2.imwrite(default_path, frame_data.frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
            save_paths.append(default_path)
        
        # Save metadata file if enabled
        if self.params.save_metadata_file:
            metadata_path = os.path.join(dir_path, f"{base_filename}.json")
            self._save_metadata(frame_data, metadata_path)
            save_paths.append(metadata_path)
            
        # Return the first save path (main image)
        return save_paths[0] if save_paths else ""
    
    def _get_save_directory(self, frame_data: FrameData) -> str:
        """
        Get directory path for saving frame
        
        Args:
            frame_data: Frame data to save
            
        Returns:
            Directory path
        """
        parts = [self.params.base_dir]
        
        # Add date subdirectory if enabled
        if self.params.use_date_subdirs:
            date_str = datetime.datetime.fromtimestamp(frame_data.timestamp).strftime("%Y-%m-%d")
            parts.append(date_str)
            
        # Add class subdirectory if enabled and there are detections
        if self.params.use_class_subdirs and frame_data.detection_results:
            # Use the class of the highest confidence detection
            detections = sorted(frame_data.detection_results, key=lambda d: d["confidence"], reverse=True)
            if detections:
                main_class = detections[0].get("class_name", str(detections[0].get("class_id", "unknown")))
                parts.append(main_class)
        
        return os.path.join(*parts)
    
    def _generate_filename(self, frame_data: FrameData) -> str:
        """
        Generate filename for frame
        
        Args:
            frame_data: Frame data to save
            
        Returns:
            Base filename (without extension)
        """
        # Extract metadata for filename
        timestamp_str = datetime.datetime.fromtimestamp(frame_data.timestamp).strftime("%Y%m%d_%H%M%S")
        ms = int((frame_data.timestamp - int(frame_data.timestamp)) * 1000)
        timestamp = f"{timestamp_str}_{ms:03d}"
        
        # Get main class if there are detections
        if frame_data.detection_results:
            detections = sorted(frame_data.detection_results, key=lambda d: d["confidence"], reverse=True)
            class_name = detections[0].get("class_name", str(detections[0].get("class_id", "unknown")))
        else:
            class_name = "unknown"
            
        # Get quality information
        quality = frame_data.quality_score
        quality_name = frame_data.quality_assessment.name.lower()
        
        # Get performance metrics if available
        detection_time = frame_data.metadata.get("inference_time", 0)
        
        # Build context dict for template formatting
        context = {
            "timestamp": timestamp,
            "ms": ms,
            "frame_id": frame_data.frame_id,
            "quality": quality,
            "quality_name": quality_name,
            "class_name": class_name,
            "detection_time": detection_time
        }
        
        # Add all metadata keys
        for key, value in frame_data.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                context[key] = value
        
        # Format filename using template
        try:
            filename = self.params.filename_template.format(**context)
        except KeyError as e:
            self.logger.warning(f"Invalid key in filename template: {e}")
            # Fallback to basic template
            filename = f"{timestamp}_{frame_data.frame_id}_{quality:.2f}_{class_name}"
            
        # Ensure filename is valid
        filename = self._sanitize_filename(filename)
        
        return filename
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to ensure it's valid
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
            
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
            
        return filename
    
    def _draw_detections(self, frame_data: FrameData) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame_data: Frame data with detections
            
        Returns:
            Annotated frame
        """
        # Make a copy of the frame
        annotated = frame_data.frame.copy()
        
        # Draw each detection
        for i, detection in enumerate(frame_data.detection_results):
            # Get detection data
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_id = detection.get("class_id", 0)
            class_name = detection.get("class_name", str(class_id))
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Generate color based on class_id
            color_id = class_id % 20  # Cycle through 20 different colors
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
                (64, 0, 64), (0, 64, 64), (192, 192, 192), (128, 128, 128)
            ]
            color = colors[color_id]
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1])
            
            # Draw label background
            cv2.rectangle(
                annotated, 
                (x1, y1_label - label_size[1] - baseline), 
                (x1 + label_size[0], y1_label),
                color, 
                cv2.FILLED
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, y1_label - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1
            )
            
        # Add frame metadata as text
        metadata_y = 20
        
        # Add quality information
        quality_text = f"Quality: {frame_data.quality_assessment.name} ({frame_data.quality_score:.2f})"
        cv2.putText(
            annotated,
            quality_text,
            (10, metadata_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1
        )
        metadata_y += 20
        
        # Add inference time if available
        if "inference_time" in frame_data.metadata:
            inference_text = f"Inference: {frame_data.metadata['inference_time']:.1f} ms"
            cv2.putText(
                annotated,
                inference_text,
                (10, metadata_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )
            metadata_y += 20
        
        # Add frame ID and timestamp
        timestamp_str = datetime.datetime.fromtimestamp(frame_data.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        frame_text = f"Frame: {frame_data.frame_id} | Time: {timestamp_str}"
        cv2.putText(
            annotated,
            frame_text,
            (10, metadata_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1
        )
        
        return annotated
    
    def _save_metadata(self, frame_data: FrameData, metadata_path: str) -> None:
        """
        Save frame metadata as JSON
        
        Args:
            frame_data: Frame data
            metadata_path: Path to save metadata
        """
        # Prepare metadata
        metadata = {
            "frame_id": frame_data.frame_id,
            "timestamp": frame_data.timestamp,
            "timestamp_str": datetime.datetime.fromtimestamp(frame_data.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "quality": {
                "score": frame_data.quality_score,
                "assessment": frame_data.quality_assessment.name
            },
            "detections": []
        }
        
        # Add detection data
        for detection in frame_data.detection_results:
            metadata["detections"].append({
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "class_id": detection.get("class_id", 0),
                "class_name": detection.get("class_name", str(detection.get("class_id", 0))),
                "area": detection.get("area", 0)
            })
            
        # Add other metadata
        for key, value in frame_data.metadata.items():
            # Skip complex objects that aren't JSON serializable
            if isinstance(value, (str, int, float, bool, list, dict)):
                metadata[key] = value
        
        # Save as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _ensure_base_dir(self) -> bool:
        """
        Ensure base directory exists
        
        Returns:
            True if directory exists or was created
        """
        try:
            os.makedirs(self.params.base_dir, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error creating base directory: {e}")
            return False
    
    def _update_save_stats(self, frame_data: FrameData, save_path: str, save_time: float) -> None:
        """
        Update statistics after saving a frame
        
        Args:
            frame_data: Saved frame data
            save_path: Path to saved file
            save_time: Time taken to save
        """
        self.stats["frames_saved"] += 1
        
        # Track save times
        if "save_times" not in self.stats:
            self.stats["save_times"] = []
            
        self.stats["save_times"].append(save_time)
        
        # Calculate average save time
        if len(self.stats["save_times"]) > 1000:
            self.stats["save_times"] = self.stats["save_times"][-1000:]
            
        self.stats["avg_save_time"] = sum(self.stats["save_times"]) / len(self.stats["save_times"])
        
        # Track file size
        try:
            if save_path and os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                self.stats["bytes_saved"] += file_size
        except Exception as e:
            self.logger.warning(f"Error getting file size: {e}")
            
        # Track classes saved
        if frame_data.detection_results:
            for detection in frame_data.detection_results:
                class_name = detection.get("class_name", str(detection.get("class_id", "unknown")))
                
                if class_name not in self.stats["classes_saved"]:
                    self.stats["classes_saved"][class_name] = 0
                    
                self.stats["classes_saved"][class_name] += 1
                self.stats["total_detections_saved"] += 1
    
    def get_stats(self) -> Dict:
        """
        Get thread statistics
        
        Returns:
            Statistics dictionary
        """
        stats = super().get_stats()
        
        # Add save specific stats
        stats.update({
            "frames_saved": self.stats["frames_saved"],
            "bytes_saved": self.stats["bytes_saved"],
            "save_errors": self.stats["save_errors"],
            "avg_save_time": self.stats["avg_save_time"],
            "classes_saved": self.stats["classes_saved"],
            "total_detections_saved": self.stats["total_detections_saved"],
            "last_save_time": self.last_save_time,
            "last_saved_path": self.last_saved_path
        })
        
        # Add storage stats if available
        if self.storage:
            storage_stats = self.storage.get_storage_stats()
            stats["storage"] = storage_stats
        
        return stats
    
    def set_parameters(self, params: SaveParameters) -> None:
        """
        Update save parameters
        
        Args:
            params: New save parameters
        """
        with self.save_lock:
            self.params = params
            
            # Update storage instance if provided
            if params.storage_instance:
                self.storage = params.storage_instance
            
            # Ensure base directory exists
            self._ensure_base_dir()
            
        self.logger.info(f"Updated save parameters: {params.to_dict()}")
    
    def get_total_saved_bytes(self) -> int:
        """
        Get total bytes saved
        
        Returns:
            Total bytes saved
        """
        return self.stats.get("bytes_saved", 0)
    
    def get_save_dir_size(self) -> int:
        """
        Get size of save directory
        
        Returns:
            Size in bytes
        """
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(self.params.base_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                        
            return total_size
            
        except Exception as e:
            self.logger.error(f"Error getting save directory size: {e}")
            return 0

    def _process_frame(self, frame_data: FrameData) -> None:
        """
        Process a single frame
        
        Args:
            frame_data: Frame data to process
        """
        if frame_data is None:
            return
        
        try:
            # Check for important detections (person, hand, etc.)
            has_important_detection = False
            detection_class = "unknown"
            
            if hasattr(frame_data, 'detection_results') and frame_data.detection_results:
                for detection in frame_data.detection_results:
                    class_id = detection.get('class_id', None)
                    class_name = detection.get('class_name', '').lower()
                    
                    if class_id == 0 or 'person' in class_name or 'hand' in class_name:
                        has_important_detection = True
                        detection_class = 'person' if 'person' in class_name or class_id == 0 else 'hand'
                        break
            
            # Prioritize saving important detections
            if has_important_detection:
                self.logger.info(f"🔍 Saving high-priority {detection_class} detection in frame {frame_data.frame_id}")
            
            # Get save directory
            save_dir = self._get_save_directory(frame_data)
            
            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Create base filename for this frame
            base_filename = self._generate_filename(frame_data)
            
            # Save original frame if configured
            if self.params.save_original:
                original_path = os.path.join(save_dir, f"{base_filename}_original.jpg")
                cv2.imwrite(original_path, frame_data.frame, [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
                self.logger.info(f"Saved original image: {original_path}")
                
                # Update last saved path for reporting
                self.last_saved_path = original_path
            
            # Save annotated frame with detection boxes
            has_detections = hasattr(frame_data, 'detection_results') and frame_data.detection_results
            if self.params.draw_detections and has_detections:
                # Make a copy of the frame to draw on
                annotated_frame = frame_data.frame.copy()
                
                # Draw detection boxes
                for det in frame_data.detection_results:
                    # Get detection info
                    bbox = det["bbox"]
                    confidence = det["confidence"]
                    class_name = det.get("class_name", str(det.get("class_id", "unknown")))
                    
                    # Determine color (red for person, green for others)
                    color = (0, 0, 255) if class_name.lower() == "person" or det.get("class_id") == 0 else (0, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        color, 
                        2
                    )
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated_frame,
                        (int(bbox[0]), int(bbox[1] - label_size[1] - 5)),
                        (int(bbox[0] + label_size[0]), int(bbox[1])),
                        color,
                        -1
                    )
                    cv2.putText(
                        annotated_frame, 
                        label, 
                        (int(bbox[0]), int(bbox[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
                    
                # Add timestamp and frame info
                timestamp = datetime.datetime.fromtimestamp(frame_data.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_data.frame_id} | Time: {timestamp}",
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                    
                # Save annotated frame
                annotated_path = os.path.join(save_dir, f"{base_filename}_annotated.jpg")
                cv2.imwrite(annotated_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
                self.logger.info(f"Saved annotated image with {len(frame_data.detection_results)} detections: {annotated_path}")
                
                # Update last saved path for reporting
                self.last_saved_path = annotated_path
            elif not self.params.save_original:
                # If not drawing detections and not saving original, save the frame as is
                default_path = os.path.join(save_dir, f"{base_filename}.jpg")
                cv2.imwrite(default_path, frame_data.frame, [cv2.IMWRITE_JPEG_QUALITY, self.params.save_quality])
                self.logger.info(f"Saved image: {default_path}")
                
                # Update last saved path for reporting
                self.last_saved_path = default_path
                
            # Save metadata if configured
            if self.params.save_metadata_file:
                metadata_path = os.path.join(save_dir, f"{base_filename}_metadata.json")
                self._save_metadata(frame_data, metadata_path)
                self.logger.debug(f"Saved metadata: {metadata_path}")
            
            # Update stats
            self.stats["processed_frames"] += 1
            self.stats["frames_saved"] += 1
            
            # Update storage metrics
            if self.params.storage_instance:
                self.params.storage_instance.update_storage_stats()
                
            self.logger.info(f"✅ Successfully saved frame {frame_data.frame_id} with quality score: {frame_data.quality_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            self.stats["error_count"] += 1
            self.stats["last_error"] = str(e)
