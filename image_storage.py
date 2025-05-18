import os
import time
import json
import shutil
import logging
import datetime
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import cv2
import numpy as np

from base import FrameData, FrameQuality, create_directory, get_timestamp

class StorageMode(Enum):
    FLAT = 0         # All images in one directory
    BY_DATE = 1      # Organized by date (YYYY/MM/DD)
    BY_QUALITY = 2   # Organized by quality assessment
    BY_CLASS = 3     # Organized by detected class
    CUSTOM = 4       # Custom organization function

class StorageFormat(Enum):
    JPEG = 0
    PNG = 1
    BMP = 2
    TIFF = 3

class ImageStorage:
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger("image_storage")
        self.config = config or {}
        
        # Base directory for all images
        self.base_dir = self.config.get("image_storage", {}).get("base_dir", "images")
        
        # Storage mode
        mode_str = self.config.get("image_storage", {}).get("storage_mode", "by_date")
        self.storage_mode = self._parse_storage_mode(mode_str)
        
        # Storage format
        format_str = self.config.get("image_storage", {}).get("format", "jpeg")
        self.format = self._parse_storage_format(format_str)
        
        # Image quality settings
        self.quality = self.config.get("image_storage", {}).get("jpeg_quality", 95)
        
        # File prefix
        self.file_prefix = self.config.get("image_storage", {}).get("file_prefix", "img")
        
        # Metadata settings
        self.save_metadata = self.config.get("image_storage", {}).get("save_metadata", True)
        
        # Create directory lock
        self.dir_lock = threading.Lock()
        
        # Create base directory
        self._ensure_base_directory()
        
        # Statistics
        self.stats = {
            "saved_images": 0,
            "saved_bytes": 0,
            "errors": 0,
            "by_quality": {q.name.lower(): 0 for q in FrameQuality}
        }
    
    def _parse_storage_mode(self, mode_str: str) -> StorageMode:
        mode_map = {
            "flat": StorageMode.FLAT,
            "by_date": StorageMode.BY_DATE,
            "by_quality": StorageMode.BY_QUALITY,
            "by_class": StorageMode.BY_CLASS,
            "custom": StorageMode.CUSTOM
        }
        return mode_map.get(mode_str.lower(), StorageMode.BY_DATE)
    
    def _parse_storage_format(self, format_str: str) -> StorageFormat:
        format_map = {
            "jpeg": StorageFormat.JPEG,
            "jpg": StorageFormat.JPEG,
            "png": StorageFormat.PNG,
            "bmp": StorageFormat.BMP,
            "tiff": StorageFormat.TIFF
        }
        return format_map.get(format_str.lower(), StorageFormat.JPEG)
    
    def _ensure_base_directory(self) -> None:
        with self.dir_lock:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir, exist_ok=True)
                self.logger.info(f"Created base directory: {self.base_dir}")
    
    def _get_storage_path(self, frame_data: FrameData) -> str:
        if self.storage_mode == StorageMode.FLAT:
            return self.base_dir
            
        elif self.storage_mode == StorageMode.BY_DATE:
            # Create YYYY/MM/DD directory structure
            date_obj = datetime.datetime.fromtimestamp(frame_data.timestamp)
            year_dir = os.path.join(self.base_dir, f"{date_obj.year:04d}")
            month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
            day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
            
            # Create directories
            with self.dir_lock:
                create_directory(year_dir)
                create_directory(month_dir)
                create_directory(day_dir)
                
            return day_dir
            
        elif self.storage_mode == StorageMode.BY_QUALITY:
            # Create directory by quality assessment
            quality_name = frame_data.quality_assessment.name.lower()
            quality_dir = os.path.join(self.base_dir, quality_name)
            
            # Create directory
            with self.dir_lock:
                create_directory(quality_dir)
                
            return quality_dir
            
        elif self.storage_mode == StorageMode.BY_CLASS:
            # Get primary detection class if available
            if frame_data.detection_results:
                # Sort by confidence and take the highest
                sorted_detections = sorted(
                    frame_data.detection_results, 
                    key=lambda x: x.get("confidence", 0), 
                    reverse=True
                )
                
                # Get class name
                class_name = sorted_detections[0].get("class", "unknown")
                class_dir = os.path.join(self.base_dir, class_name)
                
                # Create directory
                with self.dir_lock:
                    create_directory(class_dir)
                    
                return class_dir
            else:
                # No detections, use "unknown" directory
                unknown_dir = os.path.join(self.base_dir, "unknown")
                with self.dir_lock:
                    create_directory(unknown_dir)
                return unknown_dir
                
        # Default for custom or unknown modes
        return self.base_dir
    
    def _generate_filename(self, frame_data: FrameData) -> str:
        # Get timestamp components
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(frame_data.timestamp))
        ms = int((frame_data.timestamp - int(frame_data.timestamp)) * 1000)
        
        # Get quality info
        quality_str = frame_data.quality_assessment.name.lower()
        quality_score = f"{frame_data.quality_score:.2f}"
        
        # Get detection info
        detection_count = len(frame_data.detection_results)
        
        # Build filename components
        filename_parts = [
            self.file_prefix,
            timestamp_str,
            f"{ms:03d}",
            f"q{quality_score}",
            quality_str,
            f"d{detection_count}"
        ]
        
        # Add frame ID
        filename_parts.append(f"id{frame_data.frame_id}")
        
        # Join parts with underscores
        filename = "_".join(filename_parts)
        
        # Add extension based on format
        if self.format == StorageFormat.JPEG:
            filename += ".jpg"
        elif self.format == StorageFormat.PNG:
            filename += ".png"
        elif self.format == StorageFormat.BMP:
            filename += ".bmp"
        elif self.format == StorageFormat.TIFF:
            filename += ".tiff"
        
        return filename
    
    def _save_metadata(self, frame_data: FrameData, image_path: str) -> bool:
        try:
            # Create metadata dictionary
            metadata = {
                "timestamp": frame_data.timestamp,
                "frame_id": frame_data.frame_id,
                "quality": {
                    "score": frame_data.quality_score,
                    "assessment": frame_data.quality_assessment.name
                },
                "detections": frame_data.detection_results,
                "custom_metadata": frame_data.metadata
            }
            
            # Save to JSON file with same base name
            json_path = os.path.splitext(image_path)[0] + ".json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            return False
    
    def save_frame(self, frame_data: FrameData) -> Optional[str]:
        try:
            # Get storage directory
            storage_dir = self._get_storage_path(frame_data)
            
            # Generate filename
            filename = self._generate_filename(frame_data)
            
            # Full path
            image_path = os.path.join(storage_dir, filename)
            
            # Save image based on format
            if self.format == StorageFormat.JPEG:
                success = cv2.imwrite(image_path, frame_data.frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            elif self.format == StorageFormat.PNG:
                success = cv2.imwrite(image_path, frame_data.frame, 
                                     [cv2.IMWRITE_PNG_COMPRESSION, 1])
            elif self.format == StorageFormat.BMP:
                success = cv2.imwrite(image_path, frame_data.frame)
            elif self.format == StorageFormat.TIFF:
                success = cv2.imwrite(image_path, frame_data.frame)
            else:
                success = cv2.imwrite(image_path, frame_data.frame)
            
            if not success:
                self.logger.error(f"Failed to save image: {image_path}")
                self.stats["errors"] += 1
                return None
            
            # Update statistics
            self.stats["saved_images"] += 1
            self.stats["saved_bytes"] += os.path.getsize(image_path)
            
            # Update quality-specific stats
            quality_name = frame_data.quality_assessment.name.lower()
            if quality_name in self.stats["by_quality"]:
                self.stats["by_quality"][quality_name] += 1
            
            # Save metadata if enabled
            if self.save_metadata:
                self._save_metadata(frame_data, image_path)
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            self.stats["errors"] += 1
            return None
    
    def save_frames_batch(self, frames: List[FrameData]) -> List[str]:
        saved_paths = []
        for frame in frames:
            path = self.save_frame(frame)
            if path:
                saved_paths.append(path)
        return saved_paths
    
    def get_storage_stats(self) -> Dict:
        # Add current disk usage
        stats = self.stats.copy()
        
        # Add total disk usage if possible
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.base_dir):
                for filename in filenames:
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
            
            stats["total_disk_usage"] = total_size
            stats["total_disk_usage_mb"] = total_size / (1024 * 1024)
            
        except Exception as e:
            self.logger.warning(f"Error calculating total disk usage: {e}")
        
        return stats
    
    def cleanup_old_files(self, max_age_days: int = 30, 
                          min_quality_score: float = None) -> int:
        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        try:
            for dirpath, dirnames, filenames in os.walk(self.base_dir):
                for filename in filenames:
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        filepath = os.path.join(dirpath, filename)
                        file_time = os.path.getmtime(filepath)
                        file_age = current_time - file_time
                        
                        # Check age condition
                        if file_age > max_age_seconds:
                            # Check quality condition if specified
                            if min_quality_score is not None:
                                # Try to extract quality from filename or metadata
                                quality_score = self._extract_quality_score(filepath)
                                if quality_score is not None and quality_score >= min_quality_score:
                                    # High quality, keep it despite age
                                    continue
                            
                            # Delete file
                            os.remove(filepath)
                            deleted_count += 1
                            
                            # Delete metadata if exists
                            metadata_path = os.path.splitext(filepath)[0] + ".json"
                            if os.path.exists(metadata_path):
                                os.remove(metadata_path)
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
            return deleted_count
    
    def _extract_quality_score(self, filepath: str) -> Optional[float]:
        # First check metadata file
        metadata_path = os.path.splitext(filepath)[0] + ".json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if "quality" in metadata and "score" in metadata["quality"]:
                        return float(metadata["quality"]["score"])
            except Exception:
                pass
        
        # If not found in metadata, try filename
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        for part in parts:
            if part.startswith('q') and len(part) > 1:
                try:
                    return float(part[1:])
                except ValueError:
                    pass
        
        return None
    
    def archive_images(self, target_dir: str, selection_criteria: Dict = None) -> int:
        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        archived_count = 0
        min_quality = selection_criteria.get("min_quality", 0.0) if selection_criteria else 0.0
        min_date = selection_criteria.get("min_date") if selection_criteria else None
        max_date = selection_criteria.get("max_date") if selection_criteria else None
        
        # Convert dates to timestamps if provided
        min_timestamp = None
        if min_date:
            try:
                min_timestamp = datetime.datetime.strptime(min_date, "%Y-%m-%d").timestamp()
            except ValueError:
                self.logger.error(f"Invalid min_date format: {min_date}")
        
        max_timestamp = None
        if max_date:
            try:
                max_timestamp = datetime.datetime.strptime(max_date, "%Y-%m-%d").timestamp()
                # Set to end of day
                max_timestamp += 24 * 60 * 60 - 1
            except ValueError:
                self.logger.error(f"Invalid max_date format: {max_date}")
        
        try:
            for dirpath, dirnames, filenames in os.walk(self.base_dir):
                for filename in filenames:
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        filepath = os.path.join(dirpath, filename)
                        
                        # Check criteria
                        file_time = os.path.getmtime(filepath)
                        
                        # Check date criteria
                        if min_timestamp and file_time < min_timestamp:
                            continue
                        if max_timestamp and file_time > max_timestamp:
                            continue
                        
                        # Check quality criteria
                        if min_quality > 0:
                            quality_score = self._extract_quality_score(filepath)
                            if quality_score is None or quality_score < min_quality:
                                continue
                        
                        # Copy file to archive
                        target_path = os.path.join(target_dir, filename)
                        shutil.copy2(filepath, target_path)
                        archived_count += 1
                        
                        # Copy metadata if exists
                        metadata_path = os.path.splitext(filepath)[0] + ".json"
                        if os.path.exists(metadata_path):
                            metadata_target = os.path.splitext(target_path)[0] + ".json"
                            shutil.copy2(metadata_path, metadata_target)
            
            return archived_count
            
        except Exception as e:
            self.logger.error(f"Error archiving images: {e}")
            return archived_count

# Singleton instance
_storage_instance = None

def get_storage_instance(config: Optional[Dict] = None) -> ImageStorage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = ImageStorage(config)
    return _storage_instance 