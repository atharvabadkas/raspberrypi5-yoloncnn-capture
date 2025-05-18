import os
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum

from base import FrameData, FrameQuality, BaseFrameProcessor

# Set up logging
logger = logging.getLogger("image_quality")

class QualityMetrics:    
    def __init__(self):
        self.blur_score = 0.0
        self.brightness_score = 0.0
        self.contrast_score = 0.0
        self.saturation_score = 0.0
        self.exposure_score = 0.0
        self.overall_score = 0.0
        
        # Raw metrics
        self.laplacian_var = 0.0
        self.mean_brightness = 0.0
        self.brightness_std = 0.0
        self.mean_saturation = 0.0
        self.overexposed_ratio = 0.0
        self.underexposed_ratio = 0.0
        
    def to_dict(self) -> Dict:
        return {
            "blur_score": self.blur_score,
            "brightness_score": self.brightness_score,
            "contrast_score": self.contrast_score,
            "saturation_score": self.saturation_score,
            "exposure_score": self.exposure_score,
            "overall_score": self.overall_score,
            "raw": {
                "laplacian_var": self.laplacian_var,
                "mean_brightness": self.mean_brightness,
                "brightness_std": self.brightness_std,
                "mean_saturation": self.mean_saturation,
                "overexposed_ratio": self.overexposed_ratio,
                "underexposed_ratio": self.underexposed_ratio
            }
        }


class QualityAssessor:    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"quality.{name}")
        
    def assess(self, frame: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement assess()")


class BlurAssessor(QualityAssessor):
    
    def __init__(self, threshold: float = 100.0):
        super().__init__("blur")
        self.threshold = threshold
        self.laplacian_var = 0.0
        
    def assess(self, frame: np.ndarray) -> float:
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            self.laplacian_var = laplacian.var()
            
            # Calculate score (0.0 to 1.0)
            # Higher variance = sharper image
            score = min(1.0, self.laplacian_var / self.threshold)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in blur assessment: {e}")
            return 0.0


class BrightnessAssessor(QualityAssessor):
    
    def __init__(self, min_brightness: int = 40, max_brightness: int = 220):
        super().__init__("brightness")
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.mean_brightness = 0.0
        self.brightness_std = 0.0
        
    def assess(self, frame: np.ndarray) -> float:
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Calculate brightness statistics
            self.mean_brightness = np.mean(gray)
            self.brightness_std = np.std(gray)
            
            # Calculate score (0.0 to 1.0)
            # Score is highest when brightness is in the optimal range
            if self.mean_brightness < self.min_brightness:
                # Too dark
                return max(0.0, self.mean_brightness / self.min_brightness)
            elif self.mean_brightness > self.max_brightness:
                # Too bright
                bright_range = 255 - self.max_brightness
                return max(0.0, 1.0 - ((self.mean_brightness - self.max_brightness) / bright_range))
            else:
                # Optimal range
                optimal_range = self.max_brightness - self.min_brightness
                optimal_middle = self.min_brightness + (optimal_range / 2)
                distance = abs(self.mean_brightness - optimal_middle)
                return 1.0 - (distance / (optimal_range / 2))
                
        except Exception as e:
            self.logger.error(f"Error in brightness assessment: {e}")
            return 0.0


class ContrastAssessor(QualityAssessor):
    
    def __init__(self, min_std: float = 50.0):
        super().__init__("contrast")
        self.min_std = min_std
        self.contrast_std = 0.0
        
    def assess(self, frame: np.ndarray) -> float:
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Calculate standard deviation (measure of contrast)
            self.contrast_std = np.std(gray)
            
            # Calculate score (0.0 to 1.0)
            # Higher std = better contrast
            score = min(1.0, self.contrast_std / self.min_std)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in contrast assessment: {e}")
            return 0.0


class SaturationAssessor(QualityAssessor):
    
    def __init__(self, min_saturation: float = 10.0, max_saturation: float = 150.0):
        super().__init__("saturation")
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.mean_saturation = 0.0
        
    def assess(self, frame: np.ndarray) -> float:
        try:
            # Check if image is color
            if len(frame.shape) != 3:
                # Grayscale images have no saturation
                self.mean_saturation = 0.0
                return 0.0
                
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Extract saturation channel
            sat = hsv[:, :, 1]
            
            # Calculate mean saturation
            self.mean_saturation = np.mean(sat)
            
            # Calculate score (0.0 to 1.0)
            if self.mean_saturation < self.min_saturation:
                # Too low saturation
                return max(0.0, self.mean_saturation / self.min_saturation)
            elif self.mean_saturation > self.max_saturation:
                # Too high saturation
                sat_range = 255 - self.max_saturation
                return max(0.0, 1.0 - ((self.mean_saturation - self.max_saturation) / sat_range))
            else:
                # Optimal range
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error in saturation assessment: {e}")
            return 0.0


class ExposureAssessor(QualityAssessor):
    
    def __init__(self, underexposed_threshold: int = 30, overexposed_threshold: int = 220, 
                 max_bad_ratio: float = 0.1):
        super().__init__("exposure")
        self.underexposed_threshold = underexposed_threshold
        self.overexposed_threshold = overexposed_threshold
        self.max_bad_ratio = max_bad_ratio
        self.underexposed_ratio = 0.0
        self.overexposed_ratio = 0.0
        
    def assess(self, frame: np.ndarray) -> float:
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Count under/overexposed pixels
            total_pixels = gray.size
            underexposed = np.sum(gray < self.underexposed_threshold)
            overexposed = np.sum(gray > self.overexposed_threshold)
            
            # Calculate ratios
            self.underexposed_ratio = underexposed / total_pixels
            self.overexposed_ratio = overexposed / total_pixels
            
            # Calculate combined bad exposure ratio
            bad_ratio = self.underexposed_ratio + self.overexposed_ratio
            
            # Calculate score (0.0 to 1.0)
            # Lower bad_ratio = better exposure
            score = 1.0 - min(1.0, bad_ratio / self.max_bad_ratio)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in exposure assessment: {e}")
            return 0.0


class ImageQualityProcessor(BaseFrameProcessor):
    
    def __init__(self, config: Dict = None):
        super().__init__("image_quality")
        
        # Load configuration
        config_mgr = None
        if config is None:
            try:
                import config_manager
                config_mgr = config_manager.get_config_manager()
                if config_mgr:
                    config = config_mgr.get("image_quality")
                else:
                    config = {}
            except ImportError:
                config = {}
        
        # Ensure config is a dictionary
        if config is None:
            config = {}
        
        # Initialize assessors
        self.blur_assessor = BlurAssessor(
            threshold=config.get("blur_threshold", 100.0)
        )
        
        self.brightness_assessor = BrightnessAssessor(
            min_brightness=config.get("min_brightness", 40),
            max_brightness=config.get("max_brightness", 220)
        )
        
        self.contrast_assessor = ContrastAssessor(
            min_std=config.get("min_contrast", 50.0)
        )
        
        self.saturation_assessor = SaturationAssessor(
            min_saturation=config.get("min_saturation", 10.0),
            max_saturation=config.get("max_saturation", 150.0)
        )
        
        self.exposure_assessor = ExposureAssessor(
            underexposed_threshold=config.get("underexposed_threshold", 30),
            overexposed_threshold=config.get("overexposed_threshold", 220),
            max_bad_ratio=config.get("max_bad_ratio", 0.1)
        )
        
        # Weight for each assessor in overall score
        self.weights = {
            "blur": config.get("weight_blur", 0.4),
            "brightness": config.get("weight_brightness", 0.2),
            "contrast": config.get("weight_contrast", 0.2),
            "saturation": config.get("weight_saturation", 0.1),
            "exposure": config.get("weight_exposure", 0.1)
        }
        
        # Configure thresholds for quality assessment
        self.quality_thresholds = {
            FrameQuality.POOR: 0.3,
            FrameQuality.ACCEPTABLE: 0.5,
            FrameQuality.GOOD: 0.7,
            FrameQuality.EXCELLENT: 0.85
        }
        
        # Register for configuration updates
        if config_mgr:
            try:
                config_mgr.register_callback(self.update_config)
            except:
                self.logger.warning("Failed to register config callback")
    
    def update_config(self) -> None:
        try:
            import config_manager
            config_mgr = config_manager.get_config_manager()
            if not config_mgr:
                return
                
            config = config_mgr.get("image_quality")
            if not config:
                return
                
            # Update assessors
            self.blur_assessor.threshold = config.get("blur_threshold", self.blur_assessor.threshold)
            
            self.brightness_assessor.min_brightness = config.get("min_brightness", self.brightness_assessor.min_brightness)
            self.brightness_assessor.max_brightness = config.get("max_brightness", self.brightness_assessor.max_brightness)
            
            self.contrast_assessor.min_std = config.get("min_contrast", self.contrast_assessor.min_std)
            
            self.saturation_assessor.min_saturation = config.get("min_saturation", self.saturation_assessor.min_saturation)
            self.saturation_assessor.max_saturation = config.get("max_saturation", self.saturation_assessor.max_saturation)
            
            self.exposure_assessor.underexposed_threshold = config.get("underexposed_threshold", self.exposure_assessor.underexposed_threshold)
            self.exposure_assessor.overexposed_threshold = config.get("overexposed_threshold", self.exposure_assessor.overexposed_threshold)
            self.exposure_assessor.max_bad_ratio = config.get("max_bad_ratio", self.exposure_assessor.max_bad_ratio)
            
            # Update weights
            self.weights = {
                "blur": config.get("weight_blur", self.weights["blur"]),
                "brightness": config.get("weight_brightness", self.weights["brightness"]),
                "contrast": config.get("weight_contrast", self.weights["contrast"]),
                "saturation": config.get("weight_saturation", self.weights["saturation"]),
                "exposure": config.get("weight_exposure", self.weights["exposure"])
            }
        except ImportError:
            self.logger.warning("config_manager module not available")
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
    
    def assess_quality(self, frame: np.ndarray) -> QualityMetrics:
        metrics = QualityMetrics()
        
        # Calculate individual metrics
        metrics.blur_score = self.blur_assessor.assess(frame)
        metrics.brightness_score = self.brightness_assessor.assess(frame)
        metrics.contrast_score = self.contrast_assessor.assess(frame)
        metrics.saturation_score = self.saturation_assessor.assess(frame)
        metrics.exposure_score = self.exposure_assessor.assess(frame)
        
        # Store raw metrics
        metrics.laplacian_var = self.blur_assessor.laplacian_var
        metrics.mean_brightness = self.brightness_assessor.mean_brightness
        metrics.brightness_std = self.brightness_assessor.brightness_std
        metrics.mean_saturation = self.saturation_assessor.mean_saturation
        metrics.overexposed_ratio = self.exposure_assessor.overexposed_ratio
        metrics.underexposed_ratio = self.exposure_assessor.underexposed_ratio
        
        # Calculate weighted average
        metrics.overall_score = (
            metrics.blur_score * self.weights["blur"] +
            metrics.brightness_score * self.weights["brightness"] +
            metrics.contrast_score * self.weights["contrast"] +
            metrics.saturation_score * self.weights["saturation"] +
            metrics.exposure_score * self.weights["exposure"]
        )
        
        return metrics
    
    def get_quality_assessment(self, score: float) -> FrameQuality:
        if score < self.quality_thresholds[FrameQuality.POOR]:
            return FrameQuality.POOR
        elif score < self.quality_thresholds[FrameQuality.ACCEPTABLE]:
            return FrameQuality.ACCEPTABLE
        elif score < self.quality_thresholds[FrameQuality.GOOD]:
            return FrameQuality.GOOD
        else:
            return FrameQuality.EXCELLENT
    
    def process(self, frame_data: FrameData) -> FrameData:
        try:
            # Assess image quality
            metrics = self.assess_quality(frame_data.frame)
            
            # Set quality in frame data
            quality = self.get_quality_assessment(metrics.overall_score)
            frame_data.set_quality(metrics.overall_score, quality)
            
            # Add metrics to frame metadata
            frame_data.add_metadata("quality_metrics", metrics.to_dict())
            
            return frame_data
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            # Return frame with unknown quality on error
            frame_data.set_quality(0.0, FrameQuality.UNKNOWN)
            return frame_data


def create_image_quality_annotation(frame: np.ndarray, metrics: QualityMetrics) -> np.ndarray:
    # Create a copy of the frame
    result = frame.copy()
    
    # Create background rectangle for text
    cv2.rectangle(result, (0, 0), (300, 160), (0, 0, 0), -1)
    cv2.rectangle(result, (0, 0), (300, 160), (255, 255, 255), 1)
    
    # Add metrics text
    cv2.putText(result, f"Quality: {metrics.overall_score:.2f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Blur: {metrics.blur_score:.2f}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Brightness: {metrics.brightness_score:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Contrast: {metrics.contrast_score:.2f}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Saturation: {metrics.saturation_score:.2f}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Exposure: {metrics.exposure_score:.2f}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add raw metrics
    cv2.putText(result, f"Mean Brightness: {metrics.mean_brightness:.1f}", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def save_debug_image(frame: np.ndarray, metrics: QualityMetrics, save_dir: str, prefix: str = "quality") -> str:
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{metrics.overall_score:.2f}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Create annotated image
        annotated = create_image_quality_annotation(frame, metrics)
        
        # Save image
        cv2.imwrite(filepath, annotated)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving debug image: {e}")
        return ""
