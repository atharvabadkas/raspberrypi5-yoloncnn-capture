import time
import threading
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
from config_manager import ConfigManager
from image_quality import ImageQualityAnalyzer
from base import BaseFrameProcessor, FrameData, ThreadState

@dataclass
class FrameScore:
    frame_data: FrameData
    quality_score: float
    timestamp: float
    metadata: Dict[str, Any]

class FrameSelector(BaseFrameProcessor):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__("FrameSelector")
        self.config = config_manager
        self.quality_analyzer = ImageQualityAnalyzer(config_manager)
        
        # Initialize frame buffer
        self.buffer_size = self.config.get('frame_selector.buffer_size', 5)
        self.frame_buffer: deque[FrameScore] = deque(maxlen=self.buffer_size)
        
        # Selection parameters
        self.min_quality_score = self.config.get('frame_selector.min_quality_score', 0.6)
        self.selection_interval = self.config.get('frame_selector.selection_interval', 1.0)
        self.select_best_frame = self.config.get('frame_selector.select_best_frame', True)
        
        # Timing control
        self.last_selection_time = 0.0
        self.selection_lock = threading.Lock()
        
        # Statistics
        self.stats.update({
            'frames_processed': 0,
            'frames_selected': 0,
            'average_quality': 0.0,
            'buffer_utilization': 0.0,
            'selection_latency': 0.0
        })
        
        # Register for config updates
        self.config.register_callback(self._on_config_update)
        
    def _on_config_update(self, section: str, key: str, value: Any) -> None:
        if section == 'frame_selector':
            if key == 'buffer_size':
                with self.selection_lock:
                    self.buffer_size = value
                    # Resize buffer if needed
                    while len(self.frame_buffer) > self.buffer_size:
                        self.frame_buffer.popleft()
            elif key == 'min_quality_score':
                self.min_quality_score = value
            elif key == 'selection_interval':
                self.selection_interval = value
            elif key == 'select_best_frame':
                self.select_best_frame = value
    
    def process_frame(self, frame_data: FrameData) -> Optional[FrameData]:
        if not self.is_running or frame_data is None:
            return None
            
        try:
            # Start timing
            start_time = time.time()
            
            # Analyze frame quality
            quality_score = self.quality_analyzer.analyze_quality(frame_data.frame)
            
            # Create frame score
            frame_score = FrameScore(
                frame_data=frame_data,
                quality_score=quality_score,
                timestamp=time.time(),
                metadata={
                    'quality_category': self.quality_analyzer.get_quality_category(quality_score),
                    'quality_metrics': self.quality_analyzer.get_quality_metrics()
                }
            )
            
            # Update buffer
            with self.selection_lock:
                self.frame_buffer.append(frame_score)
                
                # Check if it's time to select a frame
                current_time = time.time()
                if (current_time - self.last_selection_time) >= self.selection_interval:
                    selected_frame = self._select_frame()
                    if selected_frame:
                        self.last_selection_time = current_time
                        self.stats['frames_selected'] += 1
                        return selected_frame.frame_data
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['average_quality'] = (
                (self.stats['average_quality'] * (self.stats['frames_processed'] - 1) + quality_score) /
                self.stats['frames_processed']
            )
            self.stats['buffer_utilization'] = len(self.frame_buffer) / self.buffer_size
            self.stats['selection_latency'] = time.time() - start_time
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return None
    
    def _select_frame(self) -> Optional[FrameScore]:
        if not self.frame_buffer:
            return None
            
        try:
            if self.select_best_frame:
                # Select frame with highest quality score
                best_frame = max(self.frame_buffer, key=lambda x: x.quality_score)
                if best_frame.quality_score >= self.min_quality_score:
                    return best_frame
            else:
                # Select first frame that meets quality threshold
                for frame in self.frame_buffer:
                    if frame.quality_score >= self.min_quality_score:
                        return frame
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Error selecting frame: {str(e)}")
            return None
    
    def get_buffer_status(self) -> Dict[str, Any]:
        with self.selection_lock:
            return {
                'buffer_size': self.buffer_size,
                'current_frames': len(self.frame_buffer),
                'oldest_timestamp': self.frame_buffer[0].timestamp if self.frame_buffer else None,
                'newest_timestamp': self.frame_buffer[-1].timestamp if self.frame_buffer else None,
                'average_quality': sum(f.quality_score for f in self.frame_buffer) / len(self.frame_buffer) if self.frame_buffer else 0.0
            }
    
    def clear_buffer(self) -> None:
        with self.selection_lock:
            self.frame_buffer.clear()
            self.last_selection_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(self.get_buffer_status())
        return stats
