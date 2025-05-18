#!/usr/bin/env python3
"""
Performance Metrics for WMSV4AI
Tracks system resources and performance metrics
"""

import os
import time
import logging
import threading
import json
import csv
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import numpy as np

# Import optional dependencies with fallbacks
try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
    has_gpu = True
except ImportError:
    has_gpu = False

# Define metric types
class MetricType(Enum):
    SYSTEM = 0
    CAMERA = 1
    DETECTOR = 2
    THREAD = 3
    CUSTOM = 4

class PerformanceMetrics:
    """Manages system performance metrics and resource tracking"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize performance metrics
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("performance_metrics")
        self.config = config or {}
        self.metrics = {
            "system": {},
            "camera": {},
            "detector": {},
            "threads": {},
            "custom": {}
        }
        self.history = {
            "system": [],
            "camera": [],
            "detector": [],
            "threads": {},
            "custom": {}
        }
        self.history_size = self.config.get("metrics", {}).get("history_size", 100)
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.reporting_interval = self.config.get("metrics", {}).get("reporting_interval", 5.0)
        self.last_update = time.time()
        self.start_time = time.time()
        
        # Create metrics directory
        self.metrics_dir = os.path.join("logs", "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Temp file paths
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.metrics_dir, f"metrics_{self.session_timestamp}.csv")
        self.json_path = os.path.join(self.metrics_dir, f"metrics_{self.session_timestamp}.json")
        
        # Validate system capabilities
        if psutil is None:
            self.logger.warning("psutil not installed, system metrics will be limited")

    def start_monitoring(self) -> bool:
        """
        Start metrics monitoring thread
        
        Returns:
            True if monitor started successfully
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.logger.warning("Monitor thread already running")
            return False
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="metrics_monitor"
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
        return True
    
    def stop_monitoring(self) -> None:
        """Stop metrics monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.logger.warning("Monitor thread not running")
            return
        
        self.logger.info("Stopping performance monitoring")
        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        
        # Save final reports
        self.save_metrics_json()
        self.save_metrics_csv()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Update system metrics
                self.update_system_metrics()
                
                # Check if it's time to report
                current_time = time.time()
                if current_time - self.last_update >= self.reporting_interval:
                    self.save_metrics_json()
                    self.last_update = current_time
                
                # Sleep for a short time
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics"""
        system_metrics = {}
        
        # Basic metrics that always work
        system_metrics["timestamp"] = time.time()
        system_metrics["uptime"] = time.time() - self.start_time
        
        # CPU metrics using psutil if available
        if psutil:
            # CPU usage
            system_metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            system_metrics["cpu_freq"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_metrics["memory_percent"] = memory.percent
            system_metrics["memory_available_mb"] = memory.available / (1024 * 1024)
            system_metrics["memory_used_mb"] = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            system_metrics["disk_percent"] = disk.percent
            system_metrics["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)
            
            # Temperature if available (especially for Raspberry Pi)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Find CPU temperature (different keys depending on system)
                    cpu_temp = None
                    for chip, temperatures in temps.items():
                        if chip in ["cpu_thermal", "coretemp", "cpu-thermal"]:
                            for temp in temperatures:
                                if temp.label in ["", "Core 0", "CPU"]:
                                    cpu_temp = temp.current
                                    break
                    if cpu_temp:
                        system_metrics["cpu_temp"] = cpu_temp
        
        # GPU metrics if available
        if has_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    system_metrics["gpu_percent"] = gpu.load * 100
                    system_metrics["gpu_memory_percent"] = gpu.memoryUtil * 100
                    system_metrics["gpu_temp"] = gpu.temperature
            except Exception as e:
                self.logger.warning(f"Error getting GPU metrics: {e}")
        
        # Update metrics
        self.metrics["system"] = system_metrics
        
        # Add to history
        self.history["system"].append(system_metrics)
        
        # Trim history if needed
        if len(self.history["system"]) > self.history_size:
            self.history["system"] = self.history["system"][-self.history_size:]
    
    def update_thread_metrics(self, thread_name: str, metrics: Dict) -> None:
        """
        Update metrics for a specific thread
        
        Args:
            thread_name: Name of the thread
            metrics: Thread metrics dictionary
        """
        self.metrics["threads"][thread_name] = metrics
        
        # Add to history with timestamp
        metrics_with_ts = metrics.copy()
        metrics_with_ts["timestamp"] = time.time()
        
        # Initialize thread history if not exists
        if thread_name not in self.history["threads"]:
            self.history["threads"][thread_name] = []
            
        self.history["threads"][thread_name].append(metrics_with_ts)
        
        # Trim history if needed
        if len(self.history["threads"][thread_name]) > self.history_size:
            self.history["threads"][thread_name] = self.history["threads"][thread_name][-self.history_size:]
    
    def update_detector_metrics(self, metrics: Dict) -> None:
        """
        Update object detector metrics
        
        Args:
            metrics: Detector metrics dictionary
        """
        self.metrics["detector"] = metrics
        
        # Add to history with timestamp
        metrics_with_ts = metrics.copy()
        metrics_with_ts["timestamp"] = time.time()
        self.history["detector"].append(metrics_with_ts)
        
        # Trim history if needed
        if len(self.history["detector"]) > self.history_size:
            self.history["detector"] = self.history["detector"][-self.history_size:]
    
    def update_camera_metrics(self, metrics: Dict) -> None:
        """
        Update camera metrics
        
        Args:
            metrics: Camera metrics dictionary
        """
        self.metrics["camera"] = metrics
        
        # Add to history with timestamp
        metrics_with_ts = metrics.copy()
        metrics_with_ts["timestamp"] = time.time()
        self.history["camera"].append(metrics_with_ts)
        
        # Trim history if needed
        if len(self.history["camera"]) > self.history_size:
            self.history["camera"] = self.history["camera"][-self.history_size:]
    
    def update_custom_metric(self, name: str, value: Any) -> None:
        """
        Update a custom metric
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics["custom"][name] = value
        
        # Add to history with timestamp
        if name not in self.history["custom"]:
            self.history["custom"][name] = []
            
        self.history["custom"][name].append({
            "timestamp": time.time(),
            "value": value
        })
        
        # Trim history if needed
        if len(self.history["custom"][name]) > self.history_size:
            self.history["custom"][name] = self.history["custom"][name][-self.history_size:]
    
    def get_latest_metrics(self) -> Dict:
        """
        Get latest metrics
        
        Returns:
            Combined metrics dictionary
        """
        return {
            "timestamp": time.time(),
            "system": self.metrics["system"],
            "camera": self.metrics["camera"],
            "detector": self.metrics["detector"],
            "threads": self.metrics["threads"],
            "custom": self.metrics["custom"]
        }
    
    def get_metric_history(self, metric_type: MetricType, name: Optional[str] = None) -> List:
        """
        Get history for a specific metric
        
        Args:
            metric_type: Type of metric
            name: Name for custom or thread metrics
            
        Returns:
            List of metric values over time
        """
        if metric_type == MetricType.SYSTEM:
            return self.history["system"]
        elif metric_type == MetricType.CAMERA:
            return self.history["camera"]
        elif metric_type == MetricType.DETECTOR:
            return self.history["detector"]
        elif metric_type == MetricType.THREAD:
            if name in self.history["threads"]:
                return self.history["threads"][name]
            return []
        elif metric_type == MetricType.CUSTOM:
            if name in self.history["custom"]:
                return self.history["custom"][name]
            return []
        return []
    
    def get_metric_average(self, metric_type: MetricType, metric_name: str, 
                           window: Optional[int] = None) -> float:
        """
        Calculate average value for a specific metric
        
        Args:
            metric_type: Type of metric
            metric_name: Name of the metric
            window: Number of recent samples to consider (None for all)
            
        Returns:
            Average value or 0 if not enough data
        """
        history = self.get_metric_history(metric_type)
        if not history:
            return 0
            
        # Limit window if specified
        if window is not None and window > 0:
            history = history[-window:]
        
        # Extract values
        values = []
        for entry in history:
            if metric_name in entry:
                values.append(entry[metric_name])
        
        # Calculate average
        if values:
            return sum(values) / len(values)
        return 0
    
    def save_metrics_json(self) -> bool:
        """
        Save current metrics to JSON file
        
        Returns:
            True if saved successfully
        """
        try:
            metrics = self.get_latest_metrics()
            with open(self.json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving metrics to JSON: {e}")
            return False
    
    def save_metrics_csv(self) -> bool:
        """
        Save metrics history to CSV file
        
        Returns:
            True if saved successfully
        """
        try:
            # System metrics to CSV
            if self.history["system"]:
                # Get all field names first
                fieldnames = set()
                for entry in self.history["system"]:
                    fieldnames.update(entry.keys())
                
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    for entry in self.history["system"]:
                        writer.writerow(entry)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {e}")
            return False
    
    def generate_performance_report(self) -> Dict:
        """
        Generate performance summary report
        
        Returns:
            Report dictionary
        """
        report = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "system_summary": {},
            "camera_summary": {},
            "detector_summary": {},
            "thread_summary": {}
        }
        
        # System summary
        if self.history["system"]:
            last_n = min(30, len(self.history["system"]))
            recent = self.history["system"][-last_n:]
            
            if "cpu_percent" in recent[0]:
                cpu_values = [entry.get("cpu_percent", 0) for entry in recent]
                report["system_summary"]["cpu_avg"] = sum(cpu_values) / len(cpu_values)
                report["system_summary"]["cpu_max"] = max(cpu_values)
            
            if "memory_percent" in recent[0]:
                mem_values = [entry.get("memory_percent", 0) for entry in recent]
                report["system_summary"]["memory_avg"] = sum(mem_values) / len(mem_values)
                report["system_summary"]["memory_max"] = max(mem_values)
            
            if "cpu_temp" in recent[0]:
                temp_values = [entry.get("cpu_temp", 0) for entry in recent]
                report["system_summary"]["cpu_temp_avg"] = sum(temp_values) / len(temp_values)
                report["system_summary"]["cpu_temp_max"] = max(temp_values)
        
        # Detector summary
        if self.history["detector"]:
            detector_metrics = self.history["detector"]
            if "avg_inference_time" in detector_metrics[0]:
                inf_times = [entry.get("avg_inference_time", 0) for entry in detector_metrics]
                report["detector_summary"]["inference_time_avg"] = sum(inf_times) / len(inf_times)
                report["detector_summary"]["inference_time_min"] = min(inf_times)
                report["detector_summary"]["inference_time_max"] = max(inf_times)
        
        # Thread summary for each thread
        for thread_name, history in self.history["threads"].items():
            if history and "fps" in history[0]:
                fps_values = [entry.get("fps", 0) for entry in history]
                
                thread_report = {}
                thread_report["fps_avg"] = sum(fps_values) / len(fps_values)
                thread_report["fps_max"] = max(fps_values)
                thread_report["fps_min"] = min(fps_values)
                
                report["thread_summary"][thread_name] = thread_report
        
        return report

# Singleton instance
_metrics_instance = None

def get_metrics_instance(config: Optional[Dict] = None) -> PerformanceMetrics:
    """
    Get singleton metrics instance
    
    Args:
        config: Configuration dictionary (only used for first initialization)
        
    Returns:
        PerformanceMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PerformanceMetrics(config)
    return _metrics_instance
