#!/usr/bin/env python3
"""
Main Application for WMSV4AI
Integrates all components and provides command-line interface
"""

import os
import sys
import time
import logging
import argparse
import signal
import yaml
import json
import threading
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

from thread_manager import ThreadManager
from base import load_config, create_directory
from performance_metrics import get_metrics_instance
from image_storage import get_storage_instance

# Global variables
thread_manager = None
stop_event = threading.Event()
config_path = "config.yaml"
log_path = "logs"

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level
    """
    # Create logs directory if it doesn't exist
    create_directory(log_path)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = getattr(logging, "INFO")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_path, f"wmsv4ai_{time.strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )

def load_system_config() -> Dict:
    """
    Load system configuration from YAML file
    
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger("main")
    
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
            return {}
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config:
            logger.warning("Configuration file is empty. Using default configuration.")
            return {}
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def setup_signal_handlers() -> None:
    """Set up signal handlers for clean shutdown"""
    def signal_handler(sig, frame):
        logger = logging.getLogger("main")
        if sig == signal.SIGINT:
            logger.info("Received SIGINT (Ctrl+C), shutting down...")
        elif sig == signal.SIGTERM:
            logger.info("Received SIGTERM, shutting down...")
        
        # Signal all threads to stop
        stop_event.set()
        
        # Stop thread manager
        if thread_manager:
            thread_manager.stop()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def save_stats(stats: Dict, filename: str = "stats.json") -> None:
    """
    Save statistics to file
    
    Args:
        stats: Statistics dictionary
        filename: Output filename
    """
    logger = logging.getLogger("main")
    
    try:
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving statistics: {e}")

def print_system_info() -> None:
    """Print system information"""
    logger = logging.getLogger("main")
    
    try:
        import platform
        import psutil
        
        # System information
        logger.info(f"System: {platform.system()} {platform.version()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        
        # Memory information
        mem = psutil.virtual_memory()
        logger.info(f"Memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
        
        # Disk information
        disk = psutil.disk_usage('/')
        logger.info(f"Disk: {disk.total / (1024**3):.2f} GB total, {disk.free / (1024**3):.2f} GB free")
        
    except ImportError:
        logger.warning("psutil not installed, skipping detailed system information")
        logger.info(f"System: {platform.system()} {platform.version()}")
        logger.info(f"Python: {platform.python_version()}")

def main() -> int:
    """
    Main application entry point
    
    Returns:
        Exit code
    """
    global thread_manager
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WMSV4AI - Raspberry Pi Camera System")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    parser.add_argument("--stats", "-s", default="stats.json", help="Path to save statistics")
    
    args = parser.parse_args()
    
    # Update global config path
    global config_path
    config_path = args.config
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    try:
        # Print banner
        logger.info("=" * 80)
        logger.info("WMSV4AI - Raspberry Pi Camera System")
        logger.info("=" * 80)
        
        # Print system information
        print_system_info()
        
        # Set up signal handlers
        setup_signal_handlers()
        
        # Load configuration
        config = load_system_config()
        
        # Initialize performance metrics
        logger.info("Initializing performance metrics")
        metrics = get_metrics_instance(config)
        if config.get("metrics", {}).get("enabled", True):
            metrics.start_monitoring()
        
        # Initialize image storage
        logger.info("Initializing image storage")
        storage = get_storage_instance(config)
        
        # Create thread manager
        logger.info("Creating thread manager")
        thread_manager = ThreadManager(config)
        
        # Initialize components
        logger.info("Initializing components")
        if not thread_manager.initialize_components():
            logger.error("Failed to initialize components, exiting")
            return 1
        
        # Start thread manager
        logger.info("Starting thread manager")
        if not thread_manager.start():
            logger.error("Failed to start thread manager, exiting")
            return 1
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        # Main loop - wait for stop event
        try:
            while not stop_event.is_set():
                # Wait for stop event or periodic check
                stop_event.wait(1.0)
                
                # Check if thread manager is in error state
                if thread_manager.state.name == "ERROR":
                    logger.error("Thread manager in error state, shutting down")
                    break
                    
        except KeyboardInterrupt:
            # This should be caught by the signal handler, but just in case
            logger.info("Keyboard interrupt received, shutting down...")
        
        # Stop thread manager
        logger.info("Stopping thread manager")
        thread_manager.stop()
        
        # Stop performance monitoring
        if config.get("metrics", {}).get("enabled", True):
            logger.info("Stopping performance monitoring")
            metrics.stop_monitoring()
        
        # Save statistics
        logger.info("Saving statistics")
        combined_stats = thread_manager.get_stats()
        combined_stats["storage"] = storage.get_storage_stats()
        combined_stats["performance"] = metrics.get_latest_metrics()
        save_stats(combined_stats, args.stats)
        
        # Generate performance report
        logger.info("Generating performance report")
        perf_report = metrics.generate_performance_report()
        save_stats(perf_report, "performance_report.json")
        
        # Print final statistics
        logger.info("=" * 80)
        logger.info("System Statistics:")
        logger.info(f"Total runtime: {combined_stats['uptime']:.2f} seconds")
        logger.info(f"Error count: {combined_stats['error_count']}")
        logger.info(f"Restart count: {combined_stats['restart_count']}")
        
        # Thread stats
        logger.info("-" * 40)
        logger.info("Thread Statistics:")
        for thread_name, thread_stats in combined_stats['threads'].items():
            logger.info(f"  {thread_name}:")
            logger.info(f"    Processed frames: {thread_stats.get('processed_frames', 0)}")
            logger.info(f"    Dropped frames: {thread_stats.get('dropped_frames', 0)}")
            logger.info(f"    FPS: {thread_stats.get('fps', 0.0):.2f}")
            
        # Queue stats
        logger.info("-" * 40)
        logger.info("Queue Statistics:")
        for queue_name, queue_stats in combined_stats['queues'].items():
            logger.info(f"  {queue_name}: {queue_stats['qsize']}/{queue_stats['maxsize']}")
            
        # Check saved images
        try:
            image_dir = config.get("system", {}).get("save_dir", "images")
            today = datetime.now().strftime("%Y-%m-%d")
            today_dir = os.path.join(image_dir, today)
            
            if os.path.exists(today_dir):
                image_count = len([f for f in os.listdir(today_dir) if f.endswith('.jpg')])
                logger.info(f"Images saved today: {image_count}")
            else:
                logger.info("No images saved today")
        except Exception as e:
            logger.error(f"Error checking saved images: {e}")
        
        logger.info("=" * 80)
        
        logger.info("System shutdown complete")
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        
        # Attempt to stop thread manager if it was created
        if thread_manager:
            try:
                thread_manager.stop()
            except Exception as stop_e:
                logger.error(f"Error stopping thread manager: {stop_e}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
