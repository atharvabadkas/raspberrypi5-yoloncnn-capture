import os
import sys
import time
import logging
import datetime
import re
import uuid
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Setup default logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("assistance")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    # Convert string level to numeric
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    # Create logs directory if using a log file
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers and add new ones
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    for handler in handlers:
        root_logger.addHandler(handler)
        
    return root_logger

def get_timestamp(format_str: Optional[str] = None) -> Union[float, str]:
    if format_str:
        return time.strftime(format_str)
    return time.time()

def get_formatted_timestamp(timestamp: Optional[float] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    if timestamp is None:
        timestamp = time.time()
    return time.strftime(format_str, time.localtime(timestamp))

def get_date_path(timestamp: Optional[float] = None) -> str:
    if timestamp is None:
        timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp)
    return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"

def ensure_directory(path: str) -> bool:
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return os.path.isdir(path)
    except Exception as e:
        logger.error(f"Error creating directory '{path}': {e}")
        return False

def sanitize_filename(filename: str) -> str:
    # Replace invalid characters with underscores (one at a time)
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Ensure filename isn't too long
    if len(sanitized) > 255:
        # If too long, keep extension but truncate name
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
        
    return sanitized

def generate_unique_filename(prefix: str = "", extension: str = ".jpg") -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}{extension}"
    else:
        return f"{timestamp}_{unique_id}{extension}"

def calculate_md5(file_path: str) -> Optional[str]:
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating MD5 for '{file_path}': {e}")
        return None

def get_file_size(file_path: str) -> int:
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return 0
    except Exception as e:
        logger.error(f"Error getting file size for '{file_path}': {e}")
        return 0

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    try:
        if recursive:
            return [str(p) for p in Path(directory).rglob(pattern) if p.is_file()]
        else:
            return [str(p) for p in Path(directory).glob(pattern) if p.is_file()]
    except Exception as e:
        logger.error(f"Error finding files in '{directory}': {e}")
        return []

def safe_copy_file(src: str, dst: str, overwrite: bool = True) -> bool:
    try:
        if not os.path.exists(src):
            logger.error(f"Source file not found: '{src}'")
            return False
            
        if os.path.exists(dst) and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: '{dst}'")
            return False
            
        # Create destination directory if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            
        shutil.copy2(src, dst)
        return os.path.exists(dst)
    except Exception as e:
        logger.error(f"Error copying '{src}' to '{dst}': {e}")
        return False

def safe_move_file(src: str, dst: str, overwrite: bool = True) -> bool:
    try:
        if not os.path.exists(src):
            logger.error(f"Source file not found: '{src}'")
            return False
            
        if os.path.exists(dst):
            if not overwrite:
                logger.warning(f"Destination file exists and overwrite=False: '{dst}'")
                return False
            os.remove(dst)
            
        # Create destination directory if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            
        shutil.move(src, dst)
        return os.path.exists(dst) and not os.path.exists(src)
    except Exception as e:
        logger.error(f"Error moving '{src}' to '{dst}': {e}")
        return False

def is_raspberry_pi() -> bool:
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
        return 'raspberry pi' in model.lower()
    except:
        # Alternative check
        try:
            return os.path.exists('/sys/firmware/devicetree/base/model') and 'raspberry pi' in open('/sys/firmware/devicetree/base/model', 'r').read().lower()
        except:
            return False

def get_cpu_temp() -> float:
    try:
        if not is_raspberry_pi():
            return 0.0
            
        # Check different temperature file locations
        for temp_file in ['/sys/class/thermal/thermal_zone0/temp', '/sys/devices/virtual/thermal/thermal_zone0/temp']:
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp = float(f.read().strip()) / 1000.0  # Convert millidegrees to degrees
                return temp
        return 0.0
    except Exception as e:
        logger.error(f"Error getting CPU temperature: {e}")
        return 0.0

def get_system_info() -> Dict[str, Any]:
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "is_raspberry_pi": is_raspberry_pi(),
        "cpu_count": os.cpu_count() or 0,
        "cwd": os.getcwd()
    }
    
    # Add Raspberry Pi specific information
    if is_raspberry_pi():
        info["cpu_temp"] = get_cpu_temp()
        
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
                total_match = re.search(r'MemTotal:\s+(\d+)', mem_info)
                free_match = re.search(r'MemFree:\s+(\d+)', mem_info)
                
                if total_match and free_match:
                    total_kb = int(total_match.group(1))
                    free_kb = int(free_match.group(1))
                    info["memory_total_mb"] = total_kb / 1024
                    info["memory_free_mb"] = free_kb / 1024
        except:
            pass
            
    return info
