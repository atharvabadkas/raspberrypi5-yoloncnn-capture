import os
import time
import logging
import yaml
import json
import threading
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable

# Set up logging
logger = logging.getLogger("config_manager")

# Default configuration values
DEFAULT_CONFIG = {
    "system": {
        "name": "WMSV4AI",
        "version": "1.0.0",
        "log_level": "INFO",
        "save_dir": "images",
        "enable_debug": False,
        "gpu_enabled": False,
        "num_threads": 4,
        "max_queue_size": 10
    },
    "camera": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "format": "RGB888",
        "rotation": 0,
        "camera_id": 0,
        "auto_exposure": True,
        "exposure_compensation": 0,
        "white_balance": "auto"
    },
    "detector": {
        "enabled": True,
        "model_param_path": "models/yolov8n.param",
        "model_bin_path": "models/yolov8n.bin",
        "class_file": "models/coco.names",
        "conf_threshold": 0.25,
        "nms_threshold": 0.45,
        "input_width": 640,
        "input_height": 640,
        "use_fp16": True,
        "num_threads": 4
    },
    "image_quality": {
        "enabled": True,
        "min_brightness": 40,
        "max_brightness": 220,
        "blur_threshold": 100,
        "min_contrast": 50,
        "save_all_frames": False,
        "save_good_frames": True,
        "save_debug_frames": False
    },
    "frame_selector": {
        "enabled": True,
        "buffer_size": 5,
        "min_quality_score": 0.6,
        "selection_interval": 1.0,
        "select_best_frame": True
    }
}

# Schema for configuration validation
CONFIG_SCHEMA = {
    "system": {
        "name": {"type": str, "required": False},
        "version": {"type": str, "required": False},
        "log_level": {"type": str, "required": False, "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
        "save_dir": {"type": str, "required": True},
        "enable_debug": {"type": bool, "required": False},
        "gpu_enabled": {"type": bool, "required": False},
        "num_threads": {"type": int, "required": False, "min": 1, "max": 32},
        "max_queue_size": {"type": int, "required": False, "min": 1, "max": 100}
    },
    "camera": {
        "width": {"type": int, "required": True, "min": 320, "max": 4096},
        "height": {"type": int, "required": True, "min": 240, "max": 4096},
        "fps": {"type": int, "required": True, "min": 1, "max": 120},
        "format": {"type": str, "required": False, "options": ["RGB888", "YUV420", "JPEG"]},
        "rotation": {"type": int, "required": False, "options": [0, 90, 180, 270]},
        "camera_id": {"type": int, "required": True, "min": 0, "max": 10},
        "auto_exposure": {"type": bool, "required": False},
        "exposure_compensation": {"type": int, "required": False, "min": -25, "max": 25},
        "white_balance": {"type": str, "required": False}
    },
    "detector": {
        "enabled": {"type": bool, "required": False},
        "model_param_path": {"type": str, "required": True},
        "model_bin_path": {"type": str, "required": False},
        "class_file": {"type": str, "required": False},
        "conf_threshold": {"type": float, "required": False, "min": 0.01, "max": 1.0},
        "nms_threshold": {"type": float, "required": False, "min": 0.01, "max": 1.0},
        "input_width": {"type": int, "required": False, "min": 32, "max": 4096},
        "input_height": {"type": int, "required": False, "min": 32, "max": 4096},
        "use_fp16": {"type": bool, "required": False},
        "num_threads": {"type": int, "required": False, "min": 1, "max": 32}
    },
    "image_quality": {
        "enabled": {"type": bool, "required": False},
        "min_brightness": {"type": int, "required": False, "min": 0, "max": 255},
        "max_brightness": {"type": int, "required": False, "min": 0, "max": 255},
        "blur_threshold": {"type": float, "required": False, "min": 1.0},
        "min_contrast": {"type": float, "required": False, "min": 0.0},
        "save_all_frames": {"type": bool, "required": False},
        "save_good_frames": {"type": bool, "required": False},
        "save_debug_frames": {"type": bool, "required": False}
    },
    "frame_selector": {
        "enabled": {"type": bool, "required": False},
        "buffer_size": {"type": int, "required": False, "min": 1, "max": 100},
        "min_quality_score": {"type": float, "required": False, "min": 0.0, "max": 1.0},
        "selection_interval": {"type": float, "required": False, "min": 0.1},
        "select_best_frame": {"type": bool, "required": False}
    }
}

class ConfigError(Exception):
    pass

class ConfigManager:
    
    def __init__(self, config_path: str, auto_reload: bool = False, reload_interval: float = 5.0):
        self.config_path = config_path
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.last_modified_time = 0
        self.lock = threading.RLock()
        self.reload_thread = None
        self.stop_event = threading.Event()
        self.callbacks = []
        
        # Load configuration at initialization
        self.load()
        
        # Start auto-reload thread if enabled
        if self.auto_reload:
            self._start_auto_reload()
    
    def load(self) -> bool:
        with self.lock:
            try:
                # Check if file exists
                if not os.path.exists(self.config_path):
                    logger.warning(f"Configuration file not found: {self.config_path}")
                    logger.info("Using default configuration")
                    self._save_default_config()
                    return False
                
                # Check if file modification time has changed
                modified_time = os.path.getmtime(self.config_path)
                if modified_time <= self.last_modified_time:
                    return True  # No changes
                
                # Load YAML configuration
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Update and validate configuration
                if loaded_config:
                    self._update_config(loaded_config)
                    self.last_modified_time = modified_time
                    logger.info(f"Configuration loaded from {self.config_path}")
                    self._notify_callbacks()
                    return True
                else:
                    logger.warning("Empty configuration file, using default")
                    return False
                    
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML configuration: {e}")
                return False
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return False
    
    def save(self) -> bool:
        with self.lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Write YAML configuration
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                
                self.last_modified_time = os.path.getmtime(self.config_path)
                logger.info(f"Configuration saved to {self.config_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                return False
    
    def get(self, section: str = None, key: str = None, default: Any = None) -> Any:
        with self.lock:
            try:
                if section is None:
                    return copy.deepcopy(self.config)
                
                if section not in self.config:
                    return default
                
                if key is None:
                    return copy.deepcopy(self.config[section])
                
                if key not in self.config[section]:
                    return default
                
                return copy.deepcopy(self.config[section][key])
                
            except Exception as e:
                logger.error(f"Error getting configuration value [{section}.{key}]: {e}")
                return default
    
    def set(self, section: str, key: str, value: Any) -> bool:
        with self.lock:
            try:
                # Validate section
                if section not in self.config:
                    logger.error(f"Invalid configuration section: {section}")
                    return False
                
                # Validate key
                if key not in self.config[section]:
                    logger.error(f"Invalid configuration key: {section}.{key}")
                    return False
                
                # Validate value
                schema = CONFIG_SCHEMA.get(section, {}).get(key, {})
                if schema and not self._validate_value(value, schema):
                    logger.error(f"Invalid value for {section}.{key}: {value}")
                    return False
                
                # Set value
                self.config[section][key] = value
                
                # Auto-save
                self.save()
                
                # Notify callbacks
                self._notify_callbacks()
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting configuration value [{section}.{key}]: {e}")
                return False
    
    def update_section(self, section: str, values: Dict) -> bool:
        with self.lock:
            try:
                # Validate section
                if section not in self.config:
                    logger.error(f"Invalid configuration section: {section}")
                    return False
                
                # Validate and update values
                for key, value in values.items():
                    if key not in self.config[section]:
                        logger.warning(f"Unknown configuration key: {section}.{key}")
                        continue
                    
                    # Validate value
                    schema = CONFIG_SCHEMA.get(section, {}).get(key, {})
                    if schema and not self._validate_value(value, schema):
                        logger.error(f"Invalid value for {section}.{key}: {value}")
                        continue
                    
                    # Set value
                    self.config[section][key] = value
                
                # Auto-save
                self.save()
                
                # Notify callbacks
                self._notify_callbacks()
                
                return True
                
            except Exception as e:
                logger.error(f"Error updating configuration section [{section}]: {e}")
                return False
    
    def register_callback(self, callback: Callable) -> None:
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def reset_to_defaults(self) -> bool:
        with self.lock:
            try:
                self.config = copy.deepcopy(DEFAULT_CONFIG)
                self.save()
                self._notify_callbacks()
                logger.info("Configuration reset to defaults")
                return True
                
            except Exception as e:
                logger.error(f"Error resetting configuration: {e}")
                return False
    
    def validate(self) -> List[str]:
        errors = []
        
        for section_name, section_schema in CONFIG_SCHEMA.items():
            if section_name not in self.config:
                if any(item.get("required", False) for item in section_schema.values()):
                    errors.append(f"Required section missing: {section_name}")
                continue
            
            section_config = self.config[section_name]
            
            for key_name, key_schema in section_schema.items():
                if key_schema.get("required", False) and key_name not in section_config:
                    errors.append(f"Required key missing: {section_name}.{key_name}")
                    continue
                
                if key_name in section_config:
                    value = section_config[key_name]
                    
                    # Type validation
                    expected_type = key_schema.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(f"Invalid type for {section_name}.{key_name}: expected {expected_type.__name__}, got {type(value).__name__}")
                    
                    # Range validation for numbers
                    if isinstance(value, (int, float)):
                        min_val = key_schema.get("min")
                        max_val = key_schema.get("max")
                        
                        if min_val is not None and value < min_val:
                            errors.append(f"Value for {section_name}.{key_name} is too small: {value} < {min_val}")
                        
                        if max_val is not None and value > max_val:
                            errors.append(f"Value for {section_name}.{key_name} is too large: {value} > {max_val}")
                    
                    # Options validation
                    options = key_schema.get("options")
                    if options and value not in options:
                        errors.append(f"Invalid value for {section_name}.{key_name}: {value} not in {options}")
        
        return errors
    
    def close(self) -> None:
        self.stop_event.set()
        if self.reload_thread and self.reload_thread.is_alive():
            self.reload_thread.join(timeout=1.0)
    
    def _save_default_config(self) -> None:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Write default configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Default configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving default configuration: {e}")
    
    def _update_config(self, loaded_config: Dict) -> None:
        for section_name, section_data in loaded_config.items():
            if section_name not in self.config:
                logger.warning(f"Unknown configuration section: {section_name}")
                continue
            
            if not isinstance(section_data, dict):
                logger.warning(f"Invalid section format for {section_name}")
                continue
            
            for key_name, value in section_data.items():
                if key_name not in self.config[section_name]:
                    logger.warning(f"Unknown configuration key: {section_name}.{key_name}")
                    continue
                
                # Validate value
                schema = CONFIG_SCHEMA.get(section_name, {}).get(key_name, {})
                if schema and not self._validate_value(value, schema):
                    logger.warning(f"Invalid value for {section_name}.{key_name}: {value}, using default")
                    continue
                
                # Update value
                self.config[section_name][key_name] = value
    
    def _validate_value(self, value: Any, schema: Dict) -> bool:
        # Type validation
        expected_type = schema.get("type")
        if expected_type and not isinstance(value, expected_type):
            return False
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = schema.get("min")
            max_val = schema.get("max")
            
            if min_val is not None and value < min_val:
                return False
            
            if max_val is not None and value > max_val:
                return False
        
        # Options validation
        options = schema.get("options")
        if options and value not in options:
            return False
        
        return True
    
    def _start_auto_reload(self) -> None:
        self.stop_event.clear()
        self.reload_thread = threading.Thread(
            target=self._auto_reload_loop,
            daemon=True,
            name="ConfigAutoReload"
        )
        self.reload_thread.start()
    
    def _auto_reload_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Check if file exists and has been modified
                if os.path.exists(self.config_path):
                    self.load()
            except Exception as e:
                logger.error(f"Error in auto-reload: {e}")
            
            # Sleep with interrupt check
            self.stop_event.wait(self.reload_interval)
    
    def _notify_callbacks(self) -> None:
        for callback in self.callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in configuration callback: {e}")

    def get_value_as_json(self) -> str:
        try:
            return json.dumps(self.config, indent=2)
        except Exception as e:
            logger.error(f"Error converting configuration to JSON: {e}")
            return "{}"
    
    def get_config_debug_info(self) -> Dict:
        return {
            "config_path": self.config_path,
            "auto_reload": self.auto_reload,
            "reload_interval": self.reload_interval,
            "last_modified_time": time.ctime(self.last_modified_time) if self.last_modified_time > 0 else "Never",
            "validation_errors": self.validate(),
            "callback_count": len(self.callbacks)
        }


# Function to create a global configuration manager
config_manager = None

def init_config_manager(config_path: str = "config.yaml", auto_reload: bool = False) -> ConfigManager:
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager(config_path, auto_reload)
    return config_manager

def get_config_manager() -> Optional[ConfigManager]:
    return config_manager

def get_config(section: str = None, key: str = None, default: Any = None) -> Any:
    if config_manager is None:
        logger.warning("Config manager not initialized, returning default")
        return default
        
    return config_manager.get(section, key, default)
