import os
import sys
import time
import logging
import importlib
import argparse
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("verifications")

# Define component list
COMPONENTS = [
    "base",
    "camera",
    "yolo_ncnn",
    "capture_thread",
    "inference_thread",
    "quality_thread",
    "save_thread",
    "thread_manager",
    "performance_metrics",
    "image_quality",
    "image_storage",
    "config_manager",
    "assistance"
]

# Define scripts to verify
SCRIPTS = [
    "main.py",
    "convert_model.py"
]

def print_status(message: str, success: bool = True) -> bool:
    """Print status message with color and symbol"""
    symbol = "✅" if success else "❌"
    if success:
        logger.info(f"{symbol} {message}")
    else:
        logger.error(f"{symbol} {message}")
    return success

def check_imports() -> bool:
    """Check if all components can be imported"""
    logger.info("Checking component imports...")
    
    all_passed = True
    for component in COMPONENTS:
        try:
            module = importlib.import_module(component)
            print_status(f"Import {component}: Success")
        except ImportError as e:
            print_status(f"Import {component}: Failed - {e}", False)
            all_passed = False
            
    return all_passed

def check_scripts_executable() -> bool:
    """Check if scripts are executable"""
    logger.info("Checking if scripts are executable...")
    
    all_passed = True
    for script in SCRIPTS:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print_status(f"Script {script} is executable")
            else:
                print_status(f"Script {script} is not executable", False)
                try:
                    os.chmod(script, 0o755)
                    print_status(f"Made {script} executable")
                except:
                    all_passed = False
        else:
            print_status(f"Script {script} not found", False)
            all_passed = False
            
    return all_passed

def test_base_module() -> bool:
    """Test functionality of base.py module"""
    logger.info("Testing base module...")
    
    try:
        import base
        
        # Test enums
        if not hasattr(base, 'ThreadState') or not hasattr(base, 'FrameQuality'):
            return print_status("base: Missing required enums", False)
            
        # Test FrameData class
        if not hasattr(base, 'FrameData'):
            return print_status("base: Missing FrameData class", False)
            
        # Test BaseThread class
        if not hasattr(base, 'BaseThread'):
            return print_status("base: Missing BaseThread class", False)
            
        # Test BaseDetector class
        if not hasattr(base, 'BaseDetector'):
            return print_status("base: Missing BaseDetector class", False)
            
        # Test BaseFrameProcessor class
        if not hasattr(base, 'BaseFrameProcessor'):
            return print_status("base: Missing BaseFrameProcessor class", False)
            
        # Test utility functions
        if not hasattr(base, 'load_config'):
            return print_status("base: Missing load_config function", False)
            
        # Create a test frame data object
        try:
            import numpy as np
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_data = base.FrameData(test_frame, 1, time.time())
            frame_data.add_metadata("test_key", "test_value")
            
            if frame_data.metadata.get("test_key") != "test_value":
                return print_status("base: FrameData metadata functionality failed", False)
                
        except Exception as e:
            return print_status(f"base: FrameData test failed - {e}", False)
            
        return print_status("base: All tests passed")
        
    except Exception as e:
        return print_status(f"base: Test failed - {e}", False)

def test_camera_module() -> bool:
    """Test functionality of camera.py module"""
    logger.info("Testing camera module...")
    
    try:
        import camera
        
        # Test enums and classes
        if not hasattr(camera, 'CameraStatus'):
            return print_status("camera: Missing CameraStatus enum", False)
            
        if not hasattr(camera, 'CameraConfig'):
            return print_status("camera: Missing CameraConfig class", False)
            
        if not hasattr(camera, 'Camera'):
            return print_status("camera: Missing Camera class", False)
            
        if not hasattr(camera, 'CameraManager'):
            return print_status("camera: Missing CameraManager class", False)
            
        # Test camera config creation
        try:
            config = camera.CameraConfig(
                width=640,
                height=480,
                fps=30,
                rotation=0
            )
            if config.width != 640 or config.height != 480:
                return print_status("camera: CameraConfig creation failed", False)
                
        except Exception as e:
            return print_status(f"camera: CameraConfig test failed - {e}", False)
            
        # Test camera manager (no actual camera initialization)
        try:
            manager = camera.CameraManager()
            if not hasattr(manager, 'get_camera_list'):
                return print_status("camera: CameraManager missing methods", False)
                
        except Exception as e:
            return print_status(f"camera: CameraManager test failed - {e}", False)
            
        return print_status("camera: All tests passed")
        
    except Exception as e:
        return print_status(f"camera: Test failed - {e}", False)

def test_yolo_ncnn_module() -> bool:
    """Test functionality of yolo_ncnn.py module"""
    logger.info("Testing YOLO NCNN module...")
    
    try:
        import yolo_ncnn
        
        # Test if the module has the required class
        if not hasattr(yolo_ncnn, 'YoloNcnnDetector'):
            return print_status("yolo_ncnn: Missing YoloNcnnDetector class", False)
            
        # Check if the class inherits from BaseDetector
        import base
        detector_class = getattr(yolo_ncnn, 'YoloNcnnDetector')
        if not issubclass(detector_class, base.BaseDetector):
            return print_status("yolo_ncnn: YoloNcnnDetector doesn't inherit from BaseDetector", False)
            
        # Test creating detector instance (without loading model)
        try:
            detector = yolo_ncnn.YoloNcnnDetector()
            if not hasattr(detector, 'load_model') or not hasattr(detector, 'detect'):
                return print_status("yolo_ncnn: Detector missing required methods", False)
                
        except Exception as e:
            return print_status(f"yolo_ncnn: Detector creation failed - {e}", False)
            
        # Check if model files exist (don't attempt to load)
        model_files_exist = False
        model_name = 'yolov8n'
        param_file = f"models/{model_name}.param"
        bin_file = f"models/{model_name}.bin"
        if os.path.exists(param_file) and os.path.exists(bin_file):
            model_files_exist = True
            print_status(f"yolo_ncnn: Found model files for {model_name}")
                
        if not model_files_exist:
            print_status("yolo_ncnn: No model files found - convert_model.py may need to be run", False)
            
        return print_status("yolo_ncnn: Basic tests passed")
        
    except Exception as e:
        return print_status(f"yolo_ncnn: Test failed - {e}", False)

def test_thread_classes() -> bool:
    """Test functionality of thread classes"""
    logger.info("Testing thread classes...")
    
    try:
        # Import required modules and classes
        import base
        import capture_thread
        import inference_thread
        import quality_thread
        import save_thread
        from base import BaseThread
        from capture_thread import CaptureThread
        from inference_thread import InferenceThread
        from quality_thread import QualityThread
        from save_thread import SaveThread
        
        # Check if all thread classes inherit from BaseThread
        thread_classes = [CaptureThread, InferenceThread, QualityThread, SaveThread]
        for thread_class in thread_classes:
            if not issubclass(thread_class, BaseThread):
                return print_status(f"{thread_class.__name__} doesn't inherit from BaseThread", False)
                
        # Check for essential methods on each thread
        thread_classes_names = ["CaptureThread", "InferenceThread", "QualityThread", "SaveThread"]
        thread_modules = [capture_thread, inference_thread, quality_thread, save_thread]
        
        for name, module in zip(thread_classes_names, thread_modules):
            # Check for parameter classes
            if name in ["InferenceThread", "QualityThread", "SaveThread"]:
                param_class_name = f"{name.replace('Thread', '')}Parameters"
                if not hasattr(module, param_class_name):
                    print_status(f"{name}: Missing {param_class_name} class", False)
                
        return print_status("Thread classes: All inheritance checks passed")
        
    except Exception as e:
        return print_status(f"Thread classes: Test failed - {e}", False)

def test_thread_manager() -> bool:
    """Test thread manager functionality"""
    logger.info("Testing thread manager...")
    
    try:
        from thread_manager import ThreadManager, SystemState
        
        # Test if ThreadManager and SystemState exist
        if not SystemState:
            return print_status("thread_manager: Missing SystemState enum", False)
            
        # Check if thread manager has essential methods
        essential_methods = [
            'initialize_components', 'start', 'stop', 'pause', 'resume', 'get_stats'
        ]
        
        for method in essential_methods:
            if not hasattr(ThreadManager, method):
                return print_status(f"thread_manager: Missing {method} method", False)
                
        return print_status("thread_manager: Class structure checks passed")
        
    except Exception as e:
        return print_status(f"thread_manager: Test failed - {e}", False)

def test_config_yaml() -> bool:
    """Test if config.yaml is valid"""
    logger.info("Testing config.yaml...")
    
    try:
        import yaml
        
        if not os.path.exists('config.yaml'):
            return print_status("config.yaml not found", False)
            
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Check for essential sections
        essential_sections = [
            'system', 'camera', 'detector', 'image_quality', 
            'frame_selector', 'metrics', 'image_storage'
        ]
        
        for section in essential_sections:
            if section not in config:
                return print_status(f"config.yaml: Missing '{section}' section", False)
                
        return print_status("config.yaml: Valid YAML with all required sections")
        
    except yaml.YAMLError as e:
        return print_status(f"config.yaml: Invalid YAML - {e}", False)
    except Exception as e:
        return print_status(f"config.yaml: Test failed - {e}", False)

def test_bash_scripts() -> bool:
    """Test if bash scripts are valid"""
    logger.info("Testing bash scripts...")
    
    bash_scripts = [
        'venv_install.sh',
        'global_install.sh',
        'run.sh'
    ]
    
    all_valid = True
    for script in bash_scripts:
        if not os.path.exists(script):
            print_status(f"{script}: Not found", False)
            all_valid = False
            continue
            
        try:
            # Use bash -n to check syntax without executing
            result = subprocess.run(['bash', '-n', script], 
                                    capture_output=True, 
                                    text=True)
            
            if result.returncode == 0:
                print_status(f"{script}: Valid bash syntax")
            else:
                print_status(f"{script}: Invalid bash syntax - {result.stderr}", False)
                all_valid = False
                
        except Exception as e:
            print_status(f"{script}: Test failed - {e}", False)
            all_valid = False
            
    return all_valid

def test_assistance_module() -> bool:
    """Test functionality of assistance.py module"""
    logger.info("Testing assistance module...")
    
    try:
        import assistance
        
        # Test essential functions
        essential_functions = [
            'setup_logging', 'get_timestamp', 'get_formatted_timestamp',
            'ensure_directory', 'sanitize_filename', 'generate_unique_filename',
            'get_file_size', 'format_size', 'get_system_info'
        ]
        
        for func_name in essential_functions:
            if not hasattr(assistance, func_name):
                return print_status(f"assistance: Missing {func_name} function", False)
                
        # Test filename sanitization
        try:
            clean_name = assistance.sanitize_filename('test<>:"/\\|?*name.jpg')
            if clean_name != 'test_name.jpg':
                return print_status(f"assistance: sanitize_filename produced unexpected result: {clean_name}", False)
                
        except Exception as e:
            return print_status(f"assistance: sanitize_filename test failed - {e}", False)
            
        # Test unique filename generation
        try:
            unique_name = assistance.generate_unique_filename(prefix="test")
            if not unique_name.startswith("test_") or not unique_name.endswith(".jpg"):
                return print_status(f"assistance: generate_unique_filename failed: {unique_name}", False)
                
        except Exception as e:
            return print_status(f"assistance: generate_unique_filename test failed - {e}", False)
            
        # Test system info
        try:
            system_info = assistance.get_system_info()
            if not isinstance(system_info, dict) or "platform" not in system_info:
                return print_status("assistance: get_system_info failed", False)
                
        except Exception as e:
            return print_status(f"assistance: get_system_info test failed - {e}", False)
            
        return print_status("assistance: All tests passed")
        
    except Exception as e:
        return print_status(f"assistance: Test failed - {e}", False)

def test_model_converter() -> bool:
    """Test model converter script"""
    logger.info("Testing model converter script...")
    
    try:
        # Check if script exists
        if not os.path.exists('convert_model.py'):
            return print_status("convert_model.py not found", False)
            
        # Check script content without executing
        with open('convert_model.py', 'r') as f:
            content = f.read()
            
        # Check for essential functions and classes
        required_strings = [
            "def convert_to_onnx",
            "def convert_onnx_to_ncnn",
            "def check_requirements",
            "if __name__ == \"__main__\""
        ]
        
        for required in required_strings:
            if required not in content:
                return print_status(f"convert_model.py: Missing {required}", False)
        
        # Check for converted model files
        model_name = 'yolov8n'
        param_file = f"models/{model_name}.param"
        bin_file = f"models/{model_name}.bin"
        onnx_file = f"models/{model_name}.onnx"
        
        if os.path.exists(param_file) and os.path.exists(bin_file) and os.path.exists(onnx_file):
            print_status(f"Model conversion verified: Found {model_name} NCNN and ONNX files")
        else:
            print_status(f"Model files not found. Run convert_model.py first", False)
                
        return print_status("convert_model.py: Contains all required functions")
        
    except Exception as e:
        return print_status(f"convert_model.py: Test failed - {e}", False)

def test_main_script() -> bool:
    """Test main application script"""
    logger.info("Testing main application script...")
    
    try:
        # Check if script exists
        if not os.path.exists('main.py'):
            return print_status("main.py not found", False)
            
        # Check script content without executing
        with open('main.py', 'r') as f:
            content = f.read()
            
        # Check for essential functions and main entry point
        required_strings = [
            "def main",
            "def setup_logging",
            "def load_system_config",
            "if __name__ == \"__main__\"",
            "ThreadManager",
            "get_metrics_instance",
            "get_storage_instance"
        ]
        
        for required in required_strings:
            if required not in content:
                return print_status(f"main.py: Missing {required}", False)
                
        return print_status("main.py: Contains all required functions")
        
    except Exception as e:
        return print_status(f"main.py: Test failed - {e}", False)

def verify_projects_directory() -> bool:
    """Verify if project directory structure is correct"""
    logger.info("Verifying project directory structure...")
    
    # Check required directories
    required_dirs = ['models', 'images']
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print_status(f"Created directory: {directory}")
            except Exception as e:
                return print_status(f"Failed to create {directory}: {e}", False)
        else:
            print_status(f"Directory exists: {directory}")
            
    return True

def test_ncnn_dependencies() -> bool:
    """Test NCNN dependencies"""
    logger.info("Testing NCNN dependencies...")
    
    try:
        # Check for onnx2ncnn executable
        onnx2ncnn_path = shutil.which('onnx2ncnn')
        if not onnx2ncnn_path:
            return print_status("onnx2ncnn executable not found in PATH", False)
            
        print_status(f"onnx2ncnn found at: {onnx2ncnn_path}")
        
        # Check for NCNN Python bindings
        try:
            import ncnn
            print_status(f"NCNN Python bindings installed: {ncnn.__file__}")
        except ImportError:
            return print_status("NCNN Python bindings not installed", False)
        
        # Check for converted model files
        model_name = 'yolov8n'
        param_file = f"models/{model_name}.param"
        bin_file = f"models/{model_name}.bin"
        
        if os.path.exists(param_file) and os.path.exists(bin_file):
            print_status(f"Found required NCNN model files: {param_file} and {bin_file}")
        else:
            print_status(f"NCNN model files not found. Run convert_model.py first", False)
            
        return True
        
    except Exception as e:
        return print_status(f"Error checking NCNN dependencies: {e}", False)

def run_full_verification() -> Dict[str, bool]:
    """Run all verification tests"""
    logger.info("\n===== Starting Full Verification =====\n")
    
    # Results dictionary
    results = {}
    
    # Run verification tests
    tests = [
        ("imports", check_imports),
        ("scripts_executable", check_scripts_executable),
        ("base_module", test_base_module),
        ("camera_module", test_camera_module),
        ("yolo_ncnn_module", test_yolo_ncnn_module),
        ("thread_classes", test_thread_classes),
        ("thread_manager", test_thread_manager),
        ("config_yaml", test_config_yaml),
        ("bash_scripts", test_bash_scripts),
        ("assistance_module", test_assistance_module),
        ("model_converter", test_model_converter),
        ("main_script", test_main_script),
        ("project_directory", verify_projects_directory)
    ]
    
    # Run each test and store result
    for test_name, test_func in tests:
        logger.info(f"\n----- Testing {test_name} -----")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Error running {test_name} test: {e}")
            results[test_name] = False
            
    # Try NCNN dependency test only if appropriate
    try:
        logger.info("\n----- Testing NCNN dependencies -----")
        results["ncnn_dependencies"] = test_ncnn_dependencies()
    except:
        results["ncnn_dependencies"] = False
    
    # Print summary
    logger.info("\n===== Verification Summary =====")
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        logger.info(f"{test_name}: {status}")
    
    if all_passed:
        logger.info("\n✅ All tests passed! The system is ready to run.")
    else:
        logger.warning("\n⚠️ Some tests failed. Please address the issues before running the application.")
    
    return results

if __name__ == "__main__":
    # Allow running specific tests from command line
    parser = argparse.ArgumentParser(description="Verify WMSV4AI components")
    parser.add_argument("--test", "-t", help="Run specific test (leave empty for all tests)")
    args = parser.parse_args()
    
    if args.test:
        # Run specific test if specified
        test_funcs = {
            "imports": check_imports,
            "scripts": check_scripts_executable,
            "base": test_base_module,
            "camera": test_camera_module,
            "yolo": test_yolo_ncnn_module,
            "threads": test_thread_classes,
            "manager": test_thread_manager,
            "config": test_config_yaml,
            "bash": test_bash_scripts,
            "assistance": test_assistance_module,
            "converter": test_model_converter,
            "main": test_main_script,
            "directory": verify_projects_directory,
            "ncnn": test_ncnn_dependencies
        }
        
        if args.test in test_funcs:
            test_funcs[args.test]()
        else:
            logger.error(f"Unknown test: {args.test}")
            logger.info(f"Available tests: {', '.join(test_funcs.keys())}")
    else:
        # Run all tests
        run_full_verification()
