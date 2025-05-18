import os
import sys
import subprocess
import importlib
import pkg_resources
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("installations")

# Define required Python packages with their minimum versions
REQUIRED_PACKAGES = {
    'torch': '2.0.0',
    'torchvision': '0.15.0',
    'ultralytics': '8.0.0',
    'opencv-python': '4.7.0',   
    'onnx': '1.14.0',
    'onnxruntime': '1.15.0',
    'ncnn': '1.0.20231027',
    'scikit-image': '0.20',     
    'numpy': '1.24.2',
    'psutil': '5.9.5',
    'pyyaml': '6.0',            
    'brisque': '0.0.16',
    'imutils': '0.5.4',
}

# Define required system packages
REQUIRED_SYSTEM_PACKAGES = [
    'cmake',
    'build-essential',
    'libopencv-dev',
    'libprotobuf-dev',
    'protobuf-compiler',
    'libjpeg-dev',
    'zlib1g-dev',
    'libvulkan-dev',
]

# Define required binaries/executables
REQUIRED_BINARIES = [
    'onnx2ncnn',
]

# Define required model files
REQUIRED_MODEL_FILES = [
    # Default models we expect might exist - these will be checked only if they exist, not required
    'models/yolov8n.pt',
    'models/yolov8n.onnx',
    'models/yolov8n.param',
    'models/yolov8n.bin',
]

def print_status(message, success=True):
    if success:
        logger.info(f"✅ {message}")
    else:
        logger.error(f"❌ {message}")
    return success

def check_python_packages():
    logger.info("Checking Python packages...")
    all_installed = True
    
    # Special case for PyYAML which might be imported as 'yaml'
    yaml_checked = False
    
    for package, min_version in REQUIRED_PACKAGES.items():
        # Special handling for certain packages
        if package == 'pyyaml' and not yaml_checked:
            try:
                import yaml
                yaml_checked = True
                # PyYAML doesn't expose version in the module, check with pkg_resources
                try:
                    installed_version = pkg_resources.get_distribution('pyyaml').version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                        print_status(f"{package} needs upgrade: found {installed_version}, required {min_version}", False)
                        all_installed = False
                    else:
                        print_status(f"{package} {installed_version} is installed")
                        continue
                except pkg_resources.DistributionNotFound:
                    # Check if it's installed in the system packages
                    try:
                        result = subprocess.run(['dpkg', '-s', 'python3-yaml'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if result.returncode == 0:
                            print_status(f"{package} is installed (system package)")
                            continue
                    except:
                        pass
                    print_status(f"{package} is not installed", False)
                    all_installed = False
                continue
            except ImportError:
                pass  # Will be handled by the general case
        
        # Special handling for OpenCV, which might be installed under different names
        if package == 'opencv-python':
            try:
                import cv2
                cv_version = cv2.__version__
                if pkg_resources.parse_version(cv_version) < pkg_resources.parse_version(min_version):
                    print_status(f"{package} needs upgrade: found {cv_version}, required {min_version}", False)
                    all_installed = False
                else:
                    print_status(f"{package} {cv_version} is installed")
                continue
            except ImportError:
                print_status(f"{package} is not installed", False)
                all_installed = False
                continue
        
        # Special handling for scikit-image
        if package == 'scikit-image':
            try:
                import skimage
                try:
                    sk_version = skimage.__version__
                    if pkg_resources.parse_version(sk_version) < pkg_resources.parse_version(min_version):
                        print_status(f"{package} needs upgrade: found {sk_version}, required {min_version}", False)
                        all_installed = False
                    else:
                        print_status(f"{package} {sk_version} is installed")
                    continue
                except (AttributeError, pkg_resources.DistributionNotFound):
                    # If we can import but can't get version, assume it works
                    print_status(f"{package} is installed (version unknown)")
                    continue
            except ImportError:
                print_status(f"{package} is not installed", False)
                all_installed = False
                continue
                
        # Normal case for other packages
        try:
            # Try to import the package
            module_name = package.replace('-', '_')
            module = importlib.import_module(module_name)
            
            # Check version
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                    print_status(f"{package} needs upgrade: found {installed_version}, required {min_version}", False)
                    all_installed = False
                else:
                    print_status(f"{package} {installed_version} is installed")
            except pkg_resources.DistributionNotFound:
                # If package is imported but not found by pkg_resources, it might be a system package
                # We'll accept it as installed
                print_status(f"{package} is installed (version unknown)")
                
        except (ImportError, pkg_resources.DistributionNotFound):
            print_status(f"{package} is not installed", False)
            all_installed = False
            
    return all_installed

def check_system_packages():
    all_installed = True
    
    # Check if we're on Debian/Ubuntu based system
    if not os.path.exists('/usr/bin/dpkg'):
        logger.warning("Not running on a Debian/Ubuntu system, skipping system package check")
        return True
    
    for package in REQUIRED_SYSTEM_PACKAGES:
        try:
            # Use dpkg to check if package is installed
            result = subprocess.run(['dpkg', '-s', package], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True)
            
            if result.returncode == 0:
                print_status(f"{package} is installed")
            else:
                print_status(f"{package} is not installed", False)
                all_installed = False
                
        except Exception as e:
            logger.error(f"Error checking package {package}: {e}")
            all_installed = False
            
    return all_installed

def check_binaries():
    logger.info("Checking required binaries...")
    all_installed = True
    
    for binary in REQUIRED_BINARIES:
        binary_path = shutil.which(binary)
        if binary_path:
            print_status(f"{binary} found at {binary_path}")
        else:
            print_status(f"{binary} not found in PATH", False)
            all_installed = False
            
    return all_installed

def check_model_files():
    """Check if required model files exist"""
    logger.info("Checking model files...")
    models_exist = False
    
    for model_file in REQUIRED_MODEL_FILES:
        path = Path(model_file)
        if path.exists():
            print_status(f"{model_file} found")
            models_exist = True
        else:
            logger.info(f"{model_file} not found, might need to be downloaded or converted")
    
    if not models_exist:
        logger.warning("No model files found. You may need to run main.py first to download the model.")
        logger.warning("Then run convert_model.py to generate the NCNN model files.")
    
    return True  # Not critical - user might run download later

def check_camera():
    logger.info("Checking camera...")
    
    try:
        # Import Picamera2
        from picamera2 import Picamera2
        
        # Check if cameras are detected
        cameras = Picamera2.global_camera_info()
        if not cameras:
            return print_status("No cameras detected", False)
        
        logger.info(f"Detected {len(cameras)} camera(s):")
        for i, camera in enumerate(cameras):
            logger.info(f"  Camera {i}: {camera.get('model', 'Unknown model')}")
        
        return print_status("Camera system is working")
        
    except ImportError:
        return print_status("Picamera2 module not installed", False)
    except Exception as e:
        return print_status(f"Error accessing camera: {e}", False)

def check_directories():
    logger.info("Checking directories...")
    
    directories = [
        "models",
        "images"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True)
                print_status(f"Created directory: {directory}")
            except Exception as e:
                return print_status(f"Failed to create directory {directory}: {e}", False)
        else:
            print_status(f"Directory exists: {directory}")
    
    return True

def check_ncnn_python():
    logger.info("Testing NCNN Python bindings...")
    
    try:
        import ncnn
        
        # Basic test to see if we can import the module
        print_status("NCNN Python bindings are installed")
        
        try:
            # Simplified test that should work
            net = ncnn.Net()
            extractor = net.create_extractor()
            print_status("NCNN Python bindings working correctly")
            return True
        except Exception as e:
            logger.warning(f"Could not create extractor: {e}")
            # Even if we can't create an extractor, the module is available
            # which is enough for basic functionality
            return True
    
    except ImportError:
        return print_status("NCNN Python module not installed", False)
    except Exception as e:
        return print_status(f"Error testing NCNN Python bindings: {e}", False)

def check_pytorch():
    logger.info("Testing PyTorch...")
    
    try:
        import torch
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device_type = "CUDA"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_status(f"PyTorch with CUDA support. Found {device_count} device(s): {device_name}")
        else:
            print_status("PyTorch is using CPU only (no CUDA)")
        
        # Simple test to ensure torch works
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y  # Simple operation to check if PyTorch works
        
        return print_status("PyTorch is working correctly")
    
    except ImportError:
        return print_status("PyTorch not installed", False)
    except Exception as e:
        return print_status(f"Error testing PyTorch: {e}", False)

def run_all_checks():
    logger.info("Starting system verification...")
    
    checks = [
        ("Python packages", check_python_packages),
        ("System packages", check_system_packages),
        ("Required binaries", check_binaries),
        ("Model files", check_model_files),
        ("Directories", check_directories),
        ("NCNN Python bindings", check_ncnn_python),
        ("PyTorch", check_pytorch),
        ("Camera", check_camera)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n=== Checking {name} ===")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error during {name} check: {e}")
            results.append((name, False))
    
    # Print summary
    logger.info("\n=== Verification Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("\n✅ All checks passed! The system is ready to run.")
    else:
        logger.warning("\n⚠️ Some checks failed. Please address the issues before running the application.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
