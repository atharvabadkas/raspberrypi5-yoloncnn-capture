# Global installation script for Raspberry Pi OS
# Run with: sudo bash global_install.sh

# Exit on error
set -e

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root (use sudo)."
    exit 1
fi

# Get the real user who invoked sudo
if [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$(whoami)"
fi

# Get the real user's home directory
USER_HOME=$(eval echo ~$REAL_USER)

echo "===== Installing System Dependencies for NCNN and PyTorch ====="

# Update package repositories
echo "Updating package repositories..."
apt update
apt upgrade -y

# Check Python version
echo "Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version)
    echo "Found $PYTHON_VERSION"
else
    echo "Python 3.11 not found. Installing Python 3.11..."
    # Install Python 3.11 (method may vary based on distro)
    # For Debian/Ubuntu based systems:
    apt install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt update
    apt install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils
    
    # Create symlinks if they don't exist
    if [ ! -e /usr/bin/python3.11 ]; then
        ln -s $(which python3.11) /usr/bin/python3.11
    fi
fi

# Install essential build tools
echo "Installing build tools and development libraries..."
apt install -y cmake build-essential git
apt install -y libopencv-dev libopencv-core-dev
apt install -y libprotobuf-dev protobuf-compiler
apt install -y python3-dev python3-pip python3-venv python3-setuptools python3-wheel
apt install -y python3-numpy

# Install Raspberry Pi specific packages
echo "Installing Raspberry Pi specific packages..."
apt install -y raspberrypi-kernel-headers libraspberrypi-dev

# Install camera support
echo "Installing camera support..."
apt install -y python3-picamera2

# Install additional dependencies for image quality assessment
echo "Installing additional dependencies for image quality assessment..."
apt install -y libjpeg-dev zlib1g-dev libffi-dev libblas-dev liblapack-dev

# Build and install NCNN tools
echo "Building NCNN tools (onnx2ncnn)..."
WORK_DIR=$(pwd)
mkdir -p ~/ncnn_build
cd ~/ncnn_build

# Check if NCNN already cloned
if [ ! -d "ncnn" ]; then
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    git checkout 20231027  # Use a newer version for YOLOv11 compatibility
    git submodule update --init --recursive  # Initialize all submodules
else
    cd ncnn
    git fetch
    git checkout 20231027
    git submodule update --init --recursive  # Ensure submodules are up to date
fi

# Build NCNN with optimizations for Raspberry Pi (CPU only)
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_PYTHON=OFF \
      -DNCNN_BUILD_TOOLS=ON \
      -DNCNN_ARM_NEON=ON \
      -DNCNN_ARM_FP16=ON \
      -DNCNN_SYSTEM_GLSLANG=OFF \
      -DNCNN_OPENMP=ON \
      ..

# Build onnx2ncnn and other tools
echo "Building NCNN tools..."
make -j4 onnx2ncnn

# Install onnx2ncnn to system path
echo "Installing onnx2ncnn to system path..."
cp tools/onnx/onnx2ncnn /usr/local/bin/
chmod +x /usr/local/bin/onnx2ncnn

# Return to original directory
cd "$WORK_DIR"

# Configure camera
echo "Enabling camera interface..."
if command -v raspi-config > /dev/null; then
    # Use raspi-config nonint command to enable camera
    raspi-config nonint do_camera 0
    echo "Camera interface enabled through raspi-config"
else
    echo "WARNING: raspi-config not found. Please enable camera manually:"
    echo "sudo raspi-config > Interface Options > Camera > Enable"
fi

# Create directories for application data
echo "Creating application directories..."
PROJECT_DIR="${USER_HOME}/Desktop/WMSV4AI"
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/images"
chown -R ${REAL_USER}:${REAL_USER} "${PROJECT_DIR}"

echo -e "\n===== Global Installation Complete ====="
echo "A system reboot is recommended to apply all changes."
echo "After reboot, run the Python environment setup script:"
echo "bash venv_install.sh"
echo ""
echo "To reboot now, run: sudo reboot"