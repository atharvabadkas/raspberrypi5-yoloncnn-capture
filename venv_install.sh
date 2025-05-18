#!/bin/bash
# Python virtual environment setup for NCNN and PyTorch
# Run as normal user (no sudo): bash venv_install.sh

# Exit on error
set -e

echo "===== Setting up Python Virtual Environment for NCNN and PyTorch ====="

# Define project directory
PROJECT_DIR=~/Desktop/WMSV4AI
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Check for Python 3.11.2
echo "Checking for Python 3.11..."
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version)
    echo "Found $PYTHON_VERSION"
    PYTHON_CMD=python3.11
else
    echo "Python 3.11 not found. Using system default Python:"
    PYTHON_VERSION=$(python3 --version)
    echo "Found $PYTHON_VERSION"
    PYTHON_CMD=python3
fi

# Create directories for models and images if they don't exist
mkdir -p models
mkdir -p images

# Create virtual environment with system site packages
echo "Creating Python virtual environment with system site packages..."
$PYTHON_CMD -m venv raspienv --system-site-packages

# Activate the environment
source raspienv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Fix for opencv dependencies
echo "Installing OpenCV dependencies..."
pip install numpy

# Install PyTorch (CPU only)
echo "Installing PyTorch (CPU version)..."
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install YOLOv11 and related packages
echo "Installing Ultralytics YOLOv11..."
pip install ultralytics==8.0.0

# Install NCNN and conversion dependencies
echo "Installing NCNN and conversion dependencies..."
pip install onnx==1.14.0 onnxruntime==1.15.0
pip install opencv-python==4.7.0.72

# Make sure we have correct NCNN version
echo "Installing NCNN with Python bindings..."
pip uninstall -y ncnn  # Remove any existing installation
pip install ncnn==1.0.20231027

# Install OpenCV and other vision libraries
echo "Installing OpenCV and vision libraries..."
pip install scikit-image==0.20.0

# Install image quality assessment libraries
echo "Installing image quality assessment libraries..."
pip install brisque==0.0.16
pip install imutils==0.5.4

# Install system monitoring tools
echo "Installing system monitoring tools..."
pip install psutil==5.9.5

# Install utilities
echo "Installing utilities..."
pip install numpy==1.24.2 pyyaml==6.0 tqdm==4.65.0

# Set execution permission for Python scripts
echo "Setting up project files..."
chmod +x convert_model.py
if [ -f "update_model.py" ]; then
    chmod +x update_model.py
fi
if [ -f "main.py" ]; then
    chmod +x main.py
fi
if [ -f "installations.py" ]; then
    chmod +x installations.py
fi
if [ -f "verifications.py" ]; then
    chmod +x verifications.py
fi

# Create run script for convenience
echo "Creating convenience run script..."
cat > run.sh << 'EOF'
#!/bin/bash
cd ~/Desktop/WMSV4AI
source raspienv/bin/activate
python3 "$@"
EOF

# Make scripts executable
chmod +x run.sh

echo -e "\n===== Python Environment Setup Complete ====="
echo "To use the environment:"
echo "1. Activate: source ~/Desktop/WMSV4AI/raspienv/bin/activate"
echo "2. Or use the run script: ./run.sh [script_name]"
echo ""
echo "Next steps:"
echo "1. Run the verification script: ./run.sh installations.py"
echo "2. Download the model: ./run.sh main.py (once implemented)"
echo "3. Convert the model: ./run.sh convert_model.py"
echo "4. Run the main application: ./run.sh main.py"