#!/bin/bash
# VisoMaster RunPod Quick Setup Script
# One-command installation for RunPod instances

set -e

echo "============================================"
echo "  VisoMaster RunPod Quick Setup"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on RunPod
if [ -d "/workspace" ]; then
    print_status "RunPod environment detected"
    WORKSPACE="/workspace/VisoMaster"
else
    print_warning "Not running on RunPod, using current directory"
    WORKSPACE="$(pwd)"
fi

# Step 1: Check GPU
echo ""
echo "Step 1/7: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    print_status "GPU detected: $GPU_NAME (${GPU_MEMORY}MB VRAM)"
    
    # Check if VRAM is sufficient
    if [ "$GPU_MEMORY" -lt 8000 ]; then
        print_error "Insufficient VRAM (need at least 8GB)"
        exit 1
    fi
else
    print_error "No NVIDIA GPU detected!"
    exit 1
fi

# Step 2: Update system packages
echo ""
echo "Step 2/7: Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null 2>&1
print_status "System packages updated"

# Step 3: Install system dependencies
echo ""
echo "Step 3/7: Installing system dependencies..."
apt-get install -y -qq \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    git \
    wget \
    unzip > /dev/null 2>&1

print_status "System dependencies installed"

# Step 4: Verify CUDA
echo ""
echo "Step 4/7: Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA version: $CUDA_VERSION"
else
    print_warning "nvcc not found, but nvidia-smi works (this is OK)"
fi

# Step 5: Install Python dependencies
echo ""
echo "Step 5/7: Installing Python dependencies..."
echo "    This may take 5-10 minutes..."

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements_cu124.txt 2>&1 | grep -E "(Successfully|Requirement already|ERROR)" || true

print_status "Python dependencies installed"

# Step 6: Download models
echo ""
echo "Step 6/7: Downloading AI models..."
echo "    This may take 10-15 minutes depending on connection..."

if [ -f "download_models.py" ]; then
    python3 download_models.py
    print_status "Models downloaded"
else
    print_error "download_models.py not found!"
    exit 1
fi

# Step 7: Download additional dependencies
echo ""
echo "Step 7/7: Downloading additional dependencies..."
mkdir -p dependencies

# Check if dependencies already exist
if [ -f "dependencies/rd64-uni-refined.pth" ]; then
    print_status "Dependencies already downloaded"
else
    print_warning "Downloading additional assets (this may take a few minutes)..."
    
    # Download LivePortrait models
    if [ ! -d "dependencies/liveportrait_onnx" ]; then
        wget -q --show-progress -P dependencies/ \
            https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0_dp/liveportrait_onnx.zip 2>&1 || \
            print_warning "Failed to download liveportrait_onnx.zip (optional)"
        
        if [ -f "dependencies/liveportrait_onnx.zip" ]; then
            unzip -q dependencies/liveportrait_onnx.zip -d dependencies/ 2>/dev/null || true
            rm dependencies/liveportrait_onnx.zip
        fi
    fi
    
    # Download CLIP model
    if [ ! -f "dependencies/rd64-uni-refined.pth" ]; then
        wget -q --show-progress -P dependencies/ \
            https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0_dp/rd64-uni-refined.pth 2>&1 || \
            print_warning "Failed to download rd64-uni-refined.pth (optional)"
    fi
    
    print_status "Additional dependencies downloaded"
fi

# Verify installation
echo ""
echo "============================================"
echo "  Verifying Installation"
echo "============================================"
echo ""

# Test CUDA availability
echo "Testing CUDA availability..."
python3 -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    sys.exit(0)
else:
    print('ERROR: CUDA not available!')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "CUDA verification passed"
else
    print_error "CUDA verification failed!"
    exit 1
fi

# Test ONNX Runtime
echo ""
echo "Testing ONNX Runtime..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Available providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('✓ CUDA Execution Provider available')
else:
    print('✗ CUDA Execution Provider NOT available')
" || print_warning "ONNX Runtime test failed (may still work)"

# Test TensorRT
echo ""
echo "Testing TensorRT..."
python3 -c "
try:
    import tensorrt as trt
    print(f'TensorRT version: {trt.__version__}')
    print('✓ TensorRT available')
except ImportError:
    print('✗ TensorRT not available (optional)')
" || print_warning "TensorRT not available (optional, but recommended)"

# Installation complete
echo ""
echo "============================================"
echo "  ✅ Installation Complete!"
echo "============================================"
echo ""
echo "Configuration Summary:"
echo "  GPU: $GPU_NAME"
echo "  VRAM: ${GPU_MEMORY}MB"
echo "  Location: $WORKSPACE"
echo ""
echo "Recommended Settings:"
echo "  Threads: 10-12 (for 16 vCPU)"
echo "  Provider: TensorRT"
echo "  Max DFM Models: 6-8"
echo ""
echo "To run VisoMaster:"
echo "  python3 main.py"
echo ""
echo "For GUI access, you'll need:"
echo "  - X11 forwarding: ssh -X ..."
echo "  - Or VNC server (see RUNPOD_SETUP.md)"
echo ""
echo "For more information:"
echo "  cat RUNPOD_SETUP.md"
echo ""
echo "============================================"
echo ""

# Create a quick start script
cat > run_visomaster.sh << 'EOF'
#!/bin/bash
# Quick start script for VisoMaster

# Set display for headless mode
export QT_QPA_PLATFORM=offscreen

# Run VisoMaster
python3 main.py "$@"
EOF

chmod +x run_visomaster.sh
print_status "Created run_visomaster.sh for easy launching"

echo ""
print_status "Setup complete! You can now run: python3 main.py"
echo ""
