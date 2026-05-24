# VisoMaster RunPod Quick Command Reference

## 🚀 Quick Start

```bash
# 1. Connect to RunPod
ssh ejfcfnjvk5nklb-644112ef@ssh.runpod.io -i ~/.ssh/id_ed25519

# 2. Clone with submodules (recommended)
git clone --recurse-submodules https://github.com/crazidev/VisoMaster.git
cd VisoMaster

# OR if you already cloned without submodules:
git clone https://github.com/crazidev/VisoMaster.git
cd VisoMaster
git submodule update --init --recursive

# 3. Run automated setup
bash scripts/runpod_quick_setup.sh

# 4. Run VisoMaster
python3 main.py
```

**⚠️ Important:** Always initialize submodules after cloning or pulling!

---

## 📋 Essential Commands

### Connection
```bash
# Connect to RunPod
ssh <your-pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519

# Connect with X11 forwarding (for GUI)
ssh -X <your-pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### GPU Monitoring
```bash
# Check GPU status
nvidia-smi

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Detailed GPU monitoring
nvidia-smi dmon -s u

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### System Monitoring
```bash
# Check CPU usage
htop

# Check RAM usage
free -h

# Check disk space
df -h

# Check running processes
ps aux | grep python
```

### VisoMaster Operations
```bash
# Navigate to VisoMaster
cd /workspace/VisoMaster

# Run VisoMaster
python3 main.py

# Run in background
nohup python3 main.py > visomaster.log 2>&1 &

# Check if running
ps aux | grep main.py

# Kill VisoMaster
pkill -f main.py
```

### CUDA Verification
```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Full CUDA info
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"
```

### File Transfer
```bash
# Upload file to RunPod (from local machine)
scp -i ~/.ssh/id_ed25519 video.mp4 <pod-id>@ssh.runpod.io:/workspace/VisoMaster/

# Download file from RunPod (from local machine)
scp -i ~/.ssh/id_ed25519 <pod-id>@ssh.runpod.io:/workspace/VisoMaster/output.mp4 ./

# Upload directory
scp -r -i ~/.ssh/id_ed25519 videos/ <pod-id>@ssh.runpod.io:/workspace/VisoMaster/

# Download directory
scp -r -i ~/.ssh/id_ed25519 <pod-id>@ssh.runpod.io:/workspace/VisoMaster/output/ ./
```

### Troubleshooting
```bash
# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Check Python packages
pip list | grep -E "torch|onnx|tensor"

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Check logs
tail -f visomaster.log

# Check system logs
journalctl -xe

# Test ONNX Runtime
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Test TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### Git Operations
```bash
# Update VisoMaster
cd /workspace/VisoMaster
git pull

# Initialize/update submodules (IMPORTANT after pull!)
git submodule update --init --recursive

# Check current version
git log -1 --oneline

# Reset to latest
git reset --hard origin/main
git pull
git submodule update --init --recursive
```

### Fix Missing Submodules
```bash
# If packages/streamrelay is empty after git pull:
git submodule update --init --recursive

# Verify it worked
ls -la packages/streamrelay/

# Should see: __init__.py, server.py, reader.py, etc.
```

### Cleanup
```bash
# Clear TensorRT cache
rm -rf tensorrt-engines/

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear pip cache
pip cache purge

# Free up disk space
apt-get clean
apt-get autoremove -y
```

---

## 🎮 VNC Setup (for GUI)

```bash
# Install VNC server
apt-get install -y x11vnc xvfb

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Start VNC server
x11vnc -display :99 -forever -shared -rfbport 5900 -passwd yourpassword &

# In RunPod dashboard: Expose port 5900
# Connect with VNC client to: <pod-ip>:5900
```

---

## 📊 Performance Optimization

```bash
# Check optimal thread count
# Monitor GPU utilization while processing
watch -n 1 nvidia-smi

# If GPU < 95%: increase threads
# If GPU = 99%: perfect
# If OOM errors: reduce threads

# Set environment variables for better performance
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

---

## 🔧 Common Issues & Fixes

### Issue: CUDA not available
```bash
# Check CUDA
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Issue: Out of memory
```bash
# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Reduce threads in VisoMaster settings
# Or restart the pod
```

### Issue: TensorRT errors
```bash
# Clear TensorRT cache
rm -rf tensorrt-engines/

# Reinstall TensorRT
pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com
```

### Issue: GUI not working
```bash
# Use VNC (see VNC Setup section)
# Or set offscreen rendering
export QT_QPA_PLATFORM=offscreen
```

---

## 💾 Backup & Restore

```bash
# Backup models (to avoid re-downloading)
tar -czf models_backup.tar.gz model_assets/

# Backup settings
cp -r .kiro/ .kiro_backup/

# Restore models
tar -xzf models_backup.tar.gz

# Restore settings
cp -r .kiro_backup/ .kiro/
```

---

## 🔒 Security

```bash
# Change VNC password
x11vnc -storepasswd

# Check open ports
netstat -tuln

# Firewall (if needed)
ufw allow 5900/tcp  # VNC
ufw allow 8888/tcp  # Jupyter
ufw enable
```

---

## 📈 Benchmarking

```bash
# Time a video processing
time python3 -c "
from app.processors.video_processor import VideoProcessor
# Your processing code here
"

# Monitor during processing
# Terminal 1:
python3 main.py

# Terminal 2:
watch -n 1 nvidia-smi
```

---

## 🆘 Emergency Commands

```bash
# Kill all Python processes
pkill -9 python3

# Force clear GPU memory
nvidia-smi --gpu-reset

# Restart pod (from RunPod dashboard)
# Or reboot system
sudo reboot
```

---

## 📚 Useful Aliases

Add to `~/.bashrc`:

```bash
# VisoMaster aliases
alias vm='cd /workspace/VisoMaster'
alias vmrun='cd /workspace/VisoMaster && python3 main.py'
alias vmgpu='watch -n 1 nvidia-smi'
alias vmlog='tail -f /workspace/VisoMaster/visomaster.log'
alias vmclear='python3 -c "import torch; torch.cuda.empty_cache()"'

# Reload aliases
source ~/.bashrc
```

---

## 🎯 Quick Tests

```bash
# Test 1: GPU availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✓ GPU OK')"

# Test 2: ONNX Runtime
python3 -c "import onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(), 'CUDA EP not available!'; print('✓ ONNX Runtime OK')"

# Test 3: TensorRT
python3 -c "import tensorrt; print(f'✓ TensorRT {tensorrt.__version__} OK')"

# Test 4: Models exist
ls -lh model_assets/ | head -10

# Run all tests
python3 -c "
import torch
import onnxruntime
import tensorrt
print('✓ All imports successful')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ ONNX: {\"CUDAExecutionProvider\" in onnxruntime.get_available_providers()}')
print(f'✓ TensorRT: {tensorrt.__version__}')
"
```

---

**For detailed setup instructions, see: RUNPOD_SETUP.md**
