# VisoMaster RunPod Setup Guide

Complete guide to install and run VisoMaster on RunPod with A40 GPU.

---

## 🚀 Quick Start (One-Command Install)

```bash
# Clone and install in one command
git clone https://github.com/crazidev/VisoMaster.git && cd VisoMaster && bash scripts/runpod_quick_setup.sh
```

---

## 📋 Prerequisites

### RunPod Configuration
- **GPU**: A40 (48GB) or RTX 3090/4090 (24GB)
- **vCPU**: 8-16 cores (16 recommended)
- **RAM**: 32GB+ (80GB is excellent)
- **Storage**: 30GB+ (50GB recommended)
- **Template**: PyTorch or CUDA 12.4+ template

### SSH Access
1. Add your SSH public key to RunPod:
   - Go to: https://www.runpod.io/console/user/settings
   - Navigate to "SSH Public Keys"
   - Add your key (from `~/.ssh/id_ed25519.pub`)

2. Connect to your pod:
   ```bash
   ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
   ```

---

## 📦 Step-by-Step Installation

### Step 1: Connect to RunPod

```bash
# Replace with your actual pod ID
ssh ejfcfnjvk5nklb-644112ef@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 2: Verify GPU

```bash
# Check GPU is available
nvidia-smi

# Expected output: A40 with 48GB VRAM
```

### Step 3: Clone Repository

```bash
# Clone VisoMaster
git clone https://github.com/crazidev/VisoMaster.git
cd VisoMaster

# Initialize submodules (important!)
git submodule update --init --recursive
```

**Note:** The `streamrelay` package is a git submodule. If you see an empty `packages/streamrelay` folder, run:
```bash
git submodule update --init --recursive
```

### Step 4: Install Dependencies

**Option A: Automatic Installation (Recommended)**
```bash
bash scripts/install_linux.sh
```

**Option B: Manual Installation**
```bash
# Update system packages
apt-get update
apt-get install -y python3-pip python3-dev ffmpeg libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_cu124.txt

# Download models
python3 download_models.py
```

### Step 5: Download Additional Assets

```bash
# Create dependencies directory
mkdir -p dependencies

# Download dependency files
wget -P dependencies/ https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0_dp/liveportrait_onnx.zip
wget -P dependencies/ https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0_dp/rd64-uni-refined.pth

# Extract if needed
cd dependencies
unzip -q liveportrait_onnx.zip 2>/dev/null || true
cd ..
```

### Step 6: Verify Installation

```bash
# Test CUDA availability
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Expected output:
# CUDA Available: True
# GPU: NVIDIA A40
# VRAM: 48.0 GB
```

---

## 🎮 Running VisoMaster

### Method 1: Direct Run (GUI)

```bash
# Run with GUI (requires X11 forwarding or VNC)
python3 main.py
```

**Note**: RunPod doesn't support GUI by default. See "Remote Access" section below.

### Method 2: Headless Mode (Coming Soon)

```bash
# Run in headless mode for API/CLI usage
python3 main.py --headless
```

### Method 3: Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access via RunPod's exposed port
```

---

## 🖥️ Remote Access Options

### Option 1: X11 Forwarding (SSH)

```bash
# On your local machine (Windows with VcXsrv or Xming):
# 1. Start X server (VcXsrv/Xming)
# 2. Connect with X11 forwarding:
ssh -X ejfcfnjvk5nklb-644112ef@ssh.runpod.io -i ~/.ssh/id_ed25519

# On RunPod:
export DISPLAY=localhost:10.0
python3 main.py
```

### Option 2: VNC Server

```bash
# Install VNC server
apt-get install -y x11vnc xvfb

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Start VNC server
x11vnc -display :99 -forever -shared -rfbport 5900 &

# Expose port 5900 in RunPod settings
# Connect with VNC client to: <pod-ip>:5900
```

### Option 3: NoVNC (Web-based)

```bash
# Install noVNC
apt-get install -y novnc websockify

# Start noVNC
websockify --web=/usr/share/novnc 6080 localhost:5900 &

# Access via browser: http://<pod-ip>:6080/vnc.html
```

---

## ⚙️ Optimal Settings for A40 + 16 vCPU

### Recommended Configuration

```python
# In VisoMaster Settings:
Number of Threads: 10-12
Provider Priority: TensorRT
Max DFM Models: 6-8

# For best performance:
Swapper Resolution: 512
Restorer: GPEN-BFR-1024 or CodeFormer
Frame Enhancer: RealESRGAN x4 (if needed)
```

### Performance Expectations

| Workload | Threads | Expected FPS | VRAM Usage |
|----------|---------|--------------|------------|
| 1080p Swap Only | 10 | 45-55 | 8-12 GB |
| 1080p Swap + Restore | 10-12 | 35-45 | 12-18 GB |
| 1080p Full Pipeline | 10-12 | 25-35 | 18-24 GB |
| 4K Swap + Restore | 10-12 | 12-18 | 24-32 GB |

---

## 🔧 Troubleshooting

### Issue 0: "no data transfer registered" Error

**Full Error:**
```
Error when binding input: There's no data transfer registered for copying tensors 
from Device:[DeviceType:1 MemoryType:0 DeviceId:0] to Device:[DeviceType:0 MemoryType:0 DeviceId:0]
```

**Cause:** `onnxruntime` (CPU-only) is installed instead of `onnxruntime-gpu`, or both are installed causing a conflict.

**Quick Fix:**
```bash
# Run the diagnostic script
python3 scripts/check_dependencies.py

# Or run the automated fix
bash scripts/fix_onnxruntime.sh
```

**Manual Fix:**
```bash
# 1. Uninstall CPU-only version
pip uninstall onnxruntime -y

# 2. Install GPU version
pip install onnxruntime-gpu

# 3. Verify CUDA provider is available
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Why this happens:** When you install packages, sometimes `onnxruntime` (CPU-only) gets installed as a dependency instead of `onnxruntime-gpu`. The two packages conflict, and CUDA support won't work.

### Issue 1: Submodule (streamrelay) is Empty/Missing

**Symptom:** After `git pull`, the `packages/streamrelay` folder is empty or missing

**Solution:**
```bash
# Initialize and update submodules
git submodule update --init --recursive

# Verify it worked
ls -la packages/streamrelay/

# Should see files like: __init__.py, server.py, etc.
```

**Why this happens:** Git submodules are separate repositories. When you clone or pull, you need to explicitly initialize them.

**Alternative:** Clone with submodules from the start:
```bash
git clone --recurse-submodules https://github.com/crazidev/VisoMaster.git
```

### Issue 2: CUDA Not Available

```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Issue 3: Out of Memory

```bash
# Reduce thread count in settings
# Or clear GPU cache:
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Issue 4: TensorRT Build Fails

```bash
# Check TensorRT version
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Reinstall TensorRT
pip uninstall tensorrt tensorrt-cu12-libs tensorrt-cu12-bindings -y
pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com
```

### Issue 5: Display/GUI Issues

```bash
# Set Qt platform to offscreen
export QT_QPA_PLATFORM=offscreen

# Or use VNC (see Remote Access section)
```

### Issue 6: FFmpeg Not Found

```bash
# Install FFmpeg
apt-get update
apt-get install -y ffmpeg

# Verify
ffmpeg -version
```

---

## 📊 Monitoring Performance

### GPU Monitoring

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Or detailed monitoring
nvidia-smi dmon -s u
```

### CPU Monitoring

```bash
# Install htop
apt-get install -y htop

# Monitor CPU
htop
```

### Memory Monitoring

```bash
# Check RAM usage
free -h

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 🚀 Performance Optimization

### 1. Enable TensorRT

```bash
# First run will build TensorRT engines (5-10 minutes)
# Subsequent runs will be 30-50% faster
# Engines are cached in: tensorrt-engines/
```

### 2. Optimize Thread Count

```bash
# Start with 10 threads
# Monitor GPU utilization (should be 95%+)
# If GPU < 95%: increase threads to 12
# If OOM errors: reduce threads to 8
```

### 3. Persistent Storage

```bash
# Save models to persistent storage to avoid re-downloading
# In RunPod, use network volumes for persistence
```

### 4. Batch Processing

```bash
# Process multiple videos in sequence
# Models stay loaded in VRAM between videos
# Much faster than restarting for each video
```

---

## 💾 Data Transfer

### Upload Videos to RunPod

**Option 1: SCP**
```bash
# From local machine:
scp -i ~/.ssh/id_ed25519 video.mp4 ejfcfnjvk5nklb-644112ef@ssh.runpod.io:/workspace/VisoMaster/
```

**Option 2: wget/curl**
```bash
# On RunPod:
wget https://example.com/video.mp4
```

**Option 3: RunPod Cloud Storage**
```bash
# Use RunPod's network volumes
# Mount persistent storage in pod settings
```

### Download Processed Videos

```bash
# From local machine:
scp -i ~/.ssh/id_ed25519 ejfcfnjvk5nklb-644112ef@ssh.runpod.io:/workspace/VisoMaster/output.mp4 ./
```

---

## 🔒 Security Best Practices

1. **Use SSH Keys**: Never use password authentication
2. **Firewall**: Only expose necessary ports
3. **Update Regularly**: Keep system packages updated
4. **Secure Files**: Don't store sensitive data on temporary pods
5. **Network Volumes**: Use for persistent, important data

---

## 💰 Cost Optimization

### A40 Pricing (~$0.60/hr)

| Task | Time | Cost |
|------|------|------|
| 10-min 1080p video (full pipeline) | ~20 min | $0.20 |
| 10-min 4K video (swap + restore) | ~40 min | $0.40 |
| 1-hour batch processing | 1 hour | $0.60 |

### Tips to Save Money

1. **Stop pods when not in use**
2. **Use spot instances** (cheaper but can be interrupted)
3. **Batch process** multiple videos in one session
4. **Use network volumes** to avoid re-downloading models
5. **Monitor usage** and optimize thread count

---

## 📚 Additional Resources

- **VisoMaster GitHub**: https://github.com/crazidev/VisoMaster
- **RunPod Docs**: https://docs.runpod.io/
- **Discord Support**: https://discord.gg/5rx4SQuDbp
- **Model Assets**: https://github.com/visomaster/visomaster-assets

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review RunPod logs: `journalctl -xe`
3. Check VisoMaster logs in the application
4. Join Discord: https://discord.gg/5rx4SQuDbp
5. Open GitHub issue: https://github.com/crazidev/VisoMaster/issues

---

## ✅ Quick Reference Commands

```bash
# Connect to RunPod
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519

# Check GPU
nvidia-smi

# Navigate to VisoMaster
cd /workspace/VisoMaster

# Run VisoMaster
python3 main.py

# Monitor GPU
watch -n 1 nvidia-smi

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

**Happy face swapping! 🎭**
