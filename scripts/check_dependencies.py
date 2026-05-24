#!/usr/bin/env python3
"""
Diagnostic script to check CUDA, PyTorch, and ONNX Runtime installation.
Run this to diagnose the "no data transfer registered" error.
"""

import sys
import subprocess

def check_package_version(package_name):
    """Check if a package is installed and return its version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return None
    except Exception as e:
        return None

def main():
    print("=" * 70)
    print("VisoMaster Dependency Checker")
    print("=" * 70)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Check PyTorch
    print("Checking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch installed: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  ✓ GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available in PyTorch")
    except ImportError:
        print("  ✗ PyTorch not installed")
    print()
    
    # Check ONNX Runtime
    print("Checking ONNX Runtime...")
    ort_version = check_package_version("onnxruntime")
    ort_gpu_version = check_package_version("onnxruntime-gpu")
    
    if ort_version and ort_gpu_version:
        print("  ⚠ WARNING: Both onnxruntime and onnxruntime-gpu are installed!")
        print(f"    onnxruntime: {ort_version}")
        print(f"    onnxruntime-gpu: {ort_gpu_version}")
        print("  → This can cause conflicts. Uninstall onnxruntime:")
        print("     pip uninstall onnxruntime -y")
    elif ort_gpu_version:
        print(f"  ✓ onnxruntime-gpu installed: {ort_gpu_version}")
    elif ort_version:
        print(f"  ⚠ onnxruntime (CPU-only) installed: {ort_version}")
        print("  → For CUDA support, install onnxruntime-gpu:")
        print("     pip uninstall onnxruntime -y")
        print("     pip install onnxruntime-gpu")
    else:
        print("  ✗ No ONNX Runtime installed")
        print("  → Install onnxruntime-gpu:")
        print("     pip install onnxruntime-gpu")
    
    # Test ONNX Runtime providers
    try:
        import onnxruntime as ort
        print(f"\n  Available providers: {ort.get_available_providers()}")
        
        has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        has_tensorrt = 'TensorrtExecutionProvider' in ort.get_available_providers()
        
        if has_cuda:
            print("  ✓ CUDAExecutionProvider available")
        else:
            print("  ✗ CUDAExecutionProvider NOT available")
            print("  → This is the cause of your error!")
            
        if has_tensorrt:
            print("  ✓ TensorrtExecutionProvider available")
        else:
            print("  ⚠ TensorrtExecutionProvider not available (optional)")
            
    except ImportError:
        print("  ✗ Cannot import onnxruntime")
    print()
    
    # Check CUDA toolkit
    print("Checking CUDA Toolkit...")
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  ✓ {line.strip()}")
        else:
            print("  ⚠ nvcc not found in PATH")
    except FileNotFoundError:
        print("  ⚠ nvcc not found (CUDA toolkit may not be installed)")
    print()
    
    # Check cuDNN
    print("Checking cuDNN...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  ✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
        else:
            print("  ⚠ Cannot check cuDNN (CUDA not available)")
    except:
        print("  ⚠ Cannot determine cuDNN version")
    print()
    
    # Summary and recommendations
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    issues = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("PyTorch cannot access CUDA")
    except ImportError:
        issues.append("PyTorch not installed")
    
    if ort_version and ort_gpu_version:
        issues.append("Both onnxruntime and onnxruntime-gpu installed (conflict)")
    elif not ort_gpu_version:
        issues.append("onnxruntime-gpu not installed")
    
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            issues.append("CUDAExecutionProvider not available in ONNX Runtime")
    except:
        pass
    
    if issues:
        print("\n⚠ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n📋 Recommended fix:")
        print("  1. Uninstall CPU-only onnxruntime:")
        print("     pip uninstall onnxruntime -y")
        print()
        print("  2. Install onnxruntime-gpu:")
        print("     pip install onnxruntime-gpu")
        print()
        print("  3. Verify CUDA is working:")
        print("     python -c \"import torch; print(torch.cuda.is_available())\"")
        print()
    else:
        print("\n✓ All dependencies look good!")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
