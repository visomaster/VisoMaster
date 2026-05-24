#!/bin/bash
# Fix ONNX Runtime installation for CUDA support
# This resolves the "no data transfer registered" error

set -e

echo "=========================================="
echo "Fixing ONNX Runtime for CUDA Support"
echo "=========================================="
echo ""

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "⚠ Warning: Not in a conda environment"
fi

echo ""
echo "Step 1: Uninstalling CPU-only onnxruntime..."
pip uninstall onnxruntime-gpu -y || true

echo ""
echo "Step 2: Installing onnxruntime-gpu..."
pip install onnxruntime-gpu

echo ""
echo "Step 3: Verifying installation..."
python3 << 'EOF'
import sys
try:
    import onnxruntime as ort
    print(f"✓ ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"✓ Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDAExecutionProvider is available!")
        print("\n✅ Fix successful! CUDA support is enabled.")
        sys.exit(0)
    else:
        print("✗ CUDAExecutionProvider NOT available")
        print("\n⚠ CUDA support not enabled. Check CUDA installation.")
        sys.exit(1)
except ImportError as e:
    print(f"✗ Error importing onnxruntime: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
