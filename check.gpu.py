import onnxruntime as ort
import os
import subprocess

def get_system_info():
    """
    Collects and prints system information relevant to TensorRT and ONNX Runtime.
    """

    print("--- System Information for TensorRT/ONNX Runtime Debugging ---")

    # 1. ONNX Runtime Available Providers
    try:
        available_providers = ort.get_available_providers()
        print("\n1. ONNX Runtime Available Providers:")
        for provider in available_providers:
            print(f"   - {provider}")
        if 'TensorRTExecutionProvider' in available_providers:
            print("✅ TensorRTExecutionProvider IS AVAILABLE in ONNX Runtime.")
        else:
            print("❌ TensorRTExecutionProvider is NOT AVAILABLE in ONNX Runtime.")
    except Exception as e:
        print("\n1. ONNX Runtime Available Providers: Error retrieving providers.")
        print(f"   Error: {e}")

    # 2. System PATH Environment Variable
    print("\n2. System PATH Environment Variable:")
    path_env = os.environ.get('PATH')
    if path_env:
        paths = path_env.split(os.pathsep) # Split PATH into directories
        for path_dir in paths:
            print(f"   - {path_dir}")
    else:
        print("   PATH environment variable is not set.")

    # 3. Python Executable Path (for venv check)
    print("\n3. Python Executable Path:")
    python_executable = sys.executable
    print(f"   - {python_executable}")
    venv_prefix = os.environ.get('VIRTUAL_ENV')
    if venv_prefix:
        print("   ✅ Running in a Virtual Environment (venv):")
        print(f"     Prefix: {venv_prefix}")
        if python_executable.startswith(venv_prefix):
             print("     ✅ Python executable IS inside the virtual environment.")
        else:
            print("     ❌ WARNING: Python executable is NOT inside the virtual environment path.")
    else:
        print("   ❌ NOT running in a Virtual Environment (venv) (This might be okay if you installed ONNX Runtime system-wide).")


    # 4. NVIDIA Driver Version (using nvidia-smi)
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
        driver_version = nvidia_smi_output.strip()
        print("\n4. NVIDIA Driver Version (from nvidia-smi):")
        print(f"   - Driver Version: {driver_version}")
    except FileNotFoundError:
        print("\n4. NVIDIA Driver Version: nvidia-smi not found. Make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n4. NVIDIA Driver Version: Error running nvidia-smi: {e}")

    # 5. CUDA Version (using nvcc --version)
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"], text=True)
        cuda_version_line = next((line for line in nvcc_output.splitlines() if "release" in line), None)
        if cuda_version_line:
            cuda_version = cuda_version_line.split("release")[1].strip().split(",")[0].strip()
            print("\n5. CUDA Version (from nvcc --version):")
            print(f"   - CUDA Version: {cuda_version}")
        else:
            print("\n5. CUDA Version: Could not parse CUDA version from nvcc output.")

    except FileNotFoundError:
        print("\n5. CUDA Version: nvcc not found. Make sure CUDA Toolkit is installed and nvcc is in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n5. CUDA Version: Error running nvcc --version: {e}")

    print("\n--- End of System Information ---")

if __name__ == "__main__":
    import sys  # Import sys here, needed for sys.executable
    get_system_info()