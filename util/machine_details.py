import platform
import subprocess
import torch
import os

def get_cpu_info():
    cpu_info = subprocess.getoutput("lscpu")
    return "\n".join([line for line in cpu_info.splitlines() if any(keyword in line for keyword in ["Model name", "Architecture", "CPU(s)"])])

def get_ram_info():
    mem_info = subprocess.getoutput("free -h")
    return "\n".join(mem_info.splitlines()[0:2])

def get_gpu_info():
    try:
        gpu_info = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    except Exception:
        gpu_info = "No NVIDIA GPU detected or nvidia-smi not installed."
    return gpu_info

def get_os_info():
    return subprocess.getoutput("lsb_release -a")

def get_python_env():
    python_version = platform.python_version()
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except ImportError:
        pyg_version = "Not installed"
    return python_version, pytorch_version, cuda_version, pyg_version

# --- Collecting Info ---
print("=== CPU Info ===")
print(get_cpu_info())
print("\n=== RAM Info ===")
print(get_ram_info())
print("\n=== GPU Info ===")
print(get_gpu_info())
print("\n=== OS Info ===")
print(get_os_info())
print("\n=== Python & Library Versions ===")
pyver, torchver, cudaver, pygver = get_python_env()
print(f"Python: {pyver}")
print(f"PyTorch: {torchver}")
print(f"CUDA: {cudaver}")
print(f"PyTorch Geometric: {pygver}")
