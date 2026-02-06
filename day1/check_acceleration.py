import torch
import sys
import platform

def check_system():
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split(' ')[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("-" * 30)

def check_acceleration():
    # 1. Check for NVIDIA CUDA
    if torch.cuda.is_available():
        print(f"🚀 CUDA is AVAILABLE!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return "cuda"
        
    # 2. Check for Apple Silicon MPS (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        print("🚀 MPS (Apple Silicon) is AVAILABLE!")
        print("   Using Metal Performance Shaders for acceleration.")
        return "mps"
        
    # 3. Fallback to CPU
    else:
        print("⚠️  NO ACCELERATION DETECTED.")
        print("   Training will be slow on CPU.")
        return "cpu"

if __name__ == "__main__":
    check_system()
    device_name = check_acceleration()
    
    # Simple Tensor Test
    print("-" * 30)
    print(f"Testing Tensor creation on '{device_name}'...")
    try:
        x = torch.ones(1, device=device_name)
        print("✅ Tensor created successfully.")
    except Exception as e:
        print(f"❌ Error creating tensor: {e}")
