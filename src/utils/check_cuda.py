#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import torch

def get_gpu_processes(gpu_index):
    """Get processes running on a specific GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=" + str(gpu_index), 
             "--query-compute-apps=pid,used_memory", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                pid, mem = line.split(',')
                processes.append((int(pid.strip()), float(mem.strip())))
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def check_gpu():
    expected_gpus = int(os.environ.get("NUM_GPUS", "1"))

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False

    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s) (expected {expected_gpus})")
    if device_count != expected_gpus:
        print(f"❌ Expected {expected_gpus} GPU(s), got {device_count}")
        return False

    # "Supported" is a compute-capability floor, not a product list: >= 8.0 covers
    # A100 (8.0), H100/H20/H800 (9.0) and Blackwell (10.x), so migrating to a new
    # card does not fail this check until the card is genuinely too old. Override
    # the floor with VTB_MIN_CUDA_CAPABILITY if a run needs a newer feature set.
    min_capability = os.environ.get("VTB_MIN_CUDA_CAPABILITY", "8.0")
    min_parts = (min_capability.split(".") + ["0"])[:2]
    min_major, min_minor = (int(x) for x in min_parts)

    gpu_ok = False
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {name} ({props.total_memory / 1e9:.1f} GB, sm_{props.major}{props.minor})")

        if (props.major, props.minor) >= (min_major, min_minor):
            gpu_ok = True

        # Check for running processes on this GPU
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        physical = visible.split(",")[i] if visible else i
        processes = get_gpu_processes(physical)

        if processes is None:
            print(f"  ⚠ Could not check processes on GPU {i} (nvidia-smi failed)")
        elif processes:
            print(f"  ❌ GPU {i} has {len(processes)} process(es) running:")
            for pid, mem in processes:
                print(f"      PID {pid}: {mem:.1f} MiB")
            return False
        else:
            print(f"  ✓ GPU {i} is idle")

    if gpu_ok:
        print(f"✓ Supported GPU detected ({device_count} GPU(s))")
    else:
        print(f"❌ No supported GPU (compute capability >= {min_capability}) found")
        return False

    return True

if __name__ == "__main__":
    cuda_available = check_gpu()
    if not cuda_available:
        Path("cuda_not_available").touch()

    import sys
    sys.exit(0 if cuda_available else 1)
