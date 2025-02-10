# file: code_snippets/energy_monitoring/nvidia_smi_example.py

"""
Improved example of GPU Energy Monitoring using NVIDIA Management Library (pynvml).

This script retrieves the current power usage for all available NVIDIA GPUs.
It requires the `pynvml` package. Install via: pip install nvidia-ml-py3

If pynvml is not available, it falls back to using the nvidia-smi command via subprocess.
"""

import time
import subprocess
from typing import List, Optional

try:
    import pynvml
    pynvml.nvmlInit()
    USE_PYNVML = True
except ImportError:
    print("pynvml not installed, falling back to subprocess method.")
    USE_PYNVML = False

def get_gpu_power_pynvml() -> List[float]:
    """
    Retrieves GPU power usage using pynvml.

    Returns:
        List[float]: List of power usage in Watts for each GPU.
    """
    power_list = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Power usage is returned in milliwatts; convert to Watts.
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_list.append(power_mw / 1000.0)
    return power_list

def get_gpu_power_subprocess() -> Optional[List[float]]:
    """
    Retrieves GPU power usage using nvidia-smi via subprocess.

    Returns:
        List[float]: List of power usage in Watts for each GPU, or None if an error occurs.
    """
    try:
        command = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
        output = subprocess.check_output(command, shell=True, encoding='utf-8')
        # Parse each line as a float
        power_list = [float(line.strip()) for line in output.strip().split("\n") if line.strip()]
        return power_list
    except Exception as e:
        print(f"Error getting GPU power via subprocess: {e}")
        return None

def get_all_gpu_power() -> List[float]:
    """
    Retrieves GPU power usage using the available method (pynvml or subprocess).

    Returns:
        List[float]: List of power usage in Watts for each GPU.
    """
    if USE_PYNVML:
        return get_gpu_power_pynvml()
    else:
        power = get_gpu_power_subprocess()
        return power if power is not None else []

def monitor_gpu_power(interval: float = 1.0, duration: float = 10.0) -> None:
    """
    Monitors and prints GPU power usage for a given duration at specified intervals.

    Args:
        interval (float): Time between readings in seconds.
        duration (float): Total duration for monitoring in seconds.
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        power_list = get_all_gpu_power()
        if power_list:
            print("GPU Power Usage (Watts):", power_list)
        else:
            print("No GPU power data available.")
        time.sleep(interval)

if __name__ == '__main__':
    # Single snapshot example
    power = get_all_gpu_power()
    if power:
        print("Current GPU Power Usage (Watts):", power)
    else:
        print("Failed to retrieve GPU power usage.")
    
    # Monitor GPU power usage for 10 seconds with 1-second intervals
    print("\nMonitoring GPU power usage for 10 seconds...")
    monitor_gpu_power(interval=1.0, duration=10.0)
