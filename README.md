# Geometry-Aware Diffusion Memory Consolidation for Continual Spiking Neural Networks
This repository contains the official implementation of the paper<br>
**“Geometry-Aware Diffusion Memory Consolidation for Continual Spiking Neural Networks”**

---

## Requirements <a name="requirements"></a>

| Package            | Tested Version |
|--------------------|---------------|
| Python             | 3.10.12 |
| PyTorch            | 2.2.1 |
| CUDA               | 12.3 |
| cuDNN              | 9.0 |
| DGL                | 2.2.0 |
| Torch Geometric    | 2.5.1 |
| NetworkX           | 3.3 |
| scikit-learn       | 1.5.0 |
| SciPy              | 1.13.0 |


---

## Computing Infrastructure <a name="infrastructure"></a>

The experiments reported in the paper were run on the following hardware and software stack.  
Please list *your* actual configuration if it differs.

| Component | Specification |
|-----------|---------------|
| **CPU**   | 2 × Intel Xeon Gold 6430 (32 cores each, 2.8 GHz) |
| **GPU**   | 2 × NVIDIA A100 80 GB (PCIe, 700 W cap) |
| **System Memory** | 512 GB DDR4-3200 |
| **Storage** | 2 TB NVMe SSD (Samsung PM9A3) |
| **Operating System** | Ubuntu 22.04.4 LTS, Linux 5.15 |
| **CUDA Driver** | 12.3 |
| **cuDNN** | 9.0 |
| **Python Environment** | Conda 23.7 |
| **Other Libraries** | GCC 11.4, CMake 3.29, OpenMPI 4.1 |

---

## Getting Started <a name="getting-started"></a>

* run `NUM_TASKS=5 TASK_EPOCHS=30 REPLAY_K=5 REPLAY_BATCH=256 REPLAY_TB=20 REPLAY_BURST_START=8 REPLAY_BURST_LEN=5 REPLAY_DISTILL_MODE=ema ENABLE_EMA_TEACHER=1 REPLAY_DISTILL_TEMP=1.0 python main.py`

# Notes

* Feel free to report some inefficiencies in the code! (It's just initial version)

