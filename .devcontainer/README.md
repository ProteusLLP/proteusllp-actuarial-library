# Dev Container Configurations

This project provides two dev container configurations to suit different development needs:

## 🚀 CPU-only (Lightweight)

**Path:** `.devcontainer/cpu/`

**Use when:**
- You don't need GPU acceleration
- You want a faster build time (~2-3 minutes vs ~10+ minutes)
- You want a smaller container image (~800MB vs ~4GB+)
- You're doing general development, testing, or working on non-GPU code

**Features:**
- Python 3.13
- NumPy (CPU-only)
- All standard development tools
- Fast to build and start

## 🎮 GPU (CUDA-enabled)

**Path:** `.devcontainer/gpu/`

**Use when:**
- You need GPU acceleration with CuPy
- You're running large-scale simulations
- You're testing GPU-specific functionality
- You have NVIDIA GPU hardware available

**Features:**
- Everything from CPU container
- CUDA Toolkit 12.4
- CuPy for GPU-accelerated NumPy operations
- NVIDIA Container Runtime support

**Requirements:**
- NVIDIA GPU
- NVIDIA drivers installed on host
- Docker with NVIDIA Container Runtime

## How to Choose

When you open this project in VS Code, you'll be prompted to select which dev container configuration to use:

1. **Click "Reopen in Container"**
2. **Select configuration:**
   - `Proteus Actuarial Library (CPU)` - for lightweight development
   - `Proteus Actuarial Library (GPU)` - for GPU-accelerated work

You can switch between configurations at any time:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Dev Containers: Rebuild and Reopen in Container"
- Choose the configuration you want

## Build Times & Sizes

| Configuration | Build Time | Image Size | Dependencies |
|---------------|------------|------------|--------------|
| CPU           | ~2-3 min   | ~800 MB    | No CUDA      |
| GPU           | ~10-15 min | ~4-5 GB    | CUDA 12.4    |

## Tips

- **Default choice:** Use CPU for most development work
- **When to use GPU:** Only when you need CuPy or are testing GPU-specific code
- **First time setup:** The GPU container takes longer to build due to CUDA toolkit installation, but subsequent builds are cached
