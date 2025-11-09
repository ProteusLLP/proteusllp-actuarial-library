# Use Python 3.13-slim-bookworm as the base image
FROM python:3.13-slim-bookworm AS base

# Set non-interactive mode to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and wget
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    nodejs \
    npm \
    locales \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/en_GB.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

# Add NVIDIA package repositories and install CUDA toolkit
# Using CUDA 12.4 which is more widely available and compatible with driver 580.x
# Installing cuda-toolkit-12-4 instead of cuda-runtime-12-4 to get development headers
# required for CuPy to compile CUDA kernels at runtime
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Set CUDA environment variables for compilation
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDACXX=${CUDA_HOME}/bin/nvcc


# Install pyright globally for type checking
RUN npm install -g pyright

# Install uv and PDM for fast package management
RUN pip install uv pdm

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml pdm.lock* ./

# Install only main dependencies using PDM with uv backend
RUN pdm install --no-self

# DEVELOPMENT STAGE ===========================
FROM base AS dev

# Install development tools
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install dev dependencies (test + dev groups)
RUN pdm install -dG test -dG dev --no-self

# Switch to non-root user
USER vscode

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# CI STAGE ===========================
FROM base AS ci

# Set working directory
WORKDIR /app

# Copy source code for CI
COPY . .

# Install test dependencies and the package itself
RUN pdm install -dG test
