# Base stage - Python + system dependencies
FROM python:3.13-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage - install all packages
FROM base AS deps

# Install uv and PDM for fast package management
RUN pip install uv pdm

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml pdm.lock* ./

# Install only main dependencies using PDM with uv backend
RUN pdm install --no-self

# Development stage
FROM deps AS dev

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
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

# CI stage
FROM deps AS ci

# Set working directory
WORKDIR /app

# Copy source code for CI
COPY . .

# Install test dependencies and the package itself
RUN pdm install -dG test