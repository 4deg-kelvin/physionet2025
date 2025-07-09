FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-dev \
     python3-venv \
     curl \
     git \
     wget \
     bzip2 \
     ca-certificates \
     libglib2.0-0 \
     libxext6 \
     libsm6 \
     libxrender1 \
     && rm -rf /var/lib/apt/lists/*


RUN ln -s $(which python3) /usr/local/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# # Create conda environment from YAML


# Copy project files
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install -r requirements_linux.txt

# Optional: install additional pip packages
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118


# COPY requirements_linux.txt requirements_linux.txt

# RUN python3 -m pip install --upgrade pip






