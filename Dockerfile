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

# Set the working directory early
WORKDIR /challenge

# --- CACHING OPTIMIZATION ---
# 1. Copy ONLY the requirements file first.
COPY requirements_linux.txt .

# 2. Install the dependencies. This layer is now cached. 
# It will only re-run if you change requirements_linux.txt.
RUN pip install -r requirements_linux.txt

# Also install torch in this cached section.
# Pinned: torch>=2.6 flips torch.load to weights_only=True, which breaks loading the
# fairseq-signals ECG-FM checkpoint (its `cfg` is an OmegaConf object).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# 3. NOW, copy the rest of your project code.
# Changes to your scripts will only invalidate the cache from this point onward.
COPY . .

# 4. Install packages that depend on your source code and download data.
# These commands run after your code is copied. We can combine them into one layer.
# Install the pretrained MAE, must be after installing pip reqs so that gdown is installed
# 1o4zghwYD4j61SgazijwtKRfyCZyOjkyz is the mae_vit_ecg pretrained model
# 1c_uJOo08AYPb60wRFnrhp9LPMUmuFqly is the pretrained model for MAE
# 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2 is for FAIRSEQ FOUNDATION MODEL 
RUN gdown 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2

RUN python -m pip install -e fairseq-signals/

RUN  python -m pip install "transformers==4.53.0"     
