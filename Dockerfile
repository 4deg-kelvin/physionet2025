FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Repo-independent setup.
#
# Everything in this section is deliberately ABOVE the "DO NOT EDIT" block. That
# block ends with `COPY ./ /challenge`, so every layer after it is invalidated by
# any source edit -- including, if it were placed there, the 1 GB model download.
# Only steps that do not depend on repo contents belong up here.
# ---------------------------------------------------------------------------

# System dependencies. The -devel base image is required: fairseq-signals is
# installed from source further down and needs a compiler toolchain.
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

RUN python -m pip install --upgrade pip

# Pretrained ECG-FM backbone (~1.0 GB), fetched into a repo-independent path so a
# source edit does not re-download it.
# Drive ID 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2 is the FAIRSEQ FOUNDATION MODEL,
# mimic_iv_ecg_finetuned.pt. -O pins the output filename: fm.py loads it by that
# exact bare name, so a change in Drive's suggested filename would otherwise break
# training and inference only on the organizers' machine.
RUN pip install gdown

RUN mkdir -p /opt/ecgfm \
    && gdown 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2 \
         -O /opt/ecgfm/mimic_iv_ecg_finetuned.pt

# torch is pinned deliberately: torch>=2.6 flipped torch.load to weights_only=True,
# which the fairseq-signals checkpoint loader cannot satisfy (its `cfg` is an
# OmegaConf object). Note the cu118 wheel on a cu121 base -- the wheel bundles its
# own CUDA runtime, so this works, but it is a deliberate pairing.
#
# torchvision is pinned alongside it (0.20.1 is the pairing for torch 2.5.1) and
# MUST be installed here rather than left to `timm` in requirements_linux.txt.
# timm would pull torchvision from the default PyPI index, whose cu12x wheels drag
# in nvidia-*-cu12 packages -- and, because torchvision constrains its torch
# version, could reinstall torch itself from PyPI and silently clobber this cu118
# pin. Installing both up front means the later requirements install finds them
# already satisfied and leaves them alone.
RUN pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu118

RUN pip install "transformers==4.53.0"

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install -r requirements_linux.txt

# fm.py resolves the backbone by the bare relative name, against WORKDIR.
# Symlink rather than copy, to avoid a second 1 GB layer in the image.
# -f matters: if a developer has a local copy of the checkpoint at the repo root,
# `COPY ./ /challenge` puts a real file at this exact path and a plain `ln -s`
# would abort the build with "File exists". *.pt is also in .dockerignore.
RUN ln -sf /opt/ecgfm/mimic_iv_ecg_finetuned.pt /challenge/mimic_iv_ecg_finetuned.pt

# Source install, so it must follow the COPY above.
RUN python -m pip install -e fairseq-signals/
