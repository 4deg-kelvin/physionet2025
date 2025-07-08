FROM nvidia/cuda:12.1.0-devel-ubuntu22.04


## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install software-properties-common to add PPAs
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common

# Add the deadsnakes PPA to get Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.9 and its development headers, pip, and other tools
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    python3-opencv \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_linux.txt requirements_linux.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements_linux.txt

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

RUN ln -s $(which python3) /usr/local/bin/python


