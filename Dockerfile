FROM nvidia/cuda:12.1.0-devel-ubuntu22.04


## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

COPY requirements_linux.txt requirements_linux.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements_linux.txt

RUN gdown https://drive.google.com/drive/folders/1PkjADoaXqOEDceCJxLzVgXF9Mc6oWzdV?usp=drive_link --folder

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

RUN ln -s $(which python3) /usr/local/bin/python


