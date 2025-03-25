FROM nvidia/cuda:11.4.0-base-ubuntu20.04


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

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

RUN gdown https://drive.google.com/drive/folders/1SCVc3bvU2veiS3zKuvJ-ohE9bW0ndu6B?usp=sharing --folder
