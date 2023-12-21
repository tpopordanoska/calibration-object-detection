FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# To make CONDA work
SHELL ["/bin/bash", "--login", "-c"]

# Setting up the system (as 1 layer for compactness)
RUN apt-get update && apt-get install -y --no-install-recommends wget

# Getting conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && rm Miniconda3-latest-Linux-x86_64.sh

RUN echo ". \"/opt/miniconda/etc/profile.d/conda.sh\"" >> ~/.bashrc
ENV PATH=/opt/miniconda/bin:${PATH}

# Conda env setup
RUN mkdir /opt/misc/
RUN mkdir /opt/workdir/
RUN mkdir /opt/configs/
RUN mkdir /opt/code

COPY env.yaml /opt/misc/env.yaml

RUN conda create -n detectron python=3.9 matplotlib pip
SHELL ["conda", "run", "-n", "detectron", "/bin/bash", "--login", "-c"]
RUN echo "conda activate detectron" >> ~/.bashrc
RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip install setuptools==59.5.0

ENV DETECTRON2_DATASETS=/data/
WORKDIR /opt/workdir/