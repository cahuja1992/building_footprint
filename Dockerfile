FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         python-qt4 \
         libjpeg-dev \
         zip \
         unzip \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-numpy && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    libcurl3-dev  && \
    rm -rf /var/lib/apt/lists/*    

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN ln -s -f /usr/bin/python3 /usr/bin/python

WORKDIR /ws
ENTRYPOINT ["/usr/bin/entrypoint.sh"]
