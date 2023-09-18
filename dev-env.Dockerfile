FROM nvcr.io/nvidia/pytorch:23.05-py3
LABEL authors="kindroach"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends  \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/

COPY requirements.txt .

RUN python -m pip uninstall -y opencv &&  \
    python -m pip install -r requirements.txt
