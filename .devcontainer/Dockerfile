# 1. 기반이 될 리눅스 이미지를 선택합니다. (Ubuntu 22.04)
FROM ubuntu:22.04

# 2. 시스템을 업데이트하고, 개발에 필요한 기본 도구들을 설치합니다.
# NONINTERACTIVE 설정은 설치 중 질문 창이 뜨는 것을 방지합니다.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    gdb \
    python3 \
    python3-pip \
    libgl1-mesa-glx

# 3. 파이썬 라이브러리를 설치합니다.
RUN pip3 install numpy opencv-python tqdm scikit-image pyyaml

# 설치 후 불필요한 파일을 정리하여 이미지 용량을 줄입니다.
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=