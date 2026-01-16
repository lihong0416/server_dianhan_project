# 基于 NVIDIA 官方 CUDA 镜像（Ubuntu 22.04 + CUDA 12.1 + cuDNN8）
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    #PYTHONPATH=/alg:$PYTHONPATH \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /alg

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates git \
    gcc g++ make \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    tzdata \
 && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
 && echo "Asia/Shanghai" > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda（不依赖你仓库里放安装包）
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -u -p $CONDA_DIR \
 && rm -f /tmp/miniconda.sh 

# 创建 conda 环境（env 名叫 aeq）
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r  --channel https://repo.anaconda.com/pkgs/msys2
RUN conda update --force conda
RUN conda create -n aeq python=3.12 -y \
    && echo "conda activate aeq" >> ~/.bashrc

# 先拷贝 requirements 以便缓存
COPY requirements.txt /alg/requirements.txt

# 安装 PyTorch（CUDA 12.1 -> cu121）+ 依赖（清华 pip 源）
RUN conda run -n aeq python -m pip install --upgrade pip \
 && conda run -n aeq pip install --no-cache-dir \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && conda run -n aeq pip install --no-cache-dir -r /alg/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 必须：server_qingdao.py / qdyq_peakrules_model.py / cols_to_drop_v2.py / utils / sputils / config.yaml
COPY server_qingdao.py /alg/server_qingdao.py
COPY qdyq_peakrules_model.py /alg/qdyq_peakrules_model.py
COPY cols_to_drop_v2.py /alg/cols_to_drop_v2.py
COPY utils/ /alg/utils/
COPY sputils/ /alg/sputils/
COPY config.yaml /alg/config.yaml
COPY README.md /alg/README.md

EXPOSE 8000

# 启动服务
CMD ["/opt/conda/envs/aeq/bin/python", "server_qingdao.py"]
