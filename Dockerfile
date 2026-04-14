FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev     libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev     libxmlsec1-dev libffi-dev liblzma-dev     && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz &&     tar -xzf Python-${PYTHON_VERSION}.tgz &&     cd Python-${PYTHON_VERSION} &&     ./configure --enable-optimizations &&     make -j$(nproc) &&     make altinstall &&     cd .. &&     rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python &&     ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN python -m pip install --upgrade pip &&     pip install -r requirements.txt

RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade

# Install Flash Attention 4 (CuTe) for Blackwell support
RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention &&     pip install -e /tmp/flash-attention/flash_attn/cute --no-build-isolation &&     rm -rf /tmp/flash-attention/.git

# Install Quartet-II NVFP4 kernel dependencies
RUN pip install flashinfer-python nvtx nanobind scikit-build-core cmake

# Reinstall nightly torch in case flashinfer pulled a different version
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade

# Reinstall scipy (torch reinstall may have removed it)
RUN pip install scipy

# Build and install Quartet-II NVFP4 kernels for B200 (sm100)
# https://arxiv.org/abs/2601.22813
COPY kernels /tmp/quartet2-kernels
RUN cd /tmp/quartet2-kernels && CUDAARCHS=100 pip install --no-build-isolation --no-deps . &&     rm -rf /tmp/quartet2-kernels

CMD ["bash"]
ENTRYPOINT []
