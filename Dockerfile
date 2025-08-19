FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

# Install Python, NVIDIA utilities, and other dependencies
RUN apt-get update && apt-get install -y python3 python3-pip bash-completion git make build-essential gcc cuda-toolkit nvidia-utils-580 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install torch torchsde torchvision torchaudio einops \
    'transformers' 'tokenizers' sentencepiece \
    'safetensors' aiohttp pyyaml Pillow scipy tqdm psutil dotenv

COPY requirements.txt /self-forcing/requirements.txt
WORKDIR /self-forcing

RUN pip install -r requirements.txt
RUN pip install numpy faster-whisper nvidia-cublas-cu11 nvidia-cudnn-cu11
RUN pip install flash-attn --no-build-isolation && \
    rm -rf /self-forcing

# Run with --gpus all and mount Self-Forcing to /self-forcing