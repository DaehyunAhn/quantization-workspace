FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git git-lfs ca-certificates build-essential pkg-config cmake \
      wget curl && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install vllm==0.9.1 llmcompressor==0.5.2 autoawq datasets fire transformers==4.52.4

RUN git clone --recurse-submodules https://github.com/DaehyunAhn/quantization-workspace.git

RUN pip install -r /workspace/quantization-workspace/FunctionChat-Bench/requirements.txt
RUN cd /workspace/quantization-workspace/lm-evaluation-harness && pip install -e . && cd ..
RUN mkdir -p /workspace/quantization-workspace/models
RUN mkdir -p /workspace/quantization-workspace/outputs
RUN mkdir -p /workspace/quantization-workspace/datasets

WORKDIR /workspace/quantization-workspace

RUN huggingface-cli download HAERAE-HUB/HAE_RAE_BENCH_1.1 --repo-type dataset --local-dir /workspace/quantization-workspace/datasets/HAE_RAE_BENCH_1.1
RUN huggingface-cli download HAERAE-HUB/KMMLU --repo-type dataset --local-dir /workspace/quantization-workspace/datasets/KMMLU
RUN huggingface-cli download HAERAE-HUB/HRM8K --repo-type dataset --local-dir /workspace/quantization-workspace/datasets/HRM8K
RUN huggingface-cli download neuralmagic/LLM_compression_calibration --repo-type dataset --local-dir /workspace/quantization-workspace/datasets/LLM_compression_calibration
RUN huggingface-cli download wikitext --repo-type dataset --local-dir /workspace/quantization-workspace/datasets/wikitext

CMD sleep infinity
