# Quantization Workspace

# 1. Docker build
- Build the docker image using Dockerfile or simply pull docker image.
```
docker build -t [TAG_NAME] .
```

# 2. Download the model & required datasets
```
huggingface-cli download Qwen/Qwen3-32B --local-dir <LOCAL_MODEL_PATH>

huggingface-cli download HAERAE-HUB/HAE_RAE_BENCH_1.1 --repo-type dataset --local-dir <LOCAL_HAERAE_PATH>
huggingface-cli download HAERAE-HUB/KMMLU --repo-type dataset --local-dir <LOCAL_KMMLU_PATH>
huggingface-cli download HAERAE-HUB/HRM8K --repo-type dataset --local-dir <LOCAL_HRM8K_PATH>

# Calibration dataset for quantization
huggingface-cli download neuralmagic/LLM_compression_calibration repo-type dataset <LOCAL_CALIB_PATH>

# Calibration dataset for layer pruning
huggingface-cli download wikitext --repo-type dataset <LOCAL_WIKITEXT_PATH>
```

# 3. Quantize or Prune the model & Benchmark
```
# For FP8 (Dynamic, static, KV Cache) Quantization
bash fp8.sh <LOCAL_MODEL_PATH> <TP_SIZE> <OPENAI_API_KEY> <LOCAL_CALIB_PATH> <LOCAL_HAERAE_PATH> <LOCAL_KMMLU_PATH> <LOCAL_HRM8K_PATH>

# For INT8 (Dynamic, static) Quantization
bash int8.sh <LOCAL_MODEL_PATH> <TP_SIZE> <OPENAI_API_KEY> <LOCAL_CALIB_PATH> <LOCAL_HAERAE_PATH> <LOCAL_KMMLU_PATH> <LOCAL_HRM8K_PATH>

# For GPTQ (W4A16) Quantization
bash gptq.sh <LOCAL_MODEL_PATH> <TP_SIZE> <OPENAI_API_KEY> <LOCAL_CALIB_PATH> <LOCAL_HAERAE_PATH> <LOCAL_KMMLU_PATH> <LOCAL_HRM8K_PATH>

# For AWQ (W4A16) Quantization
bash awq.sh <LOCAL_MODEL_PATH> <TP_SIZE> <OPENAI_API_KEY> <LOCAL_CALIB_PATH> <LOCAL_HAERAE_PATH> <LOCAL_KMMLU_PATH> <LOCAL_HRM8K_PATH>

# For layer-pruning
bash sleb.sh <LOCAL_MODEL_PATH> <TP_SIZE> <OPENAI_API_KEY> <LOCAL_WIKITEXT_PATH>/wikitext-2-raw-v1 <LOCAL_HAERAE_PATH> <LOCAL_KMMLU_PATH> <LOCAL_HRM8K_PATH>
```

# 4. (Optional) Save the benchmark results to csv
```
python to_csv.py output/<RESULT_FOLDER> summary.csv
```
