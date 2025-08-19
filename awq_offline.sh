MODEL_PATH=$1
TP=$2

CALIBRATION_DATASET_PATH="/workspace/quantization-workspace/datasets/LLM_compression_calibration"
HAERAE_DATASET_PATH="/workspace/quantization-workspace/datasets/HAE_RAE_BENCH_1.1"
KMMLU_DATASET_PATH="/workspace/quantization-workspace/datasets/KMMLU"
HRM8K_DATASET_PATH="/workspace/quantization-workspace/datasets/HRM8K"

QUANT_CONFIG_FILES=("configs/autoawq/w4a16_g64.json" \
                "configs/autoawq/w4a16_g32.json") 

update_dataset_path() {
    local haerae_config_file="lm-evaluation-harness/lm_eval/tasks/haerae/_default_haerae_yaml"
    local kmmlu_config_file="lm-evaluation-harness/lm_eval/tasks/kmmlu/default/_default_kmmlu_yaml"
    local hrm8k_config_file="lm-evaluation-harness/lm_eval/tasks/hrm8k/default/_hrm8k_yaml"
    
    local haerae_temp_file="${haerae_config_file}.tmp"
    local kmmlu_temp_file="${kmmlu_config_file}.tmp"
    local hrm8k_temp_file="${hrm8k_config_file}.tmp"

    cp "$haerae_config_file" "${haerae_config_file}.backup"
    cp "$kmmlu_config_file" "${kmmlu_config_file}.backup"
    cp "$hrm8k_config_file" "${hrm8k_config_file}.backup"

    sed "s|HAERAE-HUB/HAE_RAE_BENCH_1.1|$HAERAE_DATASET_PATH|g" "$haerae_config_file" > "$haerae_temp_file"
    sed "s|HAERAE-HUB/KMMLU|$KMMLU_DATASET_PATH|g" "$kmmlu_config_file" > "$kmmlu_temp_file"
    sed "s|HAERAE-HUB/HRM8K|$HRM8K_DATASET_PATH|g" "$hrm8k_config_file" > "$hrm8k_temp_file"
    mv "$haerae_temp_file" "$haerae_config_file"
    mv "$kmmlu_temp_file" "$kmmlu_config_file"
    mv "$hrm8k_temp_file" "$hrm8k_config_file"
    echo "Updated dataset path of lm-evaluation-harness"
}

restore_dataset_path() {
    local haerae_config_file="lm-evaluation-harness/lm_eval/tasks/haerae/_default_haerae_yaml"
    local kmmlu_config_file="lm-evaluation-harness/lm_eval/tasks/kmmlu/default/_default_kmmlu_yaml"
    local hrm8k_config_file="lm-evaluation-harness/lm_eval/tasks/hrm8k/default/_hrm8k_yaml"
    
    local haerae_backup_file="${haerae_config_file}.backup"
    local kmmlu_backup_file="${kmmlu_config_file}.backup"
    local hrm8k_backup_file="${hrm8k_config_file}.backup"

    mv "$haerae_backup_file" "$haerae_config_file"
    mv "$kmmlu_backup_file" "$kmmlu_config_file"
    mv "$hrm8k_backup_file" "$hrm8k_config_file"
    echo "Restore dataset path of lm-evaluation-harness"
}

# Update dataset path of lm-evaluation-harness
update_dataset_path

# Quantize model
for QUANT_CONFIG_FILE in ${QUANT_CONFIG_FILES[@]}; do
    # Extract config name from file path for output directory
    CONFIG_NAME=$(basename "$QUANT_CONFIG_FILE" .json)
    OUTPUT_DIR="outputs/${CONFIG_NAME}"
    QUANTIZED_MODEL_PATH="models/$(basename "$MODEL_PATH")_${CONFIG_NAME}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    python quantize.py --model_id $MODEL_PATH \
        --save_dir $QUANTIZED_MODEL_PATH \
        --recipe $QUANT_CONFIG_FILE \
        --dataset $CALIBRATION_DATASET_PATH \
        --compressor_type awq \
        --split train \
        --column text \
        --num_calibration_samples 512 \
        --max_sequence_length 2048

    # 1. Run lm-evaluation-harness (HAERAE_1.1, KMMLU, HRM8K)
    VLLM_USE_V1=0 lm_eval --model vllm --tasks haerae,kmmlu,hrm8k --num_fewshot 0 --batch_size 4 \
        --model_args pretrained=${QUANTIZED_MODEL_PATH},gpu_memory_utilization=0.8,tensor_parallel_size=${TP} \
        --output ${OUTPUT_DIR}/lm_evals.json
    mv ${OUTPUT_DIR}/lm_evals*.json ${OUTPUT_DIR}/lm_evals.json

done

# Restore dataset path of lm-evaluation-harness
restore_dataset_path