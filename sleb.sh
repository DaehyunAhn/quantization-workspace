MODEL_PATH=$1
TP=$2
API_KEY=$3
CALIBRATION_DATASET_PATH=$4
HAERAE_DATASET_PATH=$5
KMMLU_DATASET_PATH=$6
HRM8K_DATASET_PATH=$7

check_server_health() {
    local max_attempts=60  # 10 minutes timeout
    local attempt=1
    
    echo "Checking server health..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    echo "Server failed to start within timeout"
    return 1
}

# Function to cleanup server
cleanup_server() {
    echo "Shutting down server..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 
    fi
    # Also kill any remaining vllm processes
    pkill -f "vllm serve"

    echo "Waiting for server to shutdown..."
    if [ ! -z "$SERVER_PID" ]; then
        wait $SERVER_PID
    fi
    sleep 10
    echo "Server shutdown completed"
}

update_functionchat_config() {
    local config_file="FunctionChat-Bench/config/openai.cfg"
    local temp_file="${config_file}.tmp"

    cp "$config_file" "${config_file}.backup"
    
    sed "s/__YOUR_OPENAI_KEY__/$API_KEY/g" "$config_file" > "$temp_file"
    mv "$temp_file" "$config_file"
    
    echo "Updated FunctionChat-Bench config with API key"
}

restore_functionchat_config() {
    local config_file="FunctionChat-Bench/config/openai.cfg"
    local backup_file="${config_file}.backup"

    mv "$backup_file" "$config_file"
    
    echo "Restore FunctionChat-Bench config"
}


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

# Function to generate GPU devices string based on TP
generate_gpu_devices() {
    local tp=$1
    local devices=""
    
    for ((i=0; i<tp; i++)); do
        if [ $i -eq 0 ]; then
            devices="$i"
        else
            devices="$devices,$i"
        fi
    done
    
    echo "$devices"
}

# Set trap to cleanup on script exit
trap cleanup_server EXIT

# Update FunctionChatBench config with API key and dataset path of lm-evaluation-harness
update_functionchat_config
update_dataset_path

# Quantize model
for NUM_REMOVE_BLOCKS in 4 8 16 32; do
    # Extract config name from file path for output directory
    OUTPUT_DIR="outputs/SLEB_REMOVE_$NUM_REMOVE_BLOCKS"
    QUANTIZED_MODEL_PATH="models/$(basename "$MODEL_PATH")_SLEB_REMOVE_${NUM_REMOVE_BLOCKS}"
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    python SLEB/sleb.py --model_name $MODEL_PATH \
        --num_blocks 64 --num_remove_blocks $NUM_REMOVE_BLOCKS \
        --save --save_dir $QUANTIZED_MODEL_PATH \
        --dataset $CALIBRATION_DATASET_PATH

    # 1. Run lm-evaluation-harness (HAERAE_1.1, KMMLU, HRM8K)
    VLLM_USE_V1=0 lm_eval --model vllm --tasks haerae,kmmlu,hrm8k --num_fewshot 0 --batch_size 4 \
        --model_args pretrained=${QUANTIZED_MODEL_PATH},gpu_memory_utilization=0.8,tensor_parallel_size=${TP} \
        --output ${OUTPUT_DIR}/lm_evals.json
    mv ${OUTPUT_DIR}/lm_evals*.json ${OUTPUT_DIR}/lm_evals.json

    # # 2. Run LogicKor
    python LogicKor/generator.py --model $QUANTIZED_MODEL_PATH --gpu_devices $(generate_gpu_devices $TP) --strategy default \
                                -o LogicKor/generated/$(basename "$QUANTIZED_MODEL_PATH") -q LogicKor/questions.jsonl
    python LogicKor/evaluator.py -o LogicKor/generated/$(basename "$QUANTIZED_MODEL_PATH") -k ${API_KEY} -t 30 \
                                -e LogicKor/evaluated/$(basename "$QUANTIZED_MODEL_PATH")
    python LogicKor/score.py -p LogicKor/evaluated/$(basename "$QUANTIZED_MODEL_PATH")/default.jsonl -o ${OUTPUT_DIR}/logickor.jsonl

    3. Run Functionchat-Bench
    vllm serve ${QUANTIZED_MODEL_PATH} \
        --tensor-parallel-size ${TP} \
        --max-num-seqs 256 \
        --max-model-len 4096 \
        --tool-call-parser hermes \
        --disable-log-requests \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        >& ${OUTPUT_DIR}/vllm_server.log &

        SERVER_PID=$!
        echo "Server started with PID: $SERVER_PID"
        
        # Wait for server to be ready
        if ! check_server_health; then
            echo "Failed to start server, skipping this configuration"
            cleanup_server
            continue
        fi

    python FunctionChat-Bench/evaluate.py singlecall \
        --input_path FunctionChat-Bench/data/FunctionChat-Singlecall.jsonl \
        --tools_type all \
        --system_prompt_path FunctionChat-Bench/data/system_prompt.txt \
        --temperature 0.1 \
        --model inhouse \
        --base_url http://localhost:8000/v1/ \
        --api_key ${API_KEY} \
        --model_path ${QUANTIZED_MODEL_PATH}
 
    python evaluate.py dialog \
        --input_path FunctionChat-Bench/data/FunctionChat-Dialog.jsonl \
        --system_prompt_path FunctionChat-Bench/data/system_prompt.txt \
        --temperature 0.1 \
        --model inhouse \
        --base_url http://localhost:8000/v1/ \
        --api_key ${API_KEY} \
        --model_path ${QUANTIZED_MODEL_PATH}
    
    mv FunctionChat-Bench/output/* ${OUTPUT_DIR}

    cleanup_server
    SERVER_PID=""
    echo "Waiting a bit before next configuration..."
    sleep 10
done

# Restore FunctionChatBench config with API key and dataset path of lm-evaluation-harness
restore_functionchat_config
restore_dataset_path