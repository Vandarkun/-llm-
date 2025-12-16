#!/bin/bash
# 默认参数
#MODEL_PATH="../../model/merge_model/XiYouJi"
#MODEL_PATH="../../model/merge_model/XiYouJi"
MODEL_PATH="../../model/base_model/Qwen2.5-7B-Instruct"
TEMPLATE="qwen"
DATASET_NAME="xyj_fintune_test"
TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="2048"
INFER_OUTPUT_PATH="../../data/infer/base.jsonl"
INFER_RESULT_PATH="../../data/infer/base_output.json"

# 进入LLaMA-Factory目录（相对路径均相对于LLaMA-Factory根目录）
cd modules/LLaMA-Factory || exit 1

# 模型推理（可选合并）
run_inference() {
    local DATASET=$1
    local SAVE_PATH=$2
    local ADAPTER_PATH=$3
    local MERGED_DIR=$4

    if [ -n "$ADAPTER_PATH" ] && [ -n "$MERGED_DIR" ]; then
        # 更新merge配置文件
        sed -i.bak \
            -e "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|" \
            -e "s|adapter_name_or_path:.*|adapter_name_or_path: $ADAPTER_PATH|" \
            -e "s|template:.*|template: $TEMPLATE|" \
            -e "s|export_dir:.*|export_dir: $MERGED_DIR|" \
            ../../config/llm_merge.yaml

        # 模型合并
        if ! llamafactory-cli export ../../config/llm_merge.yaml; then
            echo "错误：模型合并失败！"
            exit 1
        fi
        mv ../../config/llm_merge.yaml.bak ../../config/llm_merge.yaml
        rm -f ../../config/llm_merge.yaml.bak

        # 使用合并后的模型路径
        MODEL_PATH_TO_USE="$MERGED_DIR"
    else
        # 使用原始模型路径
        MODEL_PATH_TO_USE="$MODEL_PATH"
    fi

    # 开始模型推理
    echo -e "\n=== 开始模型推理 --> $DATASET ==="
    if ! python scripts/vllm_infer.py \
        --model_name_or_path "$MODEL_PATH_TO_USE" \
        --dataset "$DATASET" \
        --template "$TEMPLATE" \
        --save_name "$SAVE_PATH" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_new_tokens "$MAX_NEW_TOKENS"; then
        echo "错误：模型推理失败！"
        exit 1
    fi
}

# -------------------------- 推理预处理 --------------------------
if ! PYTHONPATH="../../" python3 -c "import sys; sys.path.append('../../'); from code.infer import infer_preprocess; infer_preprocess()"; then
    echo "错误：推理预处理失败！"
    exit 1
fi

# -------------------------- 执行推理 --------------------------
# 执行推理
run_inference "$DATASET_NAME" "$INFER_OUTPUT_PATH"

# -------------------------- 推理后处理 --------------------------
if ! PYTHONPATH="../../" python3 -c "import sys; sys.path.append('../../'); from code.infer import infer_postprocess; infer_postprocess('$INFER_OUTPUT_PATH', '$INFER_RESULT_PATH')"; then
    echo "错误：推理后处理失败！"
    exit 1
fi

echo -e "\n=== 推理完成 ===" 