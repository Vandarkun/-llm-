#!/bin/bash
   
# 默认参数
MODEL_PATH="../../model/base_model/Qwen2.5-7B-Instruct"
TEMPLATE="qwen"
LEARNING_RATE="5.0e-5"
NUM_EPOCHS=7
DATASET_NAME="xyj_fintune"
SFT_MODEL_PATH="../../model/llm_adapter/${DATASET_NAME}_adapter"
MERGE_MODEL_PATH="../../model/merge_model/${DATASET_NAME}_merge"
DPO_MODEL_PATH="../../model/llm_adapter/${DATASET_NAME}_preference_adapter"
FINAL_MODEL_PATH="../../model/merge_model/${DATASET_NAME}_final"

# 进入LLaMA-Factory目录（相对路径均相对于LLaMA-Factory根目录）
cd modules/LLaMA-Factory || exit 1

# -------------------------- 微调预处理 --------------------------
echo -e "\n=== 开始微调数据预处理 ==="
if ! PYTHONPATH="../../" python3 -c "import sys; sys.path.append('../../'); from code.finetune import finetune_preprocess; finetune_preprocess()"; then
    echo "错误：微调预处理失败！"
    exit 1
fi
echo "微调数据预处理完成"

# -------------------------- 偏好数据预处理 --------------------------
echo -e "\n=== 开始偏好数据预处理 ==="
if ! PYTHONPATH="../../" python3 -c "import sys; sys.path.append('../../'); from code.finetune import process_preference_dataset; process_preference_dataset('../../data/finetune/XiYouji_Preference.json')"; then
    echo "错误：偏好数据预处理失败！"
    exit 1
fi
echo "偏好数据预处理完成"

# -------------------------- 微调训练 --------------------------
# 备份配置文件
cp ../../config/llm_lora.yaml ../../config/llm_lora.yaml.bak

# 使用sed修改配置文件
sed -i \
    -e "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|" \
    -e "s|template:.*|template: $TEMPLATE|" \
    -e "s|learning_rate:.*|learning_rate: $LEARNING_RATE|" \
    -e "s|num_train_epochs:.*|num_train_epochs: $NUM_EPOCHS|" \
    -e "s|dataset:.*|dataset: $DATASET_NAME|" \
    -e "s|output_dir:.*|output_dir: $SFT_MODEL_PATH|" \
    -e "s|overwrite_output_dir:.*|overwrite_output_dir: $SFT_MODEL_PATH|" \
    ../../config/llm_lora.yaml

# 执行模型训练
if ! llamafactory-cli train ../../config/llm_lora.yaml; then
    echo "错误：模型微调失败！"
    # 恢复配置文件
    mv ../../config/llm_lora.yaml.bak ../../config/llm_lora.yaml
    exit 1
fi

# 训练成功后删除配置文件备份
rm ../../config/llm_lora.yaml.bak

echo -e "\n=== 微调完成 ==="
echo "模型已保存到: $SFT_MODEL_PATH"

# -------------------------- 模型合并 --------------------------
echo -e "\n=== 模型合并 ==="

# 备份合并配置文件
cp ../../config/llm_merge.yaml ../../config/llm_merge.yaml.bak

# 使用sed修改合并配置文件
sed -i \
    -e "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|" \
    -e "s|adapter_name_or_path:.*|adapter_name_or_path: $SFT_MODEL_PATH|" \
    -e "s|export_dir:.*|export_dir: $MERGE_MODEL_PATH|" \
    ../../config/llm_merge.yaml

# 执行模型合并
if ! llamafactory-cli export ../../config/llm_merge.yaml; then
    echo "错误：模型合并失败！"
    # 恢复配置文件
    mv ../../config/llm_merge.yaml.bak ../../config/llm_merge.yaml
    exit 1
fi

# 合并成功后删除配置文件备份
rm ../../config/llm_merge.yaml.bak

echo -e "\n=== 模型合并完成 ==="
echo "合并模型已保存到: $MERGE_MODEL_PATH"

# -------------------------- 偏好数据集微调 --------------------------
echo -e "\n=== 开始偏好数据集微调 ==="

# 备份DPO配置文件
cp ../../config/llm_dpo.yaml ../../config/llm_dpo.yaml.bak

# 使用sed修改DPO配置文件
sed -i \
    -e "s|model_name_or_path:.*|model_name_or_path: $MERGE_MODEL_PATH|" \
    -e "s|template:.*|template: $TEMPLATE|" \
    -e "s|learning_rate:.*|learning_rate: $LEARNING_RATE|" \
    -e "s|num_train_epochs:.*|num_train_epochs: $NUM_EPOCHS|" \
    -e "s|dataset:.*|dataset: ${DATASET_NAME}_preference|" \
    -e "s|output_dir:.*|output_dir: $DPO_MODEL_PATH|" \
    -e "s|overwrite_output_dir:.*|overwrite_output_dir: $DPO_MODEL_PATH|" \
    ../../config/llm_dpo.yaml

# 执行偏好数据集微调
if ! llamafactory-cli train ../../config/llm_dpo.yaml; then
    echo "错误：偏好数据集微调失败！"
    # 恢复配置文件
    mv ../../config/llm_dpo.yaml.bak ../../config/llm_dpo.yaml
    exit 1
fi

# 微调成功后删除配置文件备份
rm ../../config/llm_dpo.yaml.bak

echo -e "\n=== 偏好数据集微调完成 ==="
echo "DPO模型已保存到: $DPO_MODEL_PATH"

# -------------------------- DPO模型合并 --------------------------
echo -e "\n=== DPO模型合并 ==="

# 备份合并配置文件
cp ../../config/llm_merge.yaml ../../config/llm_merge.yaml.bak

# 使用sed修改合并配置文件
sed -i \
    -e "s|model_name_or_path:.*|model_name_or_path: $MERGE_MODEL_PATH|" \
    -e "s|adapter_name_or_path:.*|adapter_name_or_path: $DPO_MODEL_PATH|" \
    -e "s|export_dir:.*|export_dir: $FINAL_MODEL_PATH|" \
    ../../config/llm_merge.yaml

# 执行DPO模型合并
if ! llamafactory-cli export ../../config/llm_merge.yaml; then
    echo "错误：DPO模型合并失败！"
    # 恢复配置文件
    mv ../../config/llm_merge.yaml.bak ../../config/llm_merge.yaml
    exit 1
fi

# 合并成功后删除配置文件备份
rm ../../config/llm_merge.yaml.bak

echo -e "\n=== DPO模型合并完成 ==="
echo "最终模型已保存到: $FINAL_MODEL_PATH"
echo -e "\n=== 整个微调流程已全部完成 ==="
