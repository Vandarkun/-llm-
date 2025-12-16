import pandas as pd
import json
import os

# 加载prompt模板
def load_prompt_template(prompt_type=None, config_path="../../config/llm_prompt.json"):
    """加载prompt模板
    
    Args:
        prompt_type: prompt类型名称
        config_path: prompt配置文件路径
        
    Returns:
        str: prompt模板字符串
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            if prompt_type and prompt_type in prompts:
                return prompts[prompt_type]
            return prompts.get("default", "Human: {text}\nAssistant:")
    except FileNotFoundError:
        print(f"找不到prompt配置文件：{config_path}")
        return "Human: {text}\nAssistant:"
    except json.JSONDecodeError:
        print(f"prompt配置文件格式错误：{config_path}")
        return "Human: {text}\nAssistant:"
    except Exception as e:
        print(f"加载prompt配置文件时发生未知错误：{e}")
        return "Human: {text}\nAssistant:"

# 更新数据集信息
def update_dataset_info(dataset_name, file_name, is_preference=False):
    """更新数据集信息
    
    Args:
        dataset_name: 数据集名称
        file_name: 文件名
        
    Returns:
            None
    """
    info_path = "data/dataset_info.json"  # 相对于 LLaMA-Factory 目录
    
    # 读取现有信息
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        dataset_info = {}
    
    # 更新信息
    if not is_preference:
        dataset_info[dataset_name] = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    else:
        dataset_info[dataset_name] = {
            "file_name": file_name,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    
    # 保存更新后的信息
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

# 微调预处理
def finetune_preprocess(source_path="../../data/finetune/xyj_fintune_train.csv", dataset_name="xyj_fintune", prompt_type="prompt_xyj"):
    """预处理微调数据
    
    Args:
        source_path: 源数据CSV文件路径
        dataset_name: 输出数据集名称
        prompt_type: prompt类型
        
    Returns:
        str: 生成的数据集文件路径
    """
    source_data = pd.read_csv(source_path)
    llama_dir = "data"  # 相对于 LLaMA-Factory 目录
    process_dir = "../../data/finetune"  # 相对于 LLaMA-Factory 目录
    prompt = load_prompt_template(prompt_type)
    file_name = f"{dataset_name}.json"
    
    # 构建训练数据
    dataset = []
    for _, row in source_data.iterrows():
        # item = {
        #     "instruction": prompt,
        #     "input": row["input"],
        #     "output": row["output"]
        # }
        item = {
            "instruction": prompt,
            "input": row["input"],
            "output": row["output"]
        }
        dataset.append(item)
    
    # 保存为JSON文件
    with open(f"{process_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # 保存为JSON文件
    with open(f"{llama_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    

    # 更新数据集信息
    update_dataset_info(dataset_name, file_name)

# 偏好数据集处理
def process_preference_dataset(input_file: str, dataset_name: str = "xyj_fintune_preference", limit: int = 100000000) -> None:
    """处理偏好数据集
    
    Args:
        input_file: 输入文件路径
        dataset_name: 数据集名称
        limit: 数据条数限制
        
    Returns:
        None
    """
    prompt = load_prompt_template(prompt_type="prompt_xyj")

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    processed_data = []
    for item in data[:limit]:
        # 确保output字段有至少两个句子
        if 'output' not in item or len(item['output']) < 2:
            continue
            
        processed_item = {
            'instruction': prompt,
            'input': item['instruction'],
            'chosen': item['output'][0],  # 第一句作为chosen
            'rejected': item['output'][1]   # 第二句作为reject
        }
        processed_data.append(processed_item)

    # 设置保存路径
    llama_dir = "data"  # 相对于 LLaMA-Factory 目录
    process_dir = "../../data/finetune"  # 相对于当前目录
    file_name = f"{dataset_name}.json"
    
    # 保存到两个位置
    os.makedirs(process_dir, exist_ok=True)
    with open(f"{process_dir}/{file_name}", 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    os.makedirs(llama_dir, exist_ok=True)
    with open(f"{llama_dir}/{file_name}", 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 更新数据集信息
    update_dataset_info(dataset_name, file_name, is_preference=True)

if __name__ == "__main__":
    pass
