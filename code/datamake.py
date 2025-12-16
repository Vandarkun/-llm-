import json
import pandas as pd
import os

# 加载prompt模板
def load_prompt_template(prompt_type=None, config_path="../config/llm_prompt.json"):
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

# 处理微调数据集JSON文件，保留前N个条目
def truncate_json(input_file: str, output_file: str = None, limit: int = 500) -> None:
    """
    处理微调数据集JSON文件，保留前N个条目
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，如果为None则覆盖原文件
        limit: 保留的条目数量，默认500
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是列表类型
        if not isinstance(data, list):
            raise ValueError("数据集必须是列表类型")
            
        # 截取前limit个条目
        truncated_data = data[:limit]
        
        # 如果没有指定输出文件，则覆盖原文件
        output_file = output_file or input_file
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(truncated_data, f, ensure_ascii=False, indent=2)
            
        print(f"成功处理数据集，从 {len(data)} 条数据中保留了前 {len(truncated_data)} 条")
        
    except Exception as e:
        print(f"处理JSON文件时出错: {str(e)}")

# 偏好数据集处理
def process_preference_dataset(input_file: str, output_file: str = None, limit: int = 100000000) -> None:

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
            'input': item.get('instruction', ''),
            'chosen': item['output'][0],  # 第一句作为chosen
            'rejected': item['output'][1]   # 第二句作为reject
        }
        processed_data.append(processed_item)

    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

# 拆分数据集
def split_dataset(input_file, train_ratio=0.8, output_dir="../data/finetune"):
    """
    将CSV文件按比例拆分为训练集和测试集
    Args:
        input_file: 输入的CSV文件路径
        train_ratio: 训练集占比，默认0.8
        output_dir: 输出目录，默认"../data/finetune"
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算训练集大小
    train_size = int(len(df) * train_ratio)
    
    # 拆分数据集
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # 获取文件名（不含路径和扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 构造输出文件路径
    train_file = os.path.join(output_dir, f"{base_name}_train.csv")
    test_file = os.path.join(output_dir, f"{base_name}_test.csv")
    
    # 保存拆分后的文件
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"数据集拆分完成：")
    print(f"训练集大小：{len(train_df)}，保存至：{train_file}")
    print(f"测试集大小：{len(test_df)}，保存至：{test_file}")


if __name__ == "__main__":
    # split_dataset("../data/finetune/xyj_fintune.csv", train_ratio=0.8, output_dir="../data/finetune")
    process_preference_dataset(input_file="/root/autodl-tmp/wdk_work/data/finetune/XiYouji_Preference.json", output_file="/root/autodl-tmp/wdk_work/modules/LLaMA-Factory/dataxyj_fintune_train_preference.json")