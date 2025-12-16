import pandas as pd
import json
import os

# 更新数据集信息
def update_dataset_info(dataset_name, file_name):
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
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    
    # 保存更新后的信息
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

# 推理预处理
def infer_preprocess(source_path="../../data/infer/xyj_fintune_test.csv", dataset_name="xyj_fintune_test"):
    """预处理推理数据
    
    Args:
        source_path: 源数据CSV文件路径
        dataset_name: 输出数据集名称
        
    Returns:
        str: 生成的数据集文件路径
    """
    source_data = pd.read_csv(source_path)
    llama_dir = "data"  # 相对于 LLaMA-Factory 目录
    process_dir = "../../data/infer"  # 相对于当前目录
    file_name = f"{dataset_name}.json"
    
    # 构建推理数据
    dataset = []
    for _, row in source_data.iterrows():
        item = {
            "instruction": "将以下文本转换为《西游记》原著风格，使用明代白话文和章回体句式：",
            "input": row["input"],
            "output": row["output"]
        }
        dataset.append(item)
    
    # 保存到两个位置
    os.makedirs(process_dir, exist_ok=True)
    with open(f"{process_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    os.makedirs(llama_dir, exist_ok=True)
    with open(f"{llama_dir}/{file_name}", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 更新数据集信息
    update_dataset_info(dataset_name, file_name)

# 推理后处理
def infer_postprocess(infer_path="../../data/infer/test.jsonl", output_path="../../data/infer/test_output.json"):
    """后处理推理结果
    
    Args:
        finetune_test_path: 微调测试数据JSON文件路径
        infer_path: 推理结果JSONL文件路径
        output_path: 输出文件路径
        
    Returns:
        None
    """
    
    finetune_test_path="../../data/infer/xyj_fintune_test.json"

    try:
        # 读取微调测试数据
        with open(finetune_test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        # 读取推理结果（JSONL格式）
        predictions = []
        with open(infer_path, 'r', encoding='utf-8') as f:
            for line in f:
                pred = json.loads(line)
                predictions.append(pred['predict'])
                
        # 确保数据长度匹配
        if len(test_data) != len(predictions):
            raise ValueError(f"数据长度不匹配: 测试数据 {len(test_data)} 条，预测结果 {len(predictions)} 条")
            
        # 组合结果
        results = []
        for test_item, pred in zip(test_data, predictions):
            result = {
                "base": test_item["input"],        # 原始输入文本
                "result": test_item["output"],     # 期望输出
                "predict": pred                    # 模型预测结果
            }
            results.append(result)
            
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"处理完成，共处理 {len(results)} 条数据")
        print(f"结果已保存至: {output_path}")
        
    except FileNotFoundError as e:
        print(f"文件不存在: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
    except Exception as e:
        print(f"处理出错: {str(e)}")


if __name__ == "__main__":
    None

    
