from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

def chat_yi(message, model_path="../model/merge_model/XiYouJi"):
    """简单的对话函数
    
    Args:
        message: 用户输入的消息
        model_path: 模型路径，默认为相对路径
        
    Returns:
        str: 模型的回复
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 转换为绝对路径
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    print(f"Loading model from: {model_path}")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    ).eval()
    
    model.to(device)
    print(f"Model loaded and moved to: {next(model.parameters()).device}")
    
    # 处理输入
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt'
    )
    input_ids = input_ids.to(device)
    
    # 生成回复
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=2048
    )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response

def qwen_chat(user_input, model_path="../model/merge_model/xyj_fintune_merge"):
    """最简单的Qwen对话函数（支持本地模型）
    参数:
        user_input: 用户输入的问题 (字符串)
        model_path: 模型路径，可以是：
                   - 官方模型名称（如"Qwen/Qwen2.5-7B-Instruct"）
                   - 本地路径（如"./models/Qwen2.5-7B-Instruct"）
    返回:
        模型的回答 (字符串)
    """
    # 获取模型文件绝对路径
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))

    # 配置4-bit量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 1. 加载模型（自动识别本地或远程）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,  # 使用新的量化配置
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 构建对话格式
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    # 使用Qwen的对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    
    # 3. 生成回答
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 4. 解码输出并清理
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 清理输出中的系统提示和多余文本
    response = response.replace("[System Prompt]:", "").strip()
    response = response.replace("你当前的角色是来自阿里云的超大规模语言模型，", "")
    response = response.replace("您需要根据上下文提供合适的回答。", "")
    response = response.replace("请继续之前的情景对话：", "")
    
    # 如果响应以引号开始，去掉引号
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    
    return response.strip()

if __name__ == "__main__":
    response = qwen_chat("三藏听到这番话后说道：算了吧。")
    print("Output:", response)
    # response = qwen_chat("听到这个消息，老魔虽然并不害怕，但却感到一丝惊恐。他只能强装镇定地喊道：兄弟们，别害怕，拿我的药酒来，我喝几口下去，把那只猴子药死算了！")
    # print("Output:", response)
    # 听到这个消息，老魔虽然并不害怕，但却感到一丝惊恐。他只能强装镇定地喊道：兄弟们，别害怕，拿我的药酒来，我喝几口下去，把那只猴子药死算了！