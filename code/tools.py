import os
import gc
import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List

# 获取模型配置文件路径
def get_config_path() -> str:
    """获取模型配置文件的默认路径"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'model_path.json')

# 获取API配置文件路径
def get_api_config_path() -> str:
    """获取API配置文件的默认路径"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'api_config.json')

# 读取模型配置文件
def read_json_config(json_path: str = None) -> Dict:
    """
    读取模型配置文件
    
    Args:
        json_path: JSON文件路径，默认为None时会使用默认配置文件路径
    """
    try:
        # 如果没有提供json_path，使用默认路径
        if json_path is None:
            json_path = get_config_path()
            
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {json_path}: {str(e)}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Error reading config file: {str(e)}")

# 读取API配置文件
def read_api_config() -> Dict:
    """读取API配置文件"""
    try:
        api_config_path = get_api_config_path()
        with open(api_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"API配置文件不存在: {get_api_config_path()}")
        return {}
    except Exception as e:
        print(f"读取API配置文件出错: {str(e)}")
        return {}

# 模型信息类
class ModelInfo:
    def __init__(self, device: str = "cpu", model_path: str = None, is_loaded: bool = False):
        self.device = device
        self.model_path = model_path
        self.is_loaded = is_loaded
    
    def to_dict(self) -> Dict:
        return {
            "device": self.device,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded
        }

# 聊天模型管理类
class ChatModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model_info = ModelInfo(device=cls._instance.device)
        return cls._instance
    
    # 获取所有可用的模型名称
    def get_available_models(self) -> List[str]:
        """获取所有可用的模型名称"""
        config = read_json_config()
        return list(config.keys())
    
    # 根据模型名称获取模型路径
    def get_model_path(self, model_name: str) -> str:
        """根据模型名称获取模型路径"""
        model_name = model_name.lower()
        config = read_json_config()
        if model_name not in config:
            available_models = ", ".join(config.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        return config[model_name]
    
    # 加载模型
    def load_model(self, model_name: str) -> Dict:
        """
        加载模型
        
        Args:
            model_name: 模型名称，必须在配置文件中存在
        """
        try:
            # 从配置获取模型路径
            model_path = self.get_model_path(model_name)
            
            # 如果模型已经加载且路径相同，直接返回
            if self.model is not None and self.model_info.model_path == model_path:
                return {
                    "status": "success",
                    "message": "Model already loaded",
                    "model_info": self.model_info.to_dict()
                }
            
            # 如果有其他模型已加载，先卸载
            if self.model is not None:
                self.unload_model()
            
            # 加载新模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            ).eval()
            
            self.model.to(self.device)
            self.model_info = ModelInfo(
                device=self.device,
                model_path=model_path,
                is_loaded=True
            )
            
            return {
                "status": "success",
                "message": f"Model '{model_name}' loaded successfully",
                "model_info": self.model_info.to_dict()
            }
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    # 卸载模型
    def unload_model(self) -> Dict:
        """卸载模型"""
        try:
            if self.model is None:
                return {
                    "status": "success",
                    "message": "No model is currently loaded",
                    "model_info": self.model_info.to_dict()
                }
            
            if self.device == "cuda":
                self.model.cpu()
            
            del self.model
            self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.model_info = ModelInfo(device=self.device)
            
            return {
                "status": "success",
                "message": "Model unloaded successfully",
                "model_info": self.model_info.to_dict()
            }
            
        except Exception as e:
            raise Exception(f"Failed to unload model: {str(e)}")
    
    # 生成回复
    def chat(self, message: str) -> Dict:
        """生成回复"""
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Please load model first.")
            
        try:
            # 处理输入
            messages = [{"role": "user", "content": message}]
            input_ids = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
            input_ids = input_ids.to(self.device)
            
            # 生成回复
            output_ids = self.model.generate(
                input_ids,
                do_sample=False,
                repetition_penalty=1.2,
                max_new_tokens=2048
            )
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            return {
                "status": "success",
                "message": message,
                "response": response,
                "model_info": self.model_info.to_dict()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": message,
                "error": str(e),
                "model_info": self.model_info.to_dict()
            }
            
    def polish_text(self, text: str) -> Dict:
        """
        用当前加载的模型风格润色文本
        
        Args:
            text: 要润色的原始文本
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Please load model first.")
            
        try:
            # 直接使用原始文本作为输入，不添加任何额外指令
            messages = [{"role": "user", "content": text}]
            input_ids = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
            input_ids = input_ids.to(self.device)
            
            # 生成回复
            output_ids = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                max_new_tokens=4096
            )
            polished_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            return {
                "status": "success",
                "original": text,
                "polished": polished_text
            }
            
        except Exception as e:
            raise Exception(f"Failed to polish text: {str(e)}")
    
    def get_status(self) -> Dict:
        """获取模型状态"""
        return {
            "status": "success",
            "model_info": self.model_info.to_dict()
        }

# API对话上下文管理器
class ApiConversationManager:
    _conversations = {}
    
    @classmethod
    def get_conversation(cls, session_id: str) -> List:
        """获取指定会话的上下文"""
        if session_id not in cls._conversations:
            cls._conversations[session_id] = []
        return cls._conversations[session_id]
    
    @classmethod
    def add_message(cls, session_id: str, role: str, content: str):
        """添加消息到会话上下文"""
        if session_id not in cls._conversations:
            cls._conversations[session_id] = []
        cls._conversations[session_id].append({"role": role, "content": content})
    
    @classmethod
    def clear_conversation(cls, session_id: str):
        """清空指定会话的上下文"""
        cls._conversations[session_id] = []

# API调用相关函数
class ApiChatManager:
    @staticmethod
    def get_available_apis() -> List[str]:
        """获取所有可用的API名称"""
        api_config = read_api_config()
        return list(api_config.keys())

    @staticmethod
    def chat_with_api(message: str, api_name: str = None, history: List = None, session_id: str = "default") -> str:
        """
        调用外部API获取回复（支持多轮对话）
        
        Args:
            message: 用户消息
            api_name: API名称，必须在配置文件中存在
            history: 对话历史，格式为[(user_msg, bot_msg), ...]
            session_id: 会话ID，用于区分不同的对话
        
        Returns:
            API的回复内容
        """
        try:
            if not api_name:
                return f"这是模拟回复: 您问了 '{message}'。请在配置文件中设置API或选择一个有效的API。"
                
            # 读取API配置
            api_config = read_api_config()
            if api_name not in api_config:
                available_apis = ", ".join(api_config.keys())
                return f"API '{api_name}' 不存在。可用的API: {available_apis or '无'}"
                
            api_info = api_config[api_name]
            api_url = api_info.get("api_url")
            api_key = api_info.get("api_key")
            
            if not api_url or not api_key:
                return f"API配置不完整，请检查 {api_name} 的URL和密钥配置。"
            
            # 准备消息历史
            messages = []
            
            # 如果有对话历史，转换格式添加到messages中
            if history:
                for msg in history:
                    if "role" in msg and "content" in msg:
                        # 直接使用角色和内容，不尝试解析复杂格式
                        role = msg["role"]
                        content = msg["content"]
                        messages.append({"role": role, "content": content})
                    elif isinstance(msg, tuple) and len(msg) == 2:
                        # 旧格式的元组 (user_msg, bot_msg)
                        user_msg, bot_msg = msg
                        if user_msg:
                            messages.append({"role": "user", "content": user_msg})
                        if bot_msg:
                            messages.append({"role": "assistant", "content": bot_msg})
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": message})
            
            # 以下是针对DeepSeek API的示例，可根据不同API调整
            if api_name == "deepseek":
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": messages,  # 包含完整的对话历史
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                response = requests.post(f"{api_url}/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                
                api_response = response.json()["choices"][0]["message"]["content"]
                return api_response
            
            # 其他API的实现可以在这里添加
            # elif api_name == "openai":
            #     headers = {...}
            #     payload = {"messages": messages, ...}
            #     ...
            
            # 如果未实现对应API的处理，使用模拟响应，同时考虑对话上下文
            context_info = f"[对话历史: {len(messages)-1}条]" if len(messages) > 1 else ""
            mock_response = f"这是API '{api_name}' 的模拟回复 {context_info}: 您问了 '{message}'。这是一个模拟多轮对话的回复。"
            return mock_response
            
        except requests.exceptions.RequestException as e:
            return f"API请求失败: {str(e)}"
        except Exception as e:
            return f"API调用出错: {str(e)}"
    
    @staticmethod
    def polish_api_response(message: str, api_name: str = None, model_manager: ChatModelManager = None, history: List = None, session_id: str = "default") -> Dict:
        """
        调用API获取回复并用当前模型风格润色（支持多轮对话）
        
        Args:
            message: 用户消息
            api_name: API名称
            model_manager: 模型管理器实例
            history: 对话历史
            session_id: 会话ID
        
        Returns:
            包含原始和润色后回复的字典
        """
        try:
            # 检查模型是否已加载
            if model_manager is None or model_manager.model is None:
                return {
                    "status": "error",
                    "message": "Model not loaded. Please load model first.",
                    "original": "",
                    "polished": ""
                }
            
            # 调用API获取原始回复（带历史上下文）
            original_response = ApiChatManager.chat_with_api(message, api_name, history, session_id)
            
            # 检查API返回的响应是否为空或错误消息
            if not original_response or original_response.startswith("API请求失败") or original_response.startswith("API调用出错"):
                return {
                    "status": "error",
                    "message": original_response or "API返回空响应",
                    "original": original_response,
                    "polished": ""
                }
            
            # 用模型润色回复
            polish_result = model_manager.polish_text(original_response)
            
            # 检查润色结果
            if not polish_result.get("polished"):
                return {
                    "status": "warning",
                    "message": "润色结果为空，返回原始响应",
                    "original": original_response,
                    "polished": "润色失败，请查看原始回复"
                }
            
            return {
                "status": "success",
                "original": original_response,
                "polished": polish_result["polished"]
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "original": "",
                "polished": ""
            }

# 数据处理相关函数
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

if __name__ == "__main__":
    # 测试加载模型
    try:
        manager = ChatModelManager()
        result = manager.load_model("xiyouji")
        print(f"加载模型结果: {result}")
        
        # 测试聊天
        response = manager.chat("你好，给我讲个故事")
        print(f"回复: {response['response']}")
        
        # 测试润色
        polish_result = manager.polish_text("今天天气很好，我们去公园玩吧！")
        print(f"润色结果: {polish_result['polished']}")
        
        # 卸载模型
        manager.unload_model()
        print("模型已卸载")
    except Exception as e:
        print(f"出错: {e}") 