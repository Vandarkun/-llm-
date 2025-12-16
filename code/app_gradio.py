import os
import base64
import gradio as gr
from tools import ChatModelManager, ApiChatManager

# 创建聊天模型管理器实例
chat_manager = ChatModelManager()

# 将图片转换为base64编码
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片出错: {e}")
        return None

# 获取可用模型列表
def get_available_models():
    try:
        return chat_manager.get_available_models()
    except Exception as e:
        print(f"读取模型列表出错: {e}")
        return ["xiyouji"]  # 默认返回

# 获取可用API列表
def get_available_apis():
    try:
        apis = ApiChatManager.get_available_apis()
        return apis if apis else ["无可用API"]
    except Exception as e:
        print(f"读取API列表出错: {e}")
        return ["无可用API"]

# 加载模型
def load_model(model_name):
    try:
        if not model_name:
            status_text = "请选择一个模型"
            return status_text, status_text
            
        result = chat_manager.load_model(model_name)
        if result["status"] == "success":
            status_text = f"✅ 模型 '{model_name}' 加载成功"
            return status_text, status_text
        else:
            status_text = f"❌ 模型加载失败: {result.get('message', '未知错误')}"
            return status_text, status_text
    except Exception as e:
        status_text = f"❌ 模型加载出错: {str(e)}"
        return status_text, status_text

# 卸载模型
def unload_model():
    try:
        result = chat_manager.unload_model()
        if result["status"] == "success":
            status_text = "✅ 模型已成功卸载"
            return status_text, status_text
        else:
            status_text = f"❌ 模型卸载失败: {result.get('message', '未知错误')}"
            return status_text, status_text
    except Exception as e:
        status_text = f"❌ 模型卸载出错: {str(e)}"
        return status_text, status_text

# 获取模型状态
def get_model_status():
    try:
        result = chat_manager.get_status()
        model_info = result.get("model_info", {})
        is_loaded = model_info.get("is_loaded", False)
        model_path = model_info.get("model_path", "未加载")
        device = model_info.get("device", "")  # 设备默认为空字符串
        
        if is_loaded:
            status_text = f"✅ 模型已加载 | 设备: {device}"
            return status_text, model_path
        else:
            status_text = f"❌ 模型未加载"
            return status_text, "未加载"
    except Exception as e:
        status_text = f"❌ 获取模型状态出错: {str(e)}"
        return status_text, "错误"

# 检查模型是否已加载
def is_model_loaded():
    try:
        status = chat_manager.get_status()
        return status.get("model_info", {}).get("is_loaded", False)
    except:
        return False

# 处理聊天消息
def chat(message, history):
    if not message:
        return history
        
    try:
        # 检查模型是否已加载
        if not is_model_loaded():
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ 模型未加载，请先加载模型"}]
            
        # 发送消息获取回复
        result = chat_manager.chat(message)
        if result["status"] == "success":
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result["response"]}]
        else:
            error_msg = f"❌ 生成回复失败: {result.get('message', '未知错误')}"
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]
    except Exception as e:
        error_msg = f"❌ 生成回复出错: {str(e)}"
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]

# 处理API聊天并润色
def api_chat_and_polish(message, api_name, history):
    if not message:
        return history
        
    try:
        # 检查模型是否已加载
        if not is_model_loaded():
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ 模型未加载，请先加载模型"}]
        
        # 检查API选择
        if api_name == "无可用API":
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ 请先在config/api_config.json中配置API"}]
        
        # 如果是第一次对话，直接创建新的API历史
        if not hasattr(api_chat_and_polish, "api_history"):
            api_chat_and_polish.api_history = []
        
        # 调用API获取回复并润色（使用API历史）
        result = ApiChatManager.polish_api_response(message, api_name, chat_manager, api_chat_and_polish.api_history)
        
        if result["status"] == "success":
            original_response = result["original"]
            polished_response = result["polished"]
            
            # 更新API历史（只包含干净的对话）
            api_chat_and_polish.api_history.append({"role": "user", "content": message})
            api_chat_and_polish.api_history.append({"role": "assistant", "content": original_response})
            
            # 更新展示历史（包含格式化内容）
            display_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"【API({api_name})原始回复】\n{original_response}\n\n【模型风格润色】\n{polished_response}"}
            ]
            
            # 返回展示历史
            return display_history
        else:
            # 显示错误信息
            error_msg = f"❌ 生成回复失败: {result.get('message', '未知错误')}"
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]
    except Exception as e:
        error_msg = f"❌ 生成回复出错: {str(e)}"
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]

# 清除聊天历史
def clear_chat_history():
    # 同时清除API历史
    if hasattr(api_chat_and_polish, "api_history"):
        api_chat_and_polish.api_history = []
    return []

# 更新所有状态框
def update_status_boxes():
    status_text, model_path = get_model_status()
    return status_text, status_text, model_path, model_path

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="2021216915-王大坤", theme=gr.themes.Soft()) as demo:
        # 使用base64编码图片并创建居中显示的标题栏
        try:
            img_base64 = image_to_base64("xiaohui.jpg")
            if img_base64:
                gr.Markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
                    <div style="display: flex; align-items: center;">
                        <img src="data:image/jpeg;base64,{img_base64}" alt="校徽" style="height: 35px; width: 35px; margin-right: 10px;">
                        <h1 style="margin: 0; padding: 0; font-size: 24px;">基于微调大语言模型的文本风格迁移工具</h1>
                    </div>
                </div>
                <hr style="margin-top: 10px;"/>
                """)
            else:
                # 图片加载失败时只显示居中标题
                gr.Markdown("""
                <div style="text-align: center;">
                    <h1 style="margin: 10px 0; font-size: 24px;">基于微调大语言模型的文本风格迁移工具</h1>
                </div>
                <hr/>
                """)
        except Exception as e:
            # 出错时只显示标题
            print(f"加载图片出错: {e}")
            gr.Markdown("""
            <div style="text-align: center;">
                <h1 style="margin: 10px 0; font-size: 24px;">基于微调大语言模型的文本风格迁移工具</h1>
            </div>
            <hr/>
            """)
        
        # 获取可用模型和API列表
        available_models = get_available_models()
        available_apis = get_available_apis()
        
        # 创建标签页
        with gr.Tabs() as tabs:
            # 标签页1: 模型直接聊天
            with gr.TabItem("模型聊天", id=0):
                with gr.Row():
                    # 左侧对话区域
                    with gr.Column(scale=2):
                        chat1_bot = gr.Chatbot(height=500, label="对话", type="messages")
                        
                        with gr.Row():
                            chat1_msg = gr.Textbox(
                                placeholder="在这里输入你的消息...",
                                show_label=False,
                                scale=9
                            )
                            chat1_send_btn = gr.Button("发送", scale=1)
                    
                    # 右侧控制区域
                    with gr.Column(scale=1):
                        # 状态信息区域 - 状态和设备合并到一行
                        status_box1 = gr.Textbox(label="状态", value="❌ 模型未加载", lines=1, interactive=False)
                        
                        # 模型路径 - 单行显示，允许滑动
                        model_path_box1 = gr.Textbox(
                            label="模型路径", 
                            value="未加载", 
                            lines=1, 
                            max_lines=1,
                            interactive=False,
                            show_copy_button=True  # 添加复制按钮方便查看完整路径
                        )
                        
                        gr.Markdown("<hr style='margin: 10px 0'/>")
                        
                        # 模型控制区域
                        model_dropdown = gr.Dropdown(
                            available_models,
                            label="选择模型",
                            value=available_models[0] if available_models else None
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button("加载模型", variant="primary")
                            unload_btn = gr.Button("卸载模型")
                        
                        chat1_clear_btn = gr.Button("清除对话")
            
            # 标签页2: API+模型润色
            with gr.TabItem("API对话（模型润色）", id=1):
                with gr.Row():
                    # 左侧对话区域
                    with gr.Column(scale=2):
                        chat2_bot = gr.Chatbot(height=500, label="对话", type="messages")
                        
                        with gr.Row():
                            chat2_msg = gr.Textbox(
                                placeholder="在这里输入你的消息...",
                                show_label=False,
                                scale=9
                            )
                            chat2_send_btn = gr.Button("发送", scale=1)
                    
                    # 右侧控制区域
                    with gr.Column(scale=1):
                        # 状态信息区域 - 状态和设备合并到一行
                        status_box2 = gr.Textbox(label="状态", value="❌ 模型未加载", lines=1, interactive=False)
                        
                        # 模型路径 - 单行显示，允许滑动
                        model_path_box2 = gr.Textbox(
                            label="模型路径", 
                            value="未加载", 
                            lines=1, 
                            max_lines=1,
                            interactive=False,
                            show_copy_button=True  # 添加复制按钮方便查看完整路径
                        )
                        
                        gr.Markdown("<hr style='margin: 10px 0'/>")
                        
                        # 模型控制区域
                        model_dropdown2 = gr.Dropdown(
                            available_models,
                            label="选择模型",
                            value=available_models[0] if available_models else None
                        )
                        
                        with gr.Row():
                            load_btn2 = gr.Button("加载模型", variant="primary")
                            unload_btn2 = gr.Button("卸载模型")
                        
                        gr.Markdown("<hr style='margin: 10px 0'/>")
                        
                        # API选择
                        api_dropdown = gr.Dropdown(
                            available_apis,
                            label="选择API",
                            value=available_apis[0] if available_apis else "无可用API"
                        )
                        chat2_clear_btn = gr.Button("清除对话")
        
        # 设置事件处理 - 标签页1
        load_btn.click(
            load_model, 
            inputs=[model_dropdown], 
            outputs=[status_box1, status_box2]
        ).then(
            update_status_boxes, 
            inputs=[], 
            outputs=[status_box1, status_box2, model_path_box1, model_path_box2]
        )
        
        unload_btn.click(
            unload_model, 
            inputs=[], 
            outputs=[status_box1, status_box2]
        ).then(
            update_status_boxes, 
            inputs=[], 
            outputs=[status_box1, status_box2, model_path_box1, model_path_box2]
        )
        
        chat1_msg.submit(chat, inputs=[chat1_msg, chat1_bot], outputs=[chat1_bot]).then(
            lambda: "", inputs=[], outputs=[chat1_msg]
        )
        chat1_send_btn.click(chat, inputs=[chat1_msg, chat1_bot], outputs=[chat1_bot]).then(
            lambda: "", inputs=[], outputs=[chat1_msg]
        )
        chat1_clear_btn.click(clear_chat_history, inputs=[], outputs=[chat1_bot])
        
        # 设置事件处理 - 标签页2
        load_btn2.click(
            load_model, 
            inputs=[model_dropdown2], 
            outputs=[status_box1, status_box2]
        ).then(
            update_status_boxes, 
            inputs=[], 
            outputs=[status_box1, status_box2, model_path_box1, model_path_box2]
        )
        
        unload_btn2.click(
            unload_model, 
            inputs=[], 
            outputs=[status_box1, status_box2]
        ).then(
            update_status_boxes, 
            inputs=[], 
            outputs=[status_box1, status_box2, model_path_box1, model_path_box2]
        )
        
        chat2_msg.submit(api_chat_and_polish, inputs=[chat2_msg, api_dropdown, chat2_bot], outputs=[chat2_bot]).then(
            lambda: "", inputs=[], outputs=[chat2_msg]
        )
        chat2_send_btn.click(api_chat_and_polish, inputs=[chat2_msg, api_dropdown, chat2_bot], outputs=[chat2_bot]).then(
            lambda: "", inputs=[], outputs=[chat2_msg]
        )
        chat2_clear_btn.click(clear_chat_history, inputs=[], outputs=[chat2_bot])
        
        # 在启动时进行一次状态检查
        demo.load(
            update_status_boxes, 
            inputs=[], 
            outputs=[status_box1, status_box2, model_path_box1, model_path_box2]
        )
        
    return demo

# 启动应用
if __name__ == "__main__":
    demo = create_interface()
    port = int(os.environ.get("PORT", 7861))
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port, 
        share=False, 
        allowed_paths=['.', './xiaohui.jpg']
    ) 