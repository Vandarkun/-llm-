import os
import pandas as pd  # 添加pandas导入
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 从torch.optim导入AdamW
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange  # 添加tqdm导入
import json
from difflib import SequenceMatcher
import matplotlib.pyplot as plt  # 添加matplotlib导入

class TextSimilarityDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=512):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = float(self.labels[idx])
        
        # 编码文本对
        encoding = self.tokenizer(
            text1, 
            text2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

class SimilarityModel(torch.nn.Module):
    def __init__(self, base_model_name, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.similarity = torch.nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        similarity = self.similarity(pooled_output)
        return torch.sigmoid(similarity).squeeze()
    
    def save_model(self, output_dir):
        """保存模型到指定目录
        
        Args:
            output_dir: 输出目录
        """
        # 保存bert模型
        self.bert.save_pretrained(os.path.join(output_dir, "bert"))
        # 保存整个模型的状态字典
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
    
    @classmethod
    def load_model(cls, model_dir, device='cuda'):
        """从指定目录加载模型
        
        Args:
            model_dir: 模型目录
            device: 设备
        """
        # 加载bert模型
        bert_path = os.path.join(model_dir, "bert")
        model = cls(bert_path)
        # 加载模型状态字典
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        model.to(device)
        return model

def prepare_data(data_path):
    """准备训练数据
    
    Args:
        data_path: CSV文件路径，包含input和output列
    """
    # 读取CSV文件
    df = pd.read_csv(data_path)
    
    texts1 = []  # 原始文本
    texts2 = []  # 改写文本
    labels = []  # 相似度标签
    
    # 遍历数据框
    for idx, row in df.iterrows():
        # 添加原文与改写文本对（正样本）
        # 对于正确的改写/翻译，使用随机的高相似度(0.8-1.0)
        texts1.append(row['input'])
        texts2.append(row['output'])
        labels.append(np.random.uniform(0.8, 1.0))  # 随机高相似度
        
        # 对每个样本，随机选择1-2个负样本
        num_negative = np.random.randint(1, 3)
        for _ in range(num_negative):
            # 随机选择另一个样本的改写文本
            random_idx = np.random.randint(0, len(df))
            while random_idx == idx:
                random_idx = np.random.randint(0, len(df))
            
            # 对于随机配对的文本（负样本），使用随机的低相似度(0.0-0.4)
            texts1.append(row['input'])
            texts2.append(df.iloc[random_idx]['output'])
            labels.append(np.random.uniform(0.0, 0.4))  # 随机低相似度
    
    return texts1, texts2, labels

def train_model(
    data_path,
    base_model_name="../model/base_model/chinese-roberta-wwm-ext",
    output_dir="../model/merge_model/similarity_model",
    batch_size=32,  # batch_size
    epochs=10,      # 训练轮数
    learning_rate=2e-5,
    max_length=512,
    use_local_model=True,
    patience=3      # 早停的耐心值
):
    """训练相似度模型
    
    Args:
        data_path: CSV文件路径
        base_model_name: 基础模型名称或本地路径
        output_dir: 输出目录
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        max_length: 最大序列长度
        use_local_model: 是否使用本地模型
        patience: 早停耐心值，连续多少个epoch验证集loss没有改善就停止训练
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据
    print("准备训练数据...")
    texts1, texts2, labels = prepare_data(data_path)
    print(f"生成了 {len(texts1)} 对训练数据")
    
    # 划分训练集和验证集
    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(
        texts1, texts2, labels, test_size=0.1, random_state=42
    )
    
    try:
        # 加载tokenizer和模型
        if use_local_model:
            print(f"从本地加载模型: {base_model_name}")
            if not os.path.exists(base_model_name):
                raise ValueError(f"本地模型路径不存在: {base_model_name}")
        else:
            print(f"从Hugging Face下载模型: {base_model_name}")
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = SimilarityModel(base_model_name)
        model.to(device)
        
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("请先运行 download_model.py 下载模型到本地")
        return None, None
    
    # 创建数据集和数据加载器
    train_dataset = TextSimilarityDataset(
        train_texts1, train_texts2, train_labels, tokenizer, max_length
    )
    val_dataset = TextSimilarityDataset(
        val_texts1, val_texts2, val_labels, tokenizer, max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_loss = float('inf')
    criterion = torch.nn.MSELoss()
    no_improve_epochs = 0  # 记录验证集性能没有改善的轮数
    
    # 用于记录损失值
    train_losses = []
    val_losses = []
    
    # 使用trange显示epoch进度
    for epoch in trange(epochs, desc="Epochs"):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                # 更新进度条显示当前loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型并检查早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            # 保存模型和tokenizer
            model.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")
            no_improve_epochs = 0  # 重置计数器
        else:
            no_improve_epochs += 1
            print(f"Validation loss didn't improve for {no_improve_epochs} epochs")
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存损失曲线图
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print("\nTraining completed!")
    print(f"Loss plot saved to {plot_path}")
    
    # 保存损失数据
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    loss_data_path = os.path.join(output_dir, 'loss_data.json')
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f)
    
    return model, tokenizer

if __name__ == "__main__":
    # 训练模型
    data_path = "../data/finetune/xyj_fintune_train.csv"
    model_name = "../model/base_model/chinese-roberta-wwm-ext"
    output_dir = "../model/merge_model/test"
    
    print("\n开始训练模型...")
    model, tokenizer = train_model(
        data_path=data_path,
        base_model_name=model_name,
        output_dir=output_dir,
        batch_size=64,
        epochs=10,
        learning_rate=2e-5
    ) 