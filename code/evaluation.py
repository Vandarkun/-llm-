import torch
import numpy as np
from transformers import AutoTokenizer
from train_similarity_model import SimilarityModel
import re
import Levenshtein
from difflib import SequenceMatcher
import json
from tqdm import tqdm

class MultiDimensionEvaluator:
    """多维度文本改写评估器"""
    
    def __init__(self, model_path="../model/merge_model/similarity_model", device=None):
        """初始化评估器
        
        Args:
            model_path (str): 相似度模型路径
            device: 设备 (None则自动选择)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 加载模型和tokenizer
        try:
            self.model = SimilarityModel.load_model(model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"警告: 模型加载失败: {str(e)}")
            print("将只使用基于规则的评估方法")
            self.model_loaded = False
        
        # 文言文特征词汇 - 统一定义，扩充词表
        self.xiyou_markers = {
            # 1. 语气助词（口语化+文言混合）
            'particles': [
                '哩', '耶', '呵', '罢', '也', '哉', '么', '呀', '哟', '咳', '哎', '咦', '哦', '哇', 
                '咧', '咯', '嗏', '唵', '呔', '噫', '呜呼', '乎', '兮', '耳', '尔', '咧', '啰', '唩',
                '呦', '嘞', '呗', '啵', '嗯', '嗨', '哼', '呸', '呵呀', '哎哟', '啊呀', '哇呀'
            ],
            
            # 2. 动词（战斗/法术/日常）
            'verbs': [
                # 战斗动作
                '抡', '劈', '砍', '剁', '刺', '挑', '扫', '砸', '捣', '戳', '擒', '拿', '缚', '捆', '绑',
                '掐', '捏', '揪', '扯', '撕', '抓', '挠', '踢', '踹', '蹬', '踏', '跺', '跃', '纵', '窜',
                # 法术神通
                '捻诀', '念咒', '掐诀', '画符', '结印', '诵经', '持咒', '作法', '施法', '召神', '遣将',
                '变化', '化身', '分身', '遁形', '隐形', '显圣', '降妖', '伏魔', '收妖', '镇魔',
                # 日常行为
                '化斋', '乞食', '参禅', '打坐', '诵经', '礼拜', '叩首', '作揖', '拱手', '禀告', '启奏'
            ],
            
            # 3. 否定词（强化口语）
            'negatives': [
                '不', '非', '莫', '勿', '休', '无', '未', '弗', '休要', '莫要', '切莫', '不可', '不必', 
                '未曾', '休得', '不得', '不许', '不要', '别', '甭', '勿要', '毋须', '何须', '何必'
            ],
            
            # 4. 连词与衔接词
            'conjunctions': [
                # 文言连词
                '而', '且', '然', '则', '故', '以', '因', '盖', '苟', '既', '虽', '纵然', '即便',
                # 说书人式衔接
                '却说', '且说', '话说', '不提', '不表', '正是', '但见', '原来', '谁知', '不料', '不想',
                '怎奈', '岂料', '哪曾想', '谁曾想', '可恨', '可叹', '可怜', '可喜', '可恼'
            ],
            
            # 5. 代词与称呼
            'pronouns': [
                # 第一人称
                '俺', '咱', '洒家', '老孙', '贫僧', '老猪', '小神', '小仙', '小妖', '小的',
                # 第二人称
                '你这厮', '那泼怪', '尔等', '汝', '你', '你这孽畜', '你这猢狲', '你这呆子',
                # 第三人称
                '那厮', '那怪', '那妖', '那魔王', '那泼魔', '那孽障', '那贼秃'
            ],
            
            # 6. 经典套话与俗语
            'classical_terms': [
                # 诗词引语
                '有诗为证', '诗曰', '偈云', '但见那', '怎见得', '正是那', '真个是',
                # 俗语谚语
                '常言道', '古人云', '自古道', '俗语说', '有道是', '说甚么', '休言',
                # 佛教用语
                '阿弥陀佛', '善哉善哉', '我佛慈悲', '罪过罪过', '因果报应', '轮回转世',
                # 战斗叫阵
                '速速受死', '纳命来', '吃我一记', '看打', '招架', '且住', '休走'
            ],
            
            # 7. 辅助词与小品词
            'auxiliaries': [
                '之', '乎', '者', '也', '矣', '焉', '哉', '兮', '于', '以', '而', '所', '其',
                '个', '些', '此', '彼', '这', '那', '怎', '如何', '甚么', '那么', '恁地'
            ],
            
            # 8. 拟声词（战斗/法术/自然）
            'onomatopoeia': [
                # 武器碰撞
                '当', '锵', '铮', '镗', '乒', '乓', '唰', '嗖', '嗤', '嚓',
                # 法术效果
                '轰', '隆', '哗', '啦', '呼', '啸', '呜', '嗡', '嘶', '咔',
                # 自然声响
                '淅沥', '飒飒', '簌簌', '潺潺', '汩汩', '叮咚', '咕噜', '噼啪'
            ],
            
            # 9. 形容词（夸张描写）
            'adjectives': [
                '泼天', '彻地', '翻江', '倒海', '移星', '换斗', '担山', '赶月',
                '唬人', '了得', '凶恶', '狰狞', '慈悲', '威猛', '圆睁', '血盆',
                '霞光', '瑞气', '祥云', '紫雾', '黑气', '妖风', '腥膻', '臭秽'
            ],
            
            # 10. 时间/空间词
            'temporal_spatial': [
                '顷刻', '霎时', '俄而', '未几', '良久', '倏忽', '转眼', '登时',
                '这边厢', '那边厢', '左近', '四下里', '周遭', '前后', '左右'
            ]
        }
        
    def check_semantic_preservation(self, original_text, rewritten_text):
        """检查语义保持度
        
        Args:
            original_text (str): 原文
            rewritten_text (str): 改写文本
            
        Returns:
            float: 语义保持分数 (0-1)
        """
        if not self.model_loaded:
            # 如果模型未加载，使用SequenceMatcher作为后备
            return SequenceMatcher(None, original_text, rewritten_text).ratio()
        
        # 使用相似度模型
        encoding = self.tokenizer(
            original_text,
            rewritten_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 将输入移到对应设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测相似度
        with torch.no_grad():
            score = self.model(input_ids, attention_mask).item()
        
        return score
    
    def check_rewrite_degree(self, original_text, rewritten_text):
        """检查改写程度
        
        Args:
            original_text (str): 原文
            rewritten_text (str): 改写文本
            
        Returns:
            float: 改写程度分数 (0-1)，值越高表示改写程度越大
        """
        # 计算编辑距离
        edit_distance = Levenshtein.distance(original_text, rewritten_text)
        max_length = max(len(original_text), len(rewritten_text))
        
        # 归一化并转换为改写程度分数 (距离越大，改写程度越高)
        if max_length == 0:
            return 0.0
        raw_score = edit_distance / max_length
        
        # 调整评分范围，使其更有区分度
        # 映射 [0, 0.8] 区间到 [0, 1] 区间
        normalized_score = min(1.0, raw_score / 0.8)
        return normalized_score
    
    def check_style_conformity(self, text):
        """检查风格符合度 (文言文)
        
        Args:
            text (str): 文本
            
        Returns:
            float: 风格符合分数 (0-1)
        """
        # 检查文言文特征词出现频率
        total_markers = 0
        category_counts = {}
        
        for category, words in self.xiyou_markers.items():
            category_count = 0
            for word in words:
                count = text.count(word)
                total_markers += count
                category_count += count
            category_counts[category] = category_count
        
        # 计算特征词密度
        text_length = len(text)
        if text_length == 0:
            return 0.0
            
        marker_density = total_markers / text_length
        
        # 计算关键词多样性分数 - 使用了不同类别的词汇会得到更高分数
        categories_used = sum(1 for count in category_counts.values() if count > 0)
        diversity_score = categories_used / len(self.xiyou_markers)
        
        # 计算风格分数
        # 1. 文言词密度得分：典型的文言文密度约在0.15-0.25之间
        if marker_density > 0.35:  # 过多可能不自然
            density_score = 1.0 - (marker_density - 0.35) * 3.0
        else:
            density_score = min(1.0, marker_density / 0.18)  # 0.18是理想密度
        density_score = max(0.0, min(1.0, density_score))
        
        # 2. 词汇多样性得分
        diversity_score = min(1.0, diversity_score * 1.5)  # 增强多样性权重
        
        # 最终风格分数：密度占70%，多样性占30%
        final_style_score = 0.7 * density_score + 0.3 * diversity_score
        return min(1.0, max(0.0, final_style_score))
    
    def evaluate(self, original_text, rewritten_text):
        """综合评估改写效果
        
        Args:
            original_text (str): 原文
            rewritten_text (str): 改写文本
            
        Returns:
            dict: 包含多个维度评分的字典
        """
        semantic_score = self.check_semantic_preservation(original_text, rewritten_text)
        rewrite_score = self.check_rewrite_degree(original_text, rewritten_text)
        style_score = self.check_style_conformity(rewritten_text)
        
        # 综合评分 (可根据需求调整权重)
        # 由于移除了流畅度评分，调整权重
        comprehensive_score = (
            semantic_score * 0.4 + 
            rewrite_score * 0.3 + 
            style_score * 0.3
        )
        
        return {
            'semantic_preservation': round(semantic_score, 4),
            'rewrite_degree': round(rewrite_score, 4),
            'style_conformity': round(style_score, 4),
            'comprehensive_score': round(comprehensive_score, 4)
        }


def batch_evaluate(input_path="../../data/infer/test_output.json", model_path="../../model/merge_model/similarity_model"):
    """批量评估文本改写质量
    
    Args:
        input_path: 输入JSON文件路径 (包含base/predict字段)
        model_path: 相似度模型路径
        
    Returns:
        dict: 统计结果字典
    """
    try:
        # 读取输入数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 初始化评估器
        evaluator = MultiDimensionEvaluator(model_path=model_path)
        
        # 初始化结果统计
        stats = {
            'semantic_preservation': [],
            'rewrite_degree': [],
            'style_conformity': [],
            'comprehensive_score': []
        }
        
        # 保存所有评估结果和对应的文本
        all_results = []
        filtered_count = 0
        
        # 评估每一条数据
        for item in data:
            # 提取文本
            original = item['base']          # 原始输入
            model_rewrite = item['predict']  # 模型改写
            
            # 评估模型改写
            model_eval = evaluator.evaluate(original, model_rewrite)
            
            # 过滤掉改写程度和风格符合度为0的样本
            if model_eval['rewrite_degree'] == 0 or model_eval['style_conformity'] == 0:
                filtered_count += 1
                continue
            
            # 添加到统计数据
            for key in stats.keys():
                stats[key].append(model_eval[key])
            
            # 保存评估结果和对应文本
            all_results.append({
                'original': original,
                'rewrite': model_rewrite,
                'scores': model_eval
            })
        
        # 计算统计数据
        statistics = {}
        for key, values in stats.items():
            statistics[key] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'variance': float(np.var(values))
            }
        
        # 打印评分统计
        print("\n评分统计:")
        print(f"总样本数: {len(data)}")
        print(f"过滤样本数: {filtered_count}")
        print(f"有效样本数: {len(data) - filtered_count}")
        print("-" * 40)
        print(f"语义保持度: {statistics['semantic_preservation']['mean']:.4f}")
        print(f"改写程度: {statistics['rewrite_degree']['mean']:.4f}")
        print(f"风格符合度: {statistics['style_conformity']['mean']:.4f}")
        print(f"综合评分: {statistics['comprehensive_score']['mean']:.4f}")
        
        # # 打印统计结果
        # print("\n最终评分统计:")
        # print(f"均值: {statistics['comprehensive_score']['mean']:.4f}")
        # print(f"中位数: {statistics['comprehensive_score']['median']:.4f}")
        # print(f"标准差: {statistics['comprehensive_score']['std']:.4f}")
        # print(f"方差: {statistics['comprehensive_score']['variance']:.4f}")
        # print(f"最小值: {statistics['comprehensive_score']['min']:.4f}")
        # print(f"最大值: {statistics['comprehensive_score']['max']:.4f}")
        
        # # 打印最小评分对应的样本
        # print("\n最低分样本:")
        # print(f"原文: {min_sample['original']}")
        # print(f"改写: {min_sample['rewrite']}")
        # print(f"得分详情:")
        # print(f"  语义保持度: {min_sample['scores']['semantic_preservation']:.4f}")
        # print(f"  改写程度: {min_sample['scores']['rewrite_degree']:.4f}")
        # print(f"  风格符合度: {min_sample['scores']['style_conformity']:.4f}")
        # print(f"  综合评分: {min_sample['scores']['comprehensive_score']:.4f}")
                
    except FileNotFoundError as e:
        print(f"文件不存在: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
    except Exception as e:
        print(f"评估出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
     batch_evaluate(input_path="../data/infer/rlhf_output.json", 
                    model_path="../model/merge_model/similarity_model")