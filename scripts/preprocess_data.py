#!/usr/bin/env python3
"""
数据预处理工具
清洗、格式化和准备训练数据
"""

import os
import json
import re
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.seen_hashes = set()
    
    def load_data(self, input_path: str) -> List[Dict]:
        """加载数据"""
        input_path = Path(input_path)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    return [data]
        
        elif input_path.suffix == '.jsonl':
            data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        
        elif input_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(input_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 规范化引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """去重"""
        unique_data = []
        
        for item in data:
            content = json.dumps(item, sort_keys=True, ensure_ascii=False)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                unique_data.append(item)
        
        logger.info(f"Removed {len(data) - len(unique_data)} duplicates")
        return unique_data
    
    def filter_by_length(
        self,
        data: List[Dict],
        min_length: int = 10,
        max_length: int = 4096
    ) -> List[Dict]:
        """按长度过滤"""
        filtered = []
        
        for item in data:
            prompt = item.get('prompt', '')
            reference = item.get('reference', '')
            total_length = len(prompt) + len(reference)
            
            if min_length <= total_length <= max_length:
                filtered.append(item)
        
        logger.info(f"Filtered by length: {len(data)} -> {len(filtered)}")
        return filtered
    
    def convert_to_chatml(self, data: List[Dict]) -> List[Dict]:
        """转换为 ChatML 格式"""
        converted = []
        
        for item in data:
            # 已经是 ChatML 格式
            if 'messages' in item:
                converted.append(item)
                continue
            
            # Alpaca 格式
            if 'instruction' in item:
                messages = [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": item.get('instruction', '')},
                    {"role": "assistant", "content": item.get('output', '')}
                ]
                converted.append({"messages": messages})
                continue
            
            # ShareGPT 格式
            if 'conversations' in item:
                messages = []
                for turn in item['conversations']:
                    role = turn.get('from', 'user')
                    if role == 'human':
                        role = 'user'
                    elif role == 'gpt':
                        role = 'assistant'
                    messages.append({
                        "role": role,
                        "content": turn.get('value', '')
                    })
                converted.append({"messages": messages})
                continue
            
            # prompt/reference 格式
            if 'prompt' in item and ('reference' in item or 'answer' in item):
                messages = [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": item.get('prompt', '')},
                    {"role": "assistant", "content": item.get('reference', item.get('answer', ''))}
                ]
                converted.append({"messages": messages})
                continue
            
            logger.warning(f"Unknown format for item: {item.keys()}")
        
        return converted
    
    def add_system_prompt(
        self,
        data: List[Dict],
        system_prompt: str = "你是一个有帮助的助手。"
    ) -> List[Dict]:
        """添加系统提示"""
        for item in data:
            if 'messages' in item:
                messages = item['messages']
                if not messages or messages[0].get('role') != 'system':
                    messages.insert(0, {"role": "system", "content": system_prompt})
        
        return data
    
    def split_dataset(
        self,
        data: List[Dict],
        train_ratio: float = 0.9,
        eval_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """分割数据集"""
        import random
        
        random.seed(seed)
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        
        train_data = shuffled[:n_train]
        eval_data = shuffled[n_train:]
        
        logger.info(f"Split dataset: train={len(train_data)}, eval={len(eval_data)}")
        return train_data, eval_data
    
    def save_data(self, data: List[Dict], output_path: str):
        """保存数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif output_path.suffix == '.jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} samples to {output_path}")
    
    def get_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            "total_samples": len(data),
            "format": "unknown"
        }
        
        # 检测格式
        if data and 'messages' in data[0]:
            stats["format"] = "chatml"
        elif data and 'instruction' in data[0]:
            stats["format"] = "alpaca"
        elif data and 'conversations' in data[0]:
            stats["format"] = "sharegpt"
        
        # 长度统计
        lengths = []
        for item in data:
            if 'messages' in item:
                content = ' '.join(m.get('content', '') for m in item['messages'])
            elif 'prompt' in item:
                content = item.get('prompt', '') + item.get('reference', '')
            else:
                content = str(item)
            lengths.append(len(content))
        
        if lengths:
            stats["length_stats"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2]
            }
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Data Preprocessing Tool")
    
    parser.add_argument("--input", type=str, required=True,
                       help="输入文件路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出文件路径")
    parser.add_argument("--format", type=str, default="chatml",
                       choices=["chatml", "alpaca", "sharegpt", "auto"],
                       help="输出格式")
    parser.add_argument("--min_length", type=int, default=10,
                       help="最小长度")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="最大长度")
    parser.add_argument("--remove_duplicates", action="store_true",
                       help="去重")
    parser.add_argument("--split", action="store_true",
                       help="分割训练集和验证集")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="训练集比例")
    parser.add_argument("--system_prompt", type=str, 
                       default="你是一个有帮助的助手。",
                       help="系统提示")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    
    # 加载数据
    logger.info(f"Loading data from {args.input}")
    data = preprocessor.load_data(args.input)
    logger.info(f"Loaded {len(data)} samples")
    
    # 清洗文本
    for item in data:
        for key in item:
            if isinstance(item[key], str):
                item[key] = preprocessor.clean_text(item[key])
    
    # 去重
    if args.remove_duplicates:
        data = preprocessor.remove_duplicates(data)
    
    # 按长度过滤
    data = preprocessor.filter_by_length(data, args.min_length, args.max_length)
    
    # 转换格式
    if args.format == "auto":
        data = preprocessor.convert_to_chatml(data)
    elif args.format != "chatml":
        data = preprocessor.convert_to_chatml(data)
    
    # 添加系统提示
    data = preprocessor.add_system_prompt(data, args.system_prompt)
    
    # 获取统计信息
    stats = preprocessor.get_statistics(data)
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # 保存
    if args.split:
        train_data, eval_data = preprocessor.split_dataset(data, args.train_ratio)
        
        output_path = Path(args.output)
        train_path = output_path.parent / f"train{output_path.suffix}"
        eval_path = output_path.parent / f"eval{output_path.suffix}"
        
        preprocessor.save_data(train_data, str(train_path))
        preprocessor.save_data(eval_data, str(eval_path))
    else:
        preprocessor.save_data(data, args.output)
    
    logger.info("Preprocessing completed!")


if __name__ == "__main__":
    main()
