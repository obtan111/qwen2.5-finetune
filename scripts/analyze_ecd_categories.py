#!/usr/bin/env python3
"""
ECD 数据集品类分析
通过商品关键词推断每条对话的品类, 统计分布 (ECD 无品类标注, 为近似估计)

用法:
  python scripts/analyze_ecd_categories.py                       # 分析 train.txt 抽样 + dev 全量
  python scripts/analyze_ecd_categories.py --input "dataset/E-commerce dataset/dev.txt"
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# 品类关键词词典 (淘宝客服语境, 选品类专属词降低误报)
CATEGORIES = {
    "食品/零食": [
        "零食", "瓜子", "松子", "坚果", "青团", "豆沙", "蛋黄酥", "肉松", "榴莲",
        "蛋糕", "饼干", "糖果", "蜜饯", "红枣", "枸杞", "茶叶", "蜂蜜", "薯片",
        "巧克力", "麦片", "咖啡", "试吃", "保质期", "生产日期", "口味", "好吃",
        "克装", "散装", "袋装", "新鲜", "吃货", "零食大礼包",
    ],
    "服装/鞋帽": [
        "尺码", "码数", "XL", "XXL", "版型", "宽松", "上衣", "裤子", "连衣裙",
        "外套", "羽绒服", "棉袄", "毛衣", "T恤", "衬衫", "裙子", "牛仔裤",
        "卫衣", "靴子", "高跟鞋", "拖鞋", "内衣", "袜子", "打底", "穿搭",
    ],
    "数码/电器": [
        "手机", "充电宝", "充电器", "电池", "屏幕", "耳机", "蓝牙", "电脑",
        "笔记本", "鼠标", "键盘", "内存卡", "显卡", "电视", "冰箱", "洗衣机",
        "数据线", "音箱", "摄像头", "机顶盒",
    ],
    "家居/日用": [
        "纸巾", "抽纸", "卷纸", "湿巾", "毛巾", "浴巾", "洗衣液", "收纳",
        "整理箱", "保温杯", "水壶", "四件套", "枕头", "被子", "床垫", "拖把",
        "垃圾桶", "垃圾袋", "香薰", "置物架", "沥水",
    ],
    "美妆/个护": [
        "面膜", "口红", "粉底", "防晒", "护肤", "精华", "乳液", "洗面奶",
        "洗发水", "沐浴露", "香水", "眉笔", "眼影", "爽肤水", "面霜",
    ],
    "母婴/儿童": [
        "奶粉", "尿不湿", "尿裤", "奶瓶", "辅食", "婴儿", "新生儿",
        "儿童餐", "幼儿园", "早教", "童装", "宝宝衣",
    ],
    "宠物": [
        "狗粮", "猫粮", "猫砂", "宠物", "狗狗", "猫咪", "鱼缸", "猫抓板",
        "狗窝", "猫窝", "驱虫",
    ],
}

# 快递/物流类词 (几乎所有对话都有, 单独统计用作对照, 不参与品类)
LOGISTICS_WORDS = ["快递", "发货", "物流", "韵达", "邮政", "EMS", "顺丰", "圆通", "申通", "中通", "百世"]


def load_positive_sessions(path: str, limit: int = None):
    """读取正样本对话 (label=1), 返回 [(全文文本, ), ...]"""
    sessions = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3 or parts[0] != '1':
                continue
            text = ''.join(parts[1:]).replace(' ', '')
            sessions.append(text)
            if limit and len(sessions) >= limit:
                break
    return sessions


def match_categories(text: str):
    """返回该对话命中的品类列表"""
    hits = []
    for cat, words in CATEGORIES.items():
        if any(w in text for w in words):
            hits.append(cat)
    return hits


def main():
    parser = argparse.ArgumentParser(description="ECD 品类分析")
    parser.add_argument("--input", type=str, default=None,
                       help="ECD 文件 (默认: train.txt 抽样5万 + dev.txt 全量)")
    parser.add_argument("--limit", type=int, default=50000,
                       help="train.txt 正样本抽样上限")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "dataset" / "E-commerce dataset"

    if args.input:
        files = [(args.input, None)]
    else:
        files = [(root / "train.txt", args.limit), (root / "dev.txt", None)]

    cat_counter = Counter()      # 品类命中对话数 (一条可命中多品类)
    total = 0
    no_cat = 0
    logi_only = 0

    for path, limit in files:
        for text in load_positive_sessions(str(path), limit):
            total += 1
            hits = match_categories(text)
            if hits:
                cat_counter.update(hits)
            else:
                no_cat += 1
                if any(w in text for w in LOGISTICS_WORDS):
                    logi_only += 1

    print(f"分析对话总数: {total}")
    print(f"未识别品类的对话: {no_cat} ({no_cat/total*100:.1f}%), 其中纯物流/售后者 {logi_only}")
    print()
    print(f"{'品类':<12s}{'命中对话数':>10s}{'占比':>10s}")
    print("-" * 34)
    for cat, n in cat_counter.most_common():
        print(f"{cat:<12s}{n:>10d}{n/total*100:>9.1f}%")
    print("-" * 34)
    print("注: 一条对话可命中多个品类, 占比之和可超过 100%")


if __name__ == "__main__":
    main()