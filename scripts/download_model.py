#!/usr/bin/env python3
"""
模型下载工具
支持从 HuggingFace 和 ModelScope 下载模型
"""

import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_from_huggingface(model_id: str, output_dir: str):
    """从 HuggingFace 下载模型"""
    logger.info(f"Downloading {model_id} from HuggingFace...")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            resume_download=True
        )
        
        logger.info(f"Model downloaded to {output_dir}")
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def download_from_modelscope(model_id: str, output_dir: str):
    """从 ModelScope 下载模型"""
    logger.info(f"Downloading {model_id} from ModelScope...")
    
    try:
        from modelscope import snapshot_download as ms_download
        
        ms_download(
            model_id=model_id,
            local_dir=output_dir
        )
        
        logger.info(f"Model downloaded to {output_dir}")
        
    except ImportError:
        logger.error("modelscope not installed. Run: pip install modelscope")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def download_dataset(dataset_id: str, output_dir: str, source: str = "huggingface"):
    """下载数据集"""
    logger.info(f"Downloading dataset {dataset_id}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if source == "huggingface":
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_id, cache_dir=output_dir)
            logger.info(f"Dataset downloaded to {output_dir}")
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            raise
    
    elif source == "modelscope":
        try:
            from modelscope import datasets as ms_datasets
            
            ms_datasets.load_dataset(dataset_id, cache_dir=output_dir)
            logger.info(f"Dataset downloaded to {output_dir}")
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Model Download Tool")
    
    parser.add_argument("--model_id", type=str, 
                       default="Qwen/Qwen2.5-0.5B",
                       help="模型 ID")
    parser.add_argument("--output_dir", type=str, 
                       default="./models/Qwen2.5-0.5B",
                       help="输出目录")
    parser.add_argument("--source", type=str, 
                       choices=["huggingface", "modelscope"],
                       default="huggingface",
                       help="下载源")
    parser.add_argument("--dataset_id", type=str, default=None,
                       help="数据集 ID（可选）")
    parser.add_argument("--dataset_output", type=str, default="./datasets",
                       help="数据集输出目录")
    
    args = parser.parse_args()
    
    # 下载模型
    if args.source == "huggingface":
        download_from_huggingface(args.model_id, args.output_dir)
    else:
        download_from_modelscope(args.model_id, args.output_dir)
    
    # 下载数据集（可选）
    if args.dataset_id:
        download_dataset(args.dataset_id, args.dataset_output, args.source)
    
    logger.info("Download completed!")


if __name__ == "__main__":
    main()
