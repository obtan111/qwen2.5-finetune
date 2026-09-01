"""
模型管理器
负责模型的加载、卸载和生命周期管理
"""

import gc
import logging
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaded_models = {}
        self.tokenizer = None

    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None
    ) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """加载模型"""
        # 检查是否已加载
        if model_path in self.loaded_models:
            logger.info(f"Model {model_path} already loaded")
            return self.loaded_models[model_path], self.tokenizer

        logger.info(f"Loading model from {model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.get("trust_remote_code", True)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 确定数据类型
        dtype_str = self.config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.config.get("trust_remote_code", True),
            torch_dtype=dtype,
            device_map=self.config.get("device_map", "auto"),
            low_cpu_mem_usage=True
        )

        # 加载 LoRA adapter（如果有）
        if adapter_path:
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        # 启用 Flash Attention（如果支持）
        if self.config.get("inference", {}).get("use_flash_attention", False):
            try:
                model.enable_flash_attention()
                logger.info("Flash Attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable Flash Attention: {e}")

        model.eval()
        self.loaded_models[model_path] = model

        logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
        return model, self.tokenizer

    def unload_model(self, model_path: str):
        """卸载模型"""
        if model_path in self.loaded_models:
            del self.loaded_models[model_path]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {model_path} unloaded")

    def unload_all(self):
        """卸载所有模型"""
        self.loaded_models.clear()
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "path": model_path,
            "loaded": model_path in self.loaded_models,
        }

        if model_path in self.loaded_models:
            model = self.loaded_models[model_path]
            info.update({
                "parameters": model.num_parameters(),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
            })

        return info
