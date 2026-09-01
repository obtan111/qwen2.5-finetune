"""
资源监控器
监控 CPU、内存、GPU 使用情况
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.gpu_start_memory = None

    @contextmanager
    def track(self):
        """上下文管理器，追踪资源使用"""
        self.start_time = time.time()
        self.gpu_start_memory = self._get_gpu_memory()

        yield

        self.end_time = time.time()

    def _get_gpu_memory(self) -> Dict[int, float]:
        """获取 GPU 内存使用"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    i: torch.cuda.memory_allocated(i) / (1024 ** 3)
                    for i in range(torch.cuda.device_count())
                }
        except Exception:
            pass
        return {}

    def get_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info[f"gpu_{i}"] = {
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024 ** 3),
                        "memory_cached_gb": torch.cuda.memory_cached(i) / (1024 ** 3) if hasattr(torch.cuda, 'memory_cached') else 0,
                    }
        except Exception:
            pass

        duration = 0
        if self.start_time:
            end = self.end_time if self.end_time else time.time()
            duration = end - self.start_time

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_total_gb": memory.total / (1024 ** 3),
            "gpus": gpu_info,
            "duration_seconds": duration
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """获取当前资源快照"""
        import psutil

        snapshot = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024 ** 3),
            },
            "gpu": self._get_gpu_memory()
        }

        return snapshot
