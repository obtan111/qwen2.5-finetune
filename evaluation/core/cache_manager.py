"""
缓存管理器
支持 SQLite 和内存缓存
"""

import json
import time
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.backend = config.get("backend", "sqlite")
        self.ttl = config.get("ttl", 86400)  # 默认24小时
        self.db_path = Path(config.get("db_path", "./cache/eval_cache.db"))
        self.lock = Lock()
        self.memory_cache = {}

        if self.enabled and self.backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self):
        """初始化 SQLite 数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON eval_cache(expires_at)
        """)
        conn.commit()
        conn.close()
        logger.info(f"Cache initialized at {self.db_path}")

    def _get_key_hash(self, key: str) -> str:
        """生成键的哈希"""
        return hashlib.md5(key.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.enabled:
            return None

        key_hash = self._get_key_hash(key)
        now = time.time()

        if self.backend == "memory":
            if key_hash in self.memory_cache:
                entry = self.memory_cache[key_hash]
                if entry["expires_at"] > now:
                    return entry["value"]
                else:
                    del self.memory_cache[key_hash]
            return None

        elif self.backend == "sqlite":
            try:
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.execute(
                        "SELECT value, expires_at FROM eval_cache WHERE key = ?",
                        (key_hash,)
                    )
                    row = cursor.fetchone()
                    conn.close()

                    if row and row[1] > now:
                        return json.loads(row[0])
                    elif row:
                        self._delete(key_hash)
            except Exception as e:
                logger.warning(f"Cache get error: {e}")

        return None

    async def set(self, key: str, value: Any):
        """设置缓存"""
        if not self.enabled:
            return

        key_hash = self._get_key_hash(key)
        now = time.time()
        expires_at = now + self.ttl

        serialized = json.dumps(value, default=str)

        if self.backend == "memory":
            self.memory_cache[key_hash] = {
                "value": value,
                "expires_at": expires_at
            }

        elif self.backend == "sqlite":
            try:
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute(
                        "INSERT OR REPLACE INTO eval_cache (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                        (key_hash, serialized, now, expires_at)
                    )
                    conn.commit()
                    conn.close()
            except Exception as e:
                logger.warning(f"Cache set error: {e}")

    def _delete(self, key_hash: str):
        """删除缓存条目"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("DELETE FROM eval_cache WHERE key = ?", (key_hash,))
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")

    async def clear(self):
        """清空缓存"""
        if self.backend == "memory":
            self.memory_cache.clear()
        elif self.backend == "sqlite":
            try:
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute("DELETE FROM eval_cache")
                    conn.commit()
                    conn.close()
            except Exception as e:
                logger.warning(f"Cache clear error: {e}")

    def cleanup_expired(self):
        """清理过期缓存"""
        if self.backend == "sqlite":
            try:
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute(
                        "DELETE FROM eval_cache WHERE expires_at < ?",
                        (time.time(),)
                    )
                    conn.commit()
                    conn.close()
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")

        elif self.backend == "memory":
            now = time.time()
            expired_keys = [
                k for k, v in self.memory_cache.items()
                if v["expires_at"] < now
            ]
            for k in expired_keys:
                del self.memory_cache[k]
