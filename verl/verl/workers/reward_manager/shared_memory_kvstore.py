# verl/workers/reward_manager/shared_kv.py
from __future__ import annotations

import json
import mmap
import os
import pickle
import uuid
import errno
import fcntl
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# -------------------------------------------------
# 常量
# -------------------------------------------------
SHM_VALUE_PREFIX = "kv"          # /dev/shm/kv_<ns>_<uuid>
MMAP_INDEX_PATH  = "/tmp/kv_index_{ns}.mmap"
LOCK_PATH        = "/tmp/kv_lock_{ns}.lock"
INDEX_INIT_SIZE  = 1024 * 1024 * 4      # 4 MiB 起步，自动翻倍


class SharedMemoryKeyValueStore:
    """
    进程间 KV 存储（共享内存值段 + mmap 索引）。
    修复了：
        1. lost-update   – 提供 atomic update()
        2. 段悬空        – 两阶段提交
        3. 索引段丢失    – 固定 mmap 文件
        4. 残留/泄漏     – 启动自检清理
    """

    # ============================== 初始化 ==================================
    def __init__(self, *, namespace: str = "default"):
        self.ns = namespace

        # ---------- 文件锁 ----------
        self._lock_file_path = LOCK_PATH.format(ns=self.ns)
        Path(self._lock_file_path).touch()
        self._lock_fd = os.open(self._lock_file_path, os.O_RDWR)

        # ---------- mmap 索引 ----------
        self._index_path = MMAP_INDEX_PATH.format(ns=self.ns)
        if not Path(self._index_path).exists():
            with open(self._index_path, "wb") as f:
                f.truncate(INDEX_INIT_SIZE)
        self._index_fd = os.open(self._index_path, os.O_RDWR)
        self._index_size = os.path.getsize(self._index_path)
        self._index_mm = mmap.mmap(self._index_fd, self._index_size)

        # ---------- 自检 ----------
        self._self_check()

    # ============================== 公共 API ================================
    def put(self, key: str, value: Any, *, overwrite: bool = True):
        """非原子 put；如需“读-改-写”请使用 update()。"""
        with self._locked(shared=False):
            index = self._read_index()
            if (not overwrite) and (key in index):
                return
            self._commit_value(index, key, value)

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
        *,
        retries: int = 5,
        backoff: float = 0.05,
    ) -> Any:
        """带指数退避的只读 get。"""
        attempt = 0
        cur = backoff
        while attempt < retries:
            attempt += 1
            try:
                with self._locked(shared=True):
                    index = self._read_index()
                    if key not in index:
                        return default
                    shm_name, size, _version = index[key]
                shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    buf = bytes(shm.buf[:size])
                finally:
                    shm.close()
                return pickle.loads(buf)
            except (FileNotFoundError, OSError) as exc:
                # 兼容旧段已被删、索引正在迁移等瞬时错误
                if isinstance(exc, OSError) and exc.errno not in (
                    errno.ENOENT,
                    errno.EINVAL,
                    errno.ENODEV,
                ):
                    raise
                if attempt >= retries:
                    break
                time.sleep(cur)
                cur *= 2
        return default

    # ---- 新增：原子更新 ----------------------------------------------
    def update(
        self,
        key: str,
        fn: Callable[[Any], Any],
        *,
        default: Any = None,
    ):
        """
        原子「读-改-写」：
            new_val = fn(old_val or default)
            store[key] = new_val
        该过程在同一把写锁里完成，不会出现丢更新。
        """
        with self._locked(shared=False):
            index = self._read_index()
            # 读取旧值
            if key in index:
                shm_name, size, _ver = index[key]
                shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    old_val = pickle.loads(bytes(shm.buf[:size]))
                finally:
                    shm.close()
            else:
                old_val = default
            # 计算新值并提交
            new_val = fn(old_val)
            self._commit_value(index, key, new_val)

    def delete(self, key: str):
        with self._locked(shared=False):
            index = self._read_index()
            if key in index:
                shm_name, *_ = index[key]
                self._safe_unlink_value(shm_name)
                del index[key]
                self._write_index(index)

    def exists(self, key: str) -> bool:
        with self._locked(shared=True):
            return key in self._read_index()

    # ============================ 内部实现 =================================
    # ---------- 索引读写 ----------
    def _read_index(self) -> Dict[str, Tuple[str, int, int]]:
        raw = self._index_mm[:].rstrip(b"\x00")
        if not raw:
            return {}
        return json.loads(raw.decode())

    def _write_index(self, index: Dict[str, Tuple[str, int, int]]):
        data = json.dumps(index, separators=(",", ":")).encode()
        if len(data) > self._index_size:
            self._resize_index(max(len(data) * 2, self._index_size * 2))
        # 覆盖 + 清零剩余
        self._index_mm.seek(0)
        self._index_mm.write(data)
        self._index_mm.write(b"\x00" * (self._index_size - len(data)))
        self._index_mm.flush()

    # ---------- 值段提交（两阶段） ----------
    def _commit_value(self, index: dict, key: str, value: Any):
        # 1. dump -> bytes
        pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        shm_name = f"{SHM_VALUE_PREFIX}_{self.ns}_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=len(pickled))
        shm.buf[: len(pickled)] = pickled
        shm.close()          # 先把新段写好

        # 2. 更新索引（version +1）
        old_entry = index.get(key)
        version = (old_entry[2] + 1) if old_entry else 1
        index[key] = (shm_name, len(pickled), version)
        self._write_index(index)

        # 3. 删除旧段（若有）
        if old_entry:
            self._safe_unlink_value(old_entry[0])

    # ---------- 索引扩容 ----------
    def _resize_index(self, new_size: int):
        self._index_mm.close()
        os.ftruncate(self._index_fd, new_size)
        self._index_mm = mmap.mmap(self._index_fd, new_size)
        self._index_size = new_size

    # ---------- 自检 & 清理 ----------
    def _self_check(self):
        with self._locked(shared=False):
            index = self._read_index()
            dirty = False

            # a) 删除缺失段的键
            for k, (shm_name, _sz, _ver) in list(index.items()):
                if not self._segment_exists(shm_name):
                    del index[k]
                    dirty = True

            # b) 清理孤儿段
            referenced = {v[0] for v in index.values()}
            for fname in os.listdir("/dev/shm"):
                if not fname.startswith(f"{SHM_VALUE_PREFIX}_{self.ns}_"):
                    continue
                if fname not in referenced:
                    self._safe_unlink_value(fname)

            if dirty:
                self._write_index(index)

    # ---------- 工具 ----------
    @staticmethod
    def _segment_exists(name: str) -> bool:
        try:
            shared_memory.SharedMemory(name=name).close()
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _safe_unlink_value(name: str):
        try:
            shared_memory.SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass

    # ---------- 文件锁 ----------
    @contextmanager
    def _locked(self, *, shared: bool = True):
        op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(self._lock_fd, op)
        try:
            yield
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)