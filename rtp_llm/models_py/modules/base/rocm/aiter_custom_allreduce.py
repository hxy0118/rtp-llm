"""Aiter CustomAllreduce wrapper for ROCm prefill AllReduce.

Uses aiter low-level ops (``init_custom_ar``, ``all_reduce``, etc.)
directly, exchanging IPC handles via the NCCL group (same approach as
the C++ ``CustomAllReduceComm``).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 128 * 1024 * 1024  # 128 MB

class _AiterARManager:
    """Singleton that manages aiter custom AllReduce via low-level ops."""

    def __init__(self) -> None:
        self.group: Optional[ProcessGroup] = None
        self.device_id: Optional[int] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.fa: int = 0
        self.buffer: Optional[Tensor] = None
        self.max_size: int = _DEFAULT_MAX_SIZE
        self.initialized = False
        self.disabled = False

    def _exchange_ipc_handles(self, local_buffer: Tensor):
        """Exchange IPC handles via NCCL all_gather.

        Same approach as C++ CustomAllReduceComm::prepareP2PBuffer_:
        copy the local IPC handle to GPU, NCCL all_gather, then copy back.
        """
        import aiter as ops

        handle_tensor = ops.get_meta_buffer_ipc_handle(local_buffer)
        handle_size = handle_tensor.numel()

        device = f"cuda:{self.device_id}"
        local_gpu = torch.empty(handle_size, dtype=torch.uint8, device=device)
        gathered_gpu = torch.empty(
            handle_size * self.world_size, dtype=torch.uint8, device=device
        )

        local_gpu.copy_(handle_tensor)
        dist.all_gather_into_tensor(gathered_gpu, local_gpu, group=self.group)
        # Sync only the current stream — full device sync would block
        # unrelated streams (e.g. another process group's collectives).
        torch.cuda.current_stream().synchronize()

        gathered_cpu = gathered_gpu.cpu()
        handles = []
        offsets = []
        for i in range(self.world_size):
            start = i * handle_size
            end = start + handle_size
            handles.append(gathered_cpu[start:end].clone())
            offsets.append(0)
        return handles, offsets

    def initialize(self, group: ProcessGroup, device_id: int) -> None:
        if self.initialized and group == self.group and device_id == self.device_id:
            return
        # If this is a re-init (different group/device), free the previous
        # custom-AR handle and IPC buffer first to avoid leaking
        # hipDeviceMallocUncached memory across re-inits.
        if self.fa != 0:
            try:
                import aiter as ops

                ops.dispose(self.fa)
            except Exception as exc:
                logger.warning("Aiter CustomAR dispose on re-init failed: %s", exc)
            self.fa = 0
        self.buffer = None
        self._meta = None
        self._rank_data = None
        self.initialized = False
        self.disabled = False
        self.group = group
        self.device_id = device_id

        # Phase 0: every rank locally allocates the meta + buffer.
        # We must reach a consensus that ALL ranks succeeded BEFORE issuing
        # any collective (otherwise a peer's allocate_meta_buffer OOM would
        # deadlock the survivors at all_gather_into_tensor).
        local_ok = False
        local_err: Optional[str] = None
        meta = None
        rank_data = None
        buffer = None
        try:
            import aiter as ops

            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)

            if self.world_size == 1 or self.world_size not in {2, 4, 6, 8}:
                local_err = f"unsupported world_size={self.world_size}"
            else:
                torch.cuda.set_device(device_id)
                meta = ops.allocate_meta_buffer(ops.meta_size() + self.max_size * 2)
                rank_data = torch.empty(
                    8 * 1024 * 1024,
                    dtype=torch.uint8,
                    device=f"cuda:{device_id}",
                )
                # Must use allocate_meta_buffer (hipDeviceMallocUncached)
                # instead of torch.empty. On some ROCm platforms hipMalloc
                # memory does not support IPC (hipIpcOpenMemHandle returns
                # error 17).
                buffer = ops.allocate_meta_buffer(self.max_size)
                local_ok = True
        except ImportError:
            local_err = "aiter not available"
        except Exception as exc:
            local_err = f"local allocation failed: {exc}"

        # Phase 1 consensus
        try:
            world_size = dist.get_world_size(group=group)
            ok_flags = [None] * world_size
            dist.all_gather_object(ok_flags, local_ok, group=group)
        except Exception as exc:
            logger.warning("Aiter CustomAR init consensus failed: %s", exc)
            ok_flags = [False] * (self.world_size or 1)
            local_ok = False

        if not all(bool(x) for x in ok_flags):
            if local_err:
                logger.info("Aiter CustomAllreduce disabled: %s", local_err)
            else:
                logger.info(
                    "Aiter CustomAllreduce disabled: a peer rank failed init",
                )
            # Drop locally allocated resources (Python GC frees them).
            self.disabled = True
            self.initialized = True
            return

        # Phase 2: all ranks have meta+buffer — do the IPC exchanges.
        try:
            import aiter as ops

            meta_handles, meta_offsets = self._exchange_ipc_handles(meta)
            self.fa = ops.init_custom_ar(
                meta, rank_data, meta_handles, meta_offsets, self.rank, True,
            )
            buf_handles, buf_offsets = self._exchange_ipc_handles(buffer)
            ops.register_buffer(self.fa, buffer, buf_handles, buf_offsets)
            self.buffer = buffer
            self._meta = meta
            self._rank_data = rank_data
            dist.barrier(group=group)
            self.initialized = True
        except Exception as exc:
            logger.warning("Aiter CustomAR phase-2 IPC exchange failed: %s", exc)
            if self.fa != 0:
                try:
                    ops.dispose(self.fa)
                except Exception:
                    pass
                self.fa = 0
            self.disabled = True
            self.initialized = True

    def close(self) -> None:
        """Release the custom-AR handle and IPC buffer.

        Safe to call multiple times. Call from teardown paths
        (e.g. destroy_distributed_environment) so IPC buffers don't leak
        across process-group re-creation cycles.
        """
        if self.fa != 0:
            try:
                import aiter as ops

                ops.dispose(self.fa)
            except Exception as exc:
                logger.warning("Aiter CustomAR dispose on close failed: %s", exc)
            self.fa = 0
        self.buffer = None
        self._meta = None
        self._rank_data = None
        self.disabled = True
        self.initialized = False

    def ensure_initialized(self, group: ProcessGroup, device_id: int) -> bool:
        """Lazily initialize and return True if comm is usable."""
        if not self.initialized:
            self.initialize(group, device_id)
        return self.initialized and not self.disabled

    def should_use(self, tensor: Tensor, group: ProcessGroup, device_id: int) -> bool:
        """Check whether *tensor* is eligible for aiter CustomAllreduce.

        State-only — never triggers (re-)initialization. Call sites must
        run ``ensure_initialized`` outside of stream capture beforehand.
        """
        if not self.initialized or self.disabled or self.fa == 0:
            return False
        if self.group is not group or self.device_id != device_id:
            return False
        inp_size = tensor.numel() * tensor.element_size()
        if inp_size % 16 != 0:
            return False
        # 2-stage allreduce write mode uses 2x temp buffer,
        # so effective limit is max_size / 2
        return inp_size <= self.max_size // 2

    def allreduce(self, tensor: Tensor) -> Tensor:
        """AllReduce *tensor* via aiter P2P custom allreduce.

        Supports BF16 and FP8 dtypes.
        """
        import aiter as ops

        out = torch.empty_like(tensor)
        is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

        ops.all_reduce(
            self.fa,
            tensor,
            out,
            False,
            is_fp8,
            self.buffer,
        )
        return out

aiter_ar_manager = _AiterARManager()