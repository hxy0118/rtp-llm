import os
import logging

_logger = logging.getLogger(__name__)

# === Precision Debug Tensor Dump Utility ===
# Controlled by environment variables:
#   RTP_DUMP_TENSOR=1              — enable tensor dumping
#   RTP_DUMP_TENSOR_DIR=/path      — directory to save .pt files (default: /tmp/rtp_dump_tensors)
#   RTP_DUMP_TENSOR_LAYER=0,5      — comma-separated layer indices to dump (default: all layers)
#   RTP_DUMP_TENSOR_STEPS=0,1      — comma-separated forward step indices to dump (default: 0 only)
#   RTP_DUMP_TENSOR_SKIP_STEPS=10  — skip the first N steps (e.g. warmup) before counting (default: 0)

_DUMP_ENABLED: bool = os.environ.get("RTP_DUMP_TENSOR", "0") == "1"
_DUMP_DIR: str = os.environ.get("RTP_DUMP_TENSOR_DIR", "/tmp/rtp_dump_tensors")
_DUMP_LAYERS: set = (
    set(int(x) for x in os.environ.get("RTP_DUMP_TENSOR_LAYER", "").split(",") if x.strip())
    if os.environ.get("RTP_DUMP_TENSOR_LAYER", "")
    else set()  # empty means all layers
)
_DUMP_STEPS: set = (
    set(int(x) for x in os.environ.get("RTP_DUMP_TENSOR_STEPS", "").split(",") if x.strip())
    if os.environ.get("RTP_DUMP_TENSOR_STEPS", "")
    else {0}  # default: only dump step 0
)
_DUMP_SKIP_STEPS: int = int(os.environ.get("RTP_DUMP_TENSOR_SKIP_STEPS", "0"))
_raw_step_counter: int = -1
_dump_step_counter: int = -1
_cached_rank: int = -1


def _get_rank() -> int:
    """Get the current process rank for multi-GPU disambiguation."""
    global _cached_rank
    if _cached_rank >= 0:
        return _cached_rank
    try:
        import torch.distributed as dist  # type: ignore[import]
        if dist.is_initialized():
            _cached_rank = dist.get_rank()
            return _cached_rank
    except Exception:
        pass
    _cached_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return _cached_rank


def dump_tensor_enabled() -> bool:
    """Check if tensor dumping is enabled."""
    return _DUMP_ENABLED


def dump_tensor_step_begin():
    """Call at the beginning of each forward step to track step counter.

    The first ``_DUMP_SKIP_STEPS`` calls are silently skipped (useful for
    skipping warmup / cuda-graph capture steps).  After that the logical
    step counter starts from 0.
    """
    global _raw_step_counter, _dump_step_counter
    if not _DUMP_ENABLED:
        return
    _raw_step_counter += 1
    if _raw_step_counter < _DUMP_SKIP_STEPS:
        _dump_step_counter = -1
        return
    _dump_step_counter = _raw_step_counter - _DUMP_SKIP_STEPS


def _should_dump_layer(layer_idx: int) -> bool:
    """Check if this layer should be dumped."""
    if not _DUMP_LAYERS:
        return True  # empty set = dump all
    return layer_idx in _DUMP_LAYERS


def _should_dump_step() -> bool:
    """Check if the current step should be dumped."""
    return _dump_step_counter in _DUMP_STEPS


def dump_tensor(
    tensor: 'torch.Tensor',
    name: str,
    layer_idx: int = -1,
    save_file: bool = True,
):
    """
    Dump a tensor's statistics and optionally save to file for precision comparison.

    Usage:
        from rtp_llm.models_py.utils.debug import dump_tensor, dump_tensor_enabled
        if dump_tensor_enabled():
            dump_tensor(hidden_states, "layer0.input_layernorm_out", layer_idx=0)

    Args:
        tensor: The tensor to dump
        name: A descriptive name (e.g., "layer0.qkv_proj_out")
        layer_idx: Layer index for filtering (-1 means always dump)
        save_file: Whether to save the tensor to a .pt file
    """
    if not _DUMP_ENABLED:
        return
    if not _should_dump_step():
        return
    if layer_idx >= 0 and not _should_dump_layer(layer_idx):
        return

    import torch

    rank = _get_rank()

    # Print statistics with rank info
    t = tensor.detach().float()
    stats = (
        f"[DUMP] rank={rank} step={_dump_step_counter} {name}: "
        f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
        f"abs_mean={t.abs().mean().item():.6f}, "
        f"has_nan={tensor.isnan().any().item()}, "
        f"has_inf={tensor.isinf().any().item()}"
    )
    _logger.info(stats)
    print(stats, flush=True)

    # Save tensor to per-rank subdirectory to avoid multi-GPU file conflicts
    if save_file:
        rank_dump_dir = os.path.join(_DUMP_DIR, f"rank{rank}")
        os.makedirs(rank_dump_dir, exist_ok=True)
        safe_name = name.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(rank_dump_dir, f"step{_dump_step_counter}_{safe_name}.pt")
        torch.save(tensor.detach().cpu(), file_path)


def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb

        tty_r = open("/dev/tty", "r")
        tty_w = open("/dev/tty", "w")
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback

        traceback.print_exc()


def remote_debug_breakpoint(host="localhost", port=4444):
    """
    启动一个远程 PDB 会话，监听指定的主机和端口。
    使用 telnet 连接到该主机和端口以进行调试。
    """
    import debugpy

    debugpy.listen((host, port))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    debugpy.breakpoint()


import torch


def cudagraph_debug_kernel(
    data: torch.Tensor | None,
    info_id: int = 1,
    m: int = 0,
    n: int = 0,
    start_row: int = 0,
    start_col: int = 0,
    row_len: int = 0,
    name: str = "cudagraph_debug_kernel",
):
    if data is None:
        return
    print(f"{name} shape is {data.shape}")
    if data.dim() == 1:
        data = data.unsqueeze(0)
    data = data.contiguous().to(torch.float32)
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    row_len = data.size(1) if row_len == 0 else row_len
    n = data.size(1) if (n == 0 or n > data.size(1)) else n
    m = data.size(0) if (m == 0 or m > data.size(0)) else m
    rtp_llm_ops.debug_kernel(
        data=data,
        start_row=start_row,
        start_col=start_col,
        m=m,
        n=n,
        row_len=row_len,  # 每行的长度
        info_id=info_id,
    )
