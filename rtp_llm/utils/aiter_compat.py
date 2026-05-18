import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_APPLIED = False


def patch_aiter_flydsl_version_gate():
    """Hide flydsl from aiter's importlib.util.find_spec detection.

    aiter/ops/flydsl/__init__.py raises ImportError when the installed flydsl
    version doesn't match its _REQUIRED_FLYDSL_VERSION.  That ImportError
    propagates up through ``from aiter.fused_moe import fused_moe`` and kills
    the entire MoE strategy registry on ROCm.

    This patch makes aiter's ``is_flydsl_available()`` (which calls
    ``importlib.util.find_spec("flydsl")``) return None → False, so aiter
    skips the version check entirely and falls back to CK kernels for MoE.

    rtp-llm's own flydsl usage (USE_FLYDSL=1, chunk-GDN) is unaffected
    because ``import flydsl`` goes through sys.meta_path, not find_spec.
    """
    global _APPLIED
    if _APPLIED:
        return

    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None:
        return

    aiter_spec = importlib.util.find_spec("aiter")
    if aiter_spec is None:
        return

    aiter_init = os.path.join(
        os.path.dirname(aiter_spec.origin), "ops", "flydsl", "__init__.py"
    )
    if not os.path.exists(aiter_init):
        return

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as get_version

    try:
        installed_flydsl = get_version("flydsl")
    except PackageNotFoundError:
        return

    required_version = _read_aiter_required_flydsl_version(aiter_init)
    if required_version is None or required_version == installed_flydsl:
        return

    _original_find_spec = importlib.util.find_spec

    def _patched_find_spec(name, *args, **kwargs):
        if name == "flydsl":
            return None
        return _original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _patched_find_spec
    _APPLIED = True
    logger.info(
        f"aiter_compat: hiding flydsl from aiter (aiter expects {required_version!r}, "
        f"installed {installed_flydsl!r}). aiter MoE will use CK fallback; "
        f"rtp-llm chunk-GDN (USE_FLYDSL=1) is unaffected."
    )


def _read_aiter_required_flydsl_version(init_path: str):
    """Parse _REQUIRED_FLYDSL_VERSION from aiter's flydsl __init__.py."""
    try:
        with open(init_path) as f:
            for line in f:
                if line.startswith("_REQUIRED_FLYDSL_VERSION"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip("\"'")
    except OSError:
        pass
    return None


patch_aiter_flydsl_version_gate()
