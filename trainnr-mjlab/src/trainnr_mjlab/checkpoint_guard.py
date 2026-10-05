"""A checkpoint is loaded only when it holds tensors and plain data.

rsl_rl and mjlab load checkpoints with `torch.load(weights_only=False)`,
which unpickles arbitrary objects: a `.pt` file from someone else could
run code as it loads (security review, 2026-10-04). Every trainnr_mjlab
entry point first reads the file in PyTorch's weights-only mode, which
refuses anything but tensors and plain containers, and only then hands
it to the trainer's own loader. Real rsl_rl checkpoints pass.
"""

from __future__ import annotations

from pathlib import Path


def require_tensor_only(checkpoint: str | Path) -> None:
    """Refuse `checkpoint` by name unless it loads in weights-only mode."""
    import pickle  # noqa: PLC0415

    import torch  # noqa: PLC0415

    try:
        torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as why:
        raise ValueError(
            f"{checkpoint} holds more than tensors and plain data, and loading "
            f"it could run code it carries; refused ({why})"
        ) from why
