"""A torch-saved dict of tensors, read with no torch installed.

Published identification logs often come as `torch.save` files (the
IIT sim2real-robot-identification chirps, 2026). Installing torch to
read twelve columns of floats is a 2 GB dependency for a 1 MB file, so
this reads the format itself: `torch.save` writes a zip whose
`data.pkl` is a pickle referring to storages by key, each storage a
raw little-endian byte file under `data/`. The unpickler here maps
torch's rebuild hooks onto numpy and refuses, by name, anything else
the pickle asks for — a reader for arrays, never an executor of
arbitrary objects.

Standard library plus numpy only. Nested containers (dict, list,
tuple, OrderedDict) of tensors, numpy arrays and Python scalars read
as-is.
"""

from __future__ import annotations

import codecs
import io
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

DATA_PICKLE = "data.pkl"
DATA_DIR = "data"
STORAGE_TAG = "storage"

# torch's storage class names → numpy dtypes (little-endian on disk).
STORAGE_DTYPES: dict[str, np.dtype] = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}
# The only globals the pickle may name; everything else is refused.
# numpy arrays pickled inside the dict (the IIT chirps store numpy, not
# tensors) rebuild through numpy's own hooks, listed under both module
# spellings numpy has used.
# numpy 2 moved its internals to `numpy._core` and deprecated `numpy.core`.
_MULTIARRAY = getattr(np, "_core", None) or np.core
_MULTIARRAY = _MULTIARRAY.multiarray
ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): "rebuild",
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
    ("numpy.core.multiarray", "scalar"): _MULTIARRAY.scalar,
    ("numpy._core.multiarray", "scalar"): _MULTIARRAY.scalar,
    # numpy spells a dtype descriptor as a byte string through this.
    ("_codecs", "encode"): codecs.encode,
}


class _StorageType:
    """A placeholder for `torch.FloatStorage` and friends: only its name
    matters, it says the dtype."""

    def __init__(self, name: str) -> None:
        self.name = name


def _rebuild_tensor(
    storage: np.ndarray,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *_ignored: Any,
) -> np.ndarray:
    """`torch._utils._rebuild_tensor_v2` on numpy: a strided view into
    the storage, copied so the array owns its memory. The view is checked
    to lie inside the storage first — a crafted or corrupt file would
    otherwise read memory past the buffer (review 2026-09-24)."""
    size, stride = tuple(int(n) for n in size), tuple(int(n) for n in stride)
    if (
        offset < 0
        or len(size) != len(stride)
        or any(n < 0 for n in size)
        or any(s < 0 for s in stride)
    ):
        raise ValueError(
            f"tensor view refused: offset {offset}, size {size}, stride {stride} "
            "(negative, or shape and strides of different rank)"
        )
    if all(size):
        last = offset + sum((n - 1) * s for n, s in zip(size, stride, strict=True))
        if last >= len(storage):
            raise ValueError(
                f"tensor view refused: it reaches element {last} of a storage of "
                f"{len(storage)} (offset {offset}, size {size}, stride {stride})"
            )
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=tuple(size),
        strides=tuple(s * itemsize for s in stride),
        writeable=False,
    )
    return np.array(view)


class _Unpickler(pickle.Unpickler):
    def __init__(
        self, stream: io.BytesIO, archive: zipfile.ZipFile, prefix: str
    ) -> None:
        super().__init__(stream)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in STORAGE_DTYPES:
            return _StorageType(name)
        target = ALLOWED_GLOBALS.get((module, name))
        if target == "rebuild":
            return _rebuild_tensor
        if target is not None:
            return target
        raise pickle.UnpicklingError(
            f"{module}.{name} is not something a tensor file needs; refused"
        )

    def persistent_load(self, pid: Any) -> Any:
        if not (isinstance(pid, tuple) and pid and pid[0] == STORAGE_TAG):
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        _tag, storage_type, key, _location, numel = pid[:5]
        dtype = STORAGE_DTYPES[storage_type.name]
        raw = self._archive.read(f"{self._prefix}{DATA_DIR}/{key}")
        array = np.frombuffer(raw, dtype=dtype)
        if len(array) < numel:
            raise ValueError(
                f"storage {key}: {len(array)} elements on disk, {numel} declared"
            )
        return array


def load_tensors(path: Path) -> Any:
    """The saved object, tensors as numpy arrays. Refuses a file that is
    not a torch zip archive, by name."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a torch zip archive (the legacy tar/pickle "
            "format is not read; re-save with torch.save on torch >= 1.6)"
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [
            n for n in names if n.endswith("/" + DATA_PICKLE) or n == DATA_PICKLE
        ]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected one {DATA_PICKLE}, found {pickles}")
        prefix = pickles[0][: -len(DATA_PICKLE)]
        stream = io.BytesIO(archive.read(pickles[0]))
        return _Unpickler(stream, archive, prefix).load()
