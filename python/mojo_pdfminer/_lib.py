from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np
import numpy.typing as npt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CUSTOM_LIB = os.environ.get("MOJO_PDFMINER_LIB")
_BUILT_LIB = os.path.join(ROOT, "dist", "libmojo-pdfminer-six.so")
LIB = _CUSTOM_LIB or _BUILT_LIB

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpdf_group_chars": ([I, I, F, F, I, I], None),
    "mpdf_group_lines": (
        [I, I, I, F, F, F, F, F, I, I, I, I, I],
        None,
    ),
    "mpdf_box_distances": ([I, I, I], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(_BUILT_LIB):
        detail = "\n".join(
            part.strip() for part in (proc.stdout, proc.stderr) if part.strip()
        )
        raise RuntimeError(detail[-4000:] or "Mojo build produced no library")
    return _BUILT_LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            if _CUSTOM_LIB:
                raise FileNotFoundError(
                    f"MOJO_PDFMINER_LIB does not exist: {_CUSTOM_LIB}"
                )
            build()
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty array")
    return address


def _geometry_array(geometry: npt.ArrayLike) -> np.ndarray:
    source = np.asarray(geometry)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(
            f"geometry must have shape (n, 4), got {source.shape}"
        )
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("geometry coordinates must be real numbers")
    try:
        result = np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("geometry coordinates must be real numbers") from error
    if result.shape != source.shape:
        raise RuntimeError("geometry conversion changed the array shape")
    return result


def _kind_array(kinds: npt.ArrayLike, n: int) -> np.ndarray:
    source = np.asarray(kinds)
    if source.ndim != 1 or len(source) != n:
        raise ValueError(f"kinds must have shape ({n},), got {source.shape}")
    if not (
        np.issubdtype(source.dtype, np.bool_)
        or np.issubdtype(source.dtype, np.integer)
    ):
        raise TypeError("kinds must contain integer values 0 or 1")
    if np.any((source != 0) & (source != 1)):
        raise ValueError("kinds must contain only 0 (horizontal) or 1 (vertical)")
    return np.ascontiguousarray(source, dtype=np.uint8)


def boxes(objects) -> np.ndarray:
    return np.fromiter(
        (
            value
            for obj in objects
            for value in (obj.x0, obj.y0, obj.x1, obj.y1)
        ),
        dtype=np.float64,
        count=len(objects) * 4,
    ).reshape((-1, 4))


def character_relations(
    geometry: npt.ArrayLike,
    line_overlap: float,
    char_margin: float,
    detect_vertical: bool,
) -> np.ndarray:
    geometry = _geometry_array(geometry)
    n = len(geometry)
    result = np.empty(max(0, n - 1), dtype=np.uint8)
    if n > 1:
        lib().mpdf_group_chars(
            addr(geometry),
            n,
            line_overlap,
            char_margin,
            detect_vertical,
            addr(result),
        )
    return result


def line_components(
    geometry: npt.ArrayLike,
    kinds: npt.ArrayLike,
    line_margin: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    geometry = _geometry_array(geometry)
    n = len(geometry)
    kinds = _kind_array(kinds, n)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    h_order = np.argsort(geometry[:, 1], kind="stable").astype(np.int64)
    h_prefix = np.maximum.accumulate(geometry[h_order, 3])
    v_order = np.argsort(geometry[:, 0], kind="stable").astype(np.int64)
    v_prefix = np.maximum.accumulate(geometry[v_order, 2])
    parents = np.empty(n, dtype=np.int64)
    if bbox is None:
        bbox = (
            float(geometry[:, 0].min()) - 1,
            float(geometry[:, 1].min()) - 1,
            float(geometry[:, 2].max()) + 1,
            float(geometry[:, 3].max()) + 1,
        )
    elif len(bbox) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {len(bbox)}")
    lib().mpdf_group_lines(
        addr(geometry),
        addr(kinds),
        n,
        line_margin,
        *bbox,
        addr(h_order),
        addr(h_prefix),
        addr(v_order),
        addr(v_prefix),
        addr(parents),
    )
    return parents


def box_distances(geometry: npt.ArrayLike) -> np.ndarray:
    geometry = _geometry_array(geometry)
    n = len(geometry)
    result = np.empty(n * (n - 1) // 2, dtype=np.float64)
    if n > 1:
        lib().mpdf_box_distances(addr(geometry), n, addr(result))
    return result
