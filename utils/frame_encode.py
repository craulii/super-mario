"""Encode de frames (ndarray RGB/gray) a JPEG base64 para streaming por WebSocket."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def ndarray_to_jpeg_b64(frame: np.ndarray, quality: int = 60) -> str:
    """Convierte ndarray HxW o HxWx3 en JPEG base64."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        img = Image.fromarray(frame, mode="L")
    elif frame.ndim == 3 and frame.shape[2] == 3:
        img = Image.fromarray(frame, mode="RGB")
    elif frame.ndim == 3 and frame.shape[2] == 4:
        img = Image.fromarray(frame, mode="RGBA").convert("RGB")
    else:
        raise ValueError(f"Shape de frame no soportada: {frame.shape}")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
