"""Entrées/sorties raster sûres pour le prototype pédagogique."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 12_000_000


def load_rgb(source, max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_PIXELS):
    """Decode a validated PNG/JPEG/TIFF path, byte string, or Streamlit upload."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Fichier introuvable : {path.name}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Fichier trop volumineux (> {max_bytes // (1024*1024)} Mo)")
        payload = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
    elif hasattr(source, "getvalue"):
        payload = source.getvalue()
    else:
        raise ValueError("Source d'image non prise en charge")
    if not payload:
        raise ValueError("Fichier vide")
    if len(payload) > max_bytes:
        raise ValueError(f"Fichier trop volumineux (> {max_bytes // (1024*1024)} Mo)")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError("Format réel non pris en charge (PNG, JPEG ou TIFF attendu)")
            width, height = image.size
            if width < 1 or height < 1:
                raise ValueError("Dimensions d'image invalides")
            if width * height > max_pixels:
                raise ValueError(f"Image trop grande (> {max_pixels:,} pixels)")
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image illisible, corrompue ou format incorrect") from exc


def mask_png_bytes(mask):
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Masque 2D non vide attendu")
    binary = ((arr > 127).astype(np.uint8) * 255)
    output = BytesIO()
    Image.fromarray(binary, mode="L").save(output, format="PNG")
    return output.getvalue()


def results_csv_bytes(records):
    return pd.DataFrame.from_records(records).to_csv(index=False).encode("utf-8-sig")
