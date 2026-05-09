"""Image compression and encoding utilities."""

import logging
from pathlib import Path

from PIL import Image

from .config import IMAGE_MAX_SIZE, IMAGE_QUALITY, IMAGE_SIZE_THRESHOLD

logger = logging.getLogger(__name__)


def compress_image(
    input_path: str | Path,
    max_size: int = IMAGE_MAX_SIZE,
    quality: int = IMAGE_QUALITY,
    output_path: str | Path | None = None,
) -> Path:
    """Compress an image for vision model input.

    Returns the path to the compressed image, or the original if no
    compression was needed.
    """
    input_path = Path(input_path)
    file_size = input_path.stat().st_size

    if file_size <= IMAGE_SIZE_THRESHOLD:
        logger.info("Image %s is %.1f KB, no compression needed", input_path.name, file_size / 1024)
        return input_path

    img = Image.open(input_path)
    w, h = img.size

    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_compressed.jpg"
    else:
        output_path = Path(output_path)

    img.save(output_path, "JPEG", quality=quality, optimize=True)

    compressed_size = output_path.stat().st_size
    logger.info(
        "Compressed: %s %.1f KB (%dx%d) -> %.1f KB (%dx%d)",
        input_path.name,
        file_size / 1024, w, h,
        compressed_size / 1024, img.size[0], img.size[1],
    )
    return output_path
