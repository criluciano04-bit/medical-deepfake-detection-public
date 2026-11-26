import numpy as np
import torch
import torch.nn.functional as F

def apply_hu_window(
    image: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Apply HU windowing to a CT image.

    Values below `lower` are clipped to `lower`.
    Values above `upper` are clipped to `upper`.

    Parameters
    ----------
    image : np.ndarray
        CT image with pixel intensities in Hounsfield Units.
    lower : float
        Lower HU bound.
    upper : float
        Upper HU bound.

    Returns
    -------
    np.ndarray
        Windowed image as a float32 array.
    """
    if lower >= upper:
        raise ValueError("The lower HU bound must be smaller than the upper bound.")

    return np.clip(image, lower, upper).astype(np.float32)



def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Resize a 2D CT image to the target spatial resolution.

    Bilinear interpolation with antialiasing is applied only when
    the original image size differs from the target size.

    Parameters
    ----------
    image : np.ndarray
        2D CT image.
    target_size : tuple[int, int]
        Target image dimensions (height, width).

    Returns
    -------
    np.ndarray
        Resized image as a float32 array.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2D CT image.")

    if len(target_size) != 2 or any(s <= 0 for s in target_size):
        raise ValueError("Target size must contain two positive integers.")

    image = image.astype(np.float32, copy=False)

    if image.shape == target_size:
        return image

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return resized.squeeze(0).squeeze(0).numpy()



def normalize_image(
    image: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Normalize a windowed CT image to the [0, 1] range
    using fixed HU bounds.

    Parameters
    ----------
    image : np.ndarray
        Windowed CT image in Hounsfield Units.
    lower : float
        Lower HU bound used for windowing.
    upper : float
        Upper HU bound used for windowing.

    Returns
    -------
    np.ndarray
        Normalized image as a float32 array.
    """
    if lower >= upper:
        raise ValueError("The lower HU bound must be smaller than the upper bound.")

    if image.ndim != 2:
        raise ValueError("Expected a 2D CT image.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite intensity values.")

    image = image.astype(np.float32, copy=False)

    return (image - lower) / (upper - lower)
