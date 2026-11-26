import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def repeat_to_three_channels(image: torch.Tensor) -> torch.Tensor:
    """Repeat the grayscale channel for a single image or a batch."""

    if image.ndim == 3 and image.shape[0] == 1:
        return image.repeat(3, 1, 1)

    if image.ndim == 4 and image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)

    raise ValueError("Expected shape (1, H, W) or (B, 1, H, W).")


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a single image or a batch."""

    if image.ndim == 3 and image.shape[0] == 3:
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)

    elif image.ndim == 4 and image.shape[1] == 3:
        mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)

    else:
        raise ValueError("Expected shape (3, H, W) or (B, 3, H, W).")

    return (image - mean) / std


def prepare_imagenet_input(image: torch.Tensor) -> torch.Tensor:
    """Prepare grayscale CT images for ImageNet-pretrained RGB models."""

    image = repeat_to_three_channels(image)
    image = normalize_imagenet(image)

    return image