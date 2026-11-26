import torch

from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset
from src.data.dataloader import create_dataloader
from src.data.model_transforms import prepare_imagenet_input
from src.models.model_factory import create_model


def test_imagenet_pipeline():
    # 1. Load two real dataset samples
    dataset = LIDC_M3DSynth_Dataset(split="train")

    loader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    assert images.shape == (2, 1, 512, 512)
    assert labels.shape == (2,)
    assert torch.isfinite(images).all()
    assert images.min() >= 0
    assert images.max() <= 1

    # 2. Prepare input for ImageNet-pretrained RGB models
    images = prepare_imagenet_input(images)

    assert images.shape == (2, 3, 512, 512)
    assert torch.isfinite(images).all()

    # 3. Check that EfficientNet-B0 accepts the transformed batch
    # No pretrained weights are downloaded for this structural test.
    model = create_model(
        "efficientnet_b0",
        num_classes=2,
        pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
