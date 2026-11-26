import torch

from src.data.dataloader import create_dataloader
from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset


def test_dataloader():
    # Create the training dataset
    dataset = LIDC_M3DSynth_Dataset(
        split="train",
        repeat_channels=True,
    )

    # Create the DataLoader using the dataset
    loader = create_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    # Load one batch
    images, labels = next(iter(loader))

    # Validate batch properties
    assert images.shape == (4, 3, 512, 512)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all()
    assert (images >= 0).all()
    assert (images <= 1).all()
    assert torch.isin(
        labels, torch.tensor([0, 1])
    ).all()

    # Check the three channels are identical
    assert torch.equal(images[:, 0], images[:, 1])
    assert torch.equal(images[:, 1], images[:, 2])

    print(f"Batch images: {tuple(images.shape)}")
    print(f"Batch labels: {labels.tolist()}")
    print("DataLoader test passed!")


if __name__ == "__main__":
    test_dataloader()
