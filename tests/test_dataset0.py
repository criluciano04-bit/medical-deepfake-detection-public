import torch

from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset


def test_dataset():
    dataset = LIDC_M3DSynth_Dataset(
    split="train",
    repeat_channels=True,
)

    print(f"Training samples: {len(dataset)}")

    assert len(dataset) == 15000

    # Test one Real and one Fake sample
    for dataset_name, expected_label in [
        ("lidc", 0),
        ("m3dsynth", 1),
    ]:
        indices = dataset.samples.index[
            dataset.samples["dataset"] == dataset_name
        ]

        image, label = dataset[int(indices[0])]

        assert image.shape == (3, 512, 512)
        assert image.dtype == torch.float32
        assert torch.isfinite(image).all()
        assert 0 <= image.min() <= image.max() <= 1
        assert label.item() == expected_label

        # The three temporary channels must be identical
        assert torch.equal(image[0], image[1])
        assert torch.equal(image[1], image[2])

        print(
            f"{dataset_name}: "
            f"shape={tuple(image.shape)}, "
            f"label={label.item()}, "
            f"range=[{image.min():.3f}, "
            f"{image.max():.3f}]"
        )

    print("Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
