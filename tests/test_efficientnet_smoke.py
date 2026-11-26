import torch
import torch.nn as nn

from src.data.dataloader import create_dataloader
from src.data.datasets.lidc_m3dsynth import LIDC_M3DSynth_Dataset
from src.models.model_factory import create_model


def test_efficientnet_smoke():
    torch.manual_seed(42)

    # 1. Load a small batch from the training dataset
    dataset = LIDC_M3DSynth_Dataset(
        split="train",
        repeat_channels=True,
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    # 2. Initialize EfficientNet-B0 on CPU
    model = create_model(
        "efficientnet_b0",
        num_classes=2,
        pretrained=False,
    )

    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    images = images.to(device)
    labels = labels.to(device)

    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # 4. Forward pass
    optimizer.zero_grad()

    outputs = model(images)

    assert outputs.shape == (2, 2)
    assert torch.isfinite(outputs).all()

    loss = criterion(outputs, labels)

    assert torch.isfinite(loss)

    # 5. Backpropagation
    loss.backward()

    # Check gradients exist and are finite
    gradients = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert gradients, "No gradients were computed."
    assert all(torch.isfinite(g).all() for g in gradients)

    # 6. Update parameters
    weights_before = model.classifier[1].weight.detach().clone()

    optimizer.step()

    weights_after = model.classifier[1].weight.detach()

    assert not torch.equal(weights_before, weights_after), (
        "Model parameters were not updated."
    )

    print("Images:", tuple(images.shape))
    print("Labels:", labels.tolist())
    print("Outputs:", tuple(outputs.shape))
    print("Loss:", loss.item())
    print("Gradients: OK")
    print("Weight update: OK")
    print("EfficientNet-B0 smoke test passed!")


if __name__ == "__main__":
    test_efficientnet_smoke()
