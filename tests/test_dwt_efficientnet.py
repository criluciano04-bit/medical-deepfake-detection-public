"""CPU smoke tests. Run from repo root with: python -m pytest -q tests/test_dwt_efficientnet.py"""
import torch
from src.data.model_transforms import prepare_imagenet_input
from src.models.dwt_efficientnet_b0 import FixedHaarDWT2D
from src.models.model_factory import create_model


def test_haar_shape_and_constant_signal():
    dwt = FixedHaarDWT2D()
    x = torch.ones(2, 1, 32, 32)
    bands = dwt(x)
    assert bands.shape == (2, 4, 16, 16)
    assert torch.allclose(bands[:, 0], torch.full_like(bands[:, 0], 2.0), atol=1e-6)
    assert torch.allclose(bands[:, 1:], torch.zeros_like(bands[:, 1:]), atol=1e-6)


def test_hybrid_backprop_reaches_both_branches():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = create_model("dwt_efficientnet_b0", num_classes=2, pretrained=False)
    model.train()
    grayscale = torch.rand(2, 1, 64, 64)
    images = prepare_imagenet_input(grayscale)
    logits = model(images)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    for name, module in (("spatial", model.spatial_branch),
                         ("frequency", model.frequency_branch),
                         ("classifier", model.classifier)):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads), f"No gradient in {name}"
        assert all(torch.isfinite(g).all() for g in grads), f"Nonfinite gradient in {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"All-zero gradient in {name}"
    assert not any(p.requires_grad for p in model.dwt.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.frequency_branch[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.frequency_branch[0].weight.detach())
