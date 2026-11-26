"""Quick real-resolution forward/backward CUDA check, no dataset or checkpoints.

From repository root, with CUDA available:
    python -m scripts.training.smoke_dwt_gpu --batch-size 2

No optimizer step: verifies loss.backward and peak GPU memory for the model.
"""
import argparse
import torch
from src.data.model_transforms import prepare_imagenet_input
from src.models.model_factory import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error('Batch size must be >=2 for BatchNorm during training')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available')
    torch.manual_seed(42)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    model = create_model('dwt_efficientnet_b0', pretrained=True).to(device)
    model.train()
    images = torch.rand(args.batch_size, 1, 512, 512, device=device)
    logits = model(prepare_imagenet_input(images))
    loss = torch.nn.functional.cross_entropy(
        logits, torch.arange(args.batch_size, device=device) % 2
    )
    loss.backward()
    for name, branch in (
        ('spatial', model.spatial_branch),
        ('frequency', model.frequency_branch),
        ('classifier', model.classifier),
    ):
        gradients = [p.grad for p in branch.parameters() if p.requires_grad]
        if not gradients or not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f'Missing/nonfinite gradients: {name}')
        print(f'{name}: gradients OK')
    print('Output shape:', tuple(logits.shape))
    print('Loss:', float(loss.detach()))
    print('Peak GPU allocated (GiB):', round(torch.cuda.max_memory_allocated(device)/1024**3, 3))
    print('Smoke test PASSED')


if __name__ == '__main__':
    main()
