from src.models.model_factory import create_model


efficientnet = create_model("efficientnet_b0", pretrained=False)
convnext = create_model("convnext_tiny", pretrained=False)

print("EfficientNet:")
print(efficientnet.classifier)

print("\nConvNeXt:")
print(convnext.classifier)