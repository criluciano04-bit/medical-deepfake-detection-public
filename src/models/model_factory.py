import torch.nn as nn
from torchvision import models

def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True):

    model_name = model_name.lower() #togli letter maiuscole

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features   #classifier[1] è il layer finale (l'indice cambia in base al modello), in_featuer è l'input del layer
        model.classifier[1] = nn.Linear(in_features, num_classes) #sostituzione ultimo layer per avere 2 logits (fake and true), invece dei 1000 di efficientnet

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == "dwt_efficientnet_b0":
        from src.models.dwt_efficientnet_b0 import DWTEfficientNetB0
        model = DWTEfficientNetB0(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model