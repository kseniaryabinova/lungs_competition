import torch
from torch import nn
from torch.cuda.amp import autocast
import timm
from torchsummary import summary


class EfficientNet(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision, model_name='tf_efficientnet_b5_ns'):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, 11)

    def forward(self, x):
        if self.amp:
            with autocast():
                bs = x.size(0)
                features = self.model(x)
                pooled_features = self.pooling(features).view(bs, -1)
                x = self.classifier(pooled_features)
        else:
            bs = x.size(0)
            features = self.model(x)
            pooled_features = self.pooling(features).view(bs, -1)
            x = self.classifier(pooled_features)
        return x


if __name__ == '__main__':
    model = EfficientNet(11, pretrained_backbone=False, mixed_precision=False, model_name='tf_efficientnet_b5_ns')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(summary(model, (3, 300, 300)))
