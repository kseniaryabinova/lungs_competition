import torch
from torch import nn
from torch.cuda.amp import autocast
import timm
from torchsummary import summary


class EfficientNet(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='tf_efficientnet_b5_ns', checkpoint_path=None):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_features, momentum=0.99, eps=1e-3, affine=True),
            nn.Linear(n_features, n_classes)
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            self.load_state_dict(new_state_dict)

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


class EfficientNetNoisyStudent(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='tf_efficientnet_b5_ns', checkpoint_path=None):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(n_features, n_classes)
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            self.load_state_dict(new_state_dict)

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



class EfficientNetB5(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b5_ns'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, 11)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class EfficientNet3Stage(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='tf_efficientnet_b5_ns', checkpoint_path=None):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_features, momentum=0.99, eps=1e-3, affine=True),
            nn.Linear(n_features, n_classes)
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            self.load_state_dict(new_state_dict)

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
        return features, pooled_features, x


if __name__ == '__main__':
    model = EfficientNet(11, pretrained_backbone=False, mixed_precision=False, model_name='tf_efficientnet_b7_ns')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)
    print(summary(model, (3, 300, 300)))
