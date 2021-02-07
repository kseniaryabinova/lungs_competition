import timm
import torch
from torch import nn
from torch.cuda.amp import autocast
from torchsummary import summary


class ViT(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='ViT-B_16', checkpoint_path=None):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

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
                x = self.model(x)
        else:
            x = self.model(x)
        return x


if __name__ == '__main__':
    model = ViT(11, pretrained_backbone=False, mixed_precision=False, model_name='vit_base_patch16_384')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(summary(model, (3, 384, 384)))
