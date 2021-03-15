import torch
from torch import nn
from torch.cuda.amp import autocast
import timm
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='inception_v3', checkpoint_path=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone, num_classes=n_classes)

        if checkpoint_path is not None:
            pretrained_path = checkpoint_path
            checkpoint = torch.load(pretrained_path)['model']
            for key in list(checkpoint.keys()):
                if 'model.' in key:
                    checkpoint[key.replace('model.', '')] = checkpoint[key]
                    del checkpoint[key]
            self.model.load_state_dict(checkpoint)
            print(f'load {model_name} pretrained model')

    def forward(self, x):
        features = self.model(x)
        return features


if __name__ == '__main__':
    model = Inception(11, pretrained_backbone=False, mixed_precision=False, model_name='inception_v3',
                      checkpoint_path='inception_v3_chestx.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print(summary(model, (3, 512, 512)))
