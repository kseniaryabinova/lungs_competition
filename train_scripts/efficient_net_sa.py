import torch
from torch import nn
from torch.cuda.amp import autocast
import timm
from torchsummary import summary

""" Position attention module"""


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = torch.softmax(energy_new, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self):
        super(CBAM, self).__init__()
        inter_channels = in_channels // 4
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())

        self.conv1_s = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())

        self.channel_gate = CAM_Module(inter_channels)
        self.spatial_gate = PAM_Module(inter_channels)

        self.conv2_c = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.conv2_a = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

    def forward(self, x):
        feat1 = self.conv1_c(x)
        chnl_att = self.channel_gate(feat1)
        chnl_att = self.conv2_c(chnl_att)

        feat2 = self.conv1_s(x)
        spat_att = self.spatial_gate(feat2)
        spat_att = self.conv2_a(spat_att)

        x_out = chnl_att + spat_att

        return x_out


class EfficientNetSA(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision,
                 model_name='tf_efficientnet_b5_ns', checkpoint_path=None):
        super().__init__()
        self.amp = mixed_precision
        self.model = timm.create_model(model_name, pretrained=pretrained_backbone)

        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.local_fe = CBAM(n_features)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(nn.Linear(n_features + n_features, n_features),
                                        nn.BatchNorm1d(n_features),
                                        nn.Dropout(0.1),
                                        nn.ReLU(),
                                        nn.Linear(n_features, n_classes))

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
                enc_feas = self.model(x)

                # use default's global features
                global_feas = self.pooling(enc_feas)
                global_feas = global_feas.flatten(start_dim=1)
                global_feas = self.dropout(global_feas)

                local_feas = self.local_fe(enc_feas)
                local_feas = torch.sum(local_feas, dim=[2, 3])
                local_feas = self.dropout(local_feas)

                all_feas = torch.cat([global_feas, local_feas], dim=1)
                outputs = self.classifier(all_feas)
        else:
            enc_feas = self.model(x)

            # use default's global features
            global_feas = self.pooling(enc_feas)
            global_feas = global_feas.flatten(start_dim=1)
            global_feas = self.dropout(global_feas)

            local_feas = self.local_fe(enc_feas)
            local_feas = torch.sum(local_feas, dim=[2, 3])
            local_feas = self.dropout(local_feas)

            all_feas = torch.cat([global_feas, local_feas], dim=1)
            outputs = self.classifier(all_feas)
        return outputs


if __name__ == '__main__':
    model = EfficientNetSA(11, pretrained_backbone=False, mixed_precision=False, model_name='tf_efficientnet_b5_ns')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    model.to(device)
    print(summary(model, (3, 512, 512)))
