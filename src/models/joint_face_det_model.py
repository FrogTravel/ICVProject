from torch import nn
import torch


class SELayer(nn.Module):
    """
    Squeeze-Excitation block:
    https://arxiv.org/abs/1709.01507
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel):
        super(SELayer, self).__init__()
        reduction = channel // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # Equal to torch.einsum('bck,ck->bc', A, B)
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)


class Fire(nn.Module):
    """
    Fire Module:
    https://arxiv.org/abs/1602.07360
    https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, stride=None):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        if stride:
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, stride=stride,
                                       kernel_size=1)
            self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                       kernel_size=3, padding=1, stride=stride)

        else:
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                       kernel_size=1)
            self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                       kernel_size=3, padding=1)

        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        x1 = self.expand1x1_activation(self.expand1x1(x))
        x2 = self.expand3x3_activation(self.expand3x3(x))
        cat = torch.cat([x1, x2], 1)
        return cat


class FireSE(nn.Module):
    """
    Fire_SE module composed of Fire module followed by SE one repeated two times
    from the paper:
    "made up of Fire Modules(FM) of SqueezeNet (with 16 and 64 filters in squeeze and expand layers respectively)
    and squeeze-and-excite modules(SE) in order to compress the model size and reduce the model execution time
    without compromising the accuracy"
    """

    def __init__(self, inplanes, squeeze_1,
                 expand1x1_1, expand3x3_1, squeeze_2, expand1x1_2, expand3x3_2, stride=None):
        super(FireSE, self).__init__()
        self.fire1 = Fire(inplanes, squeeze_1,
                          expand1x1_1, expand3x3_1)
        self.se1 = SELayer(expand1x1_1 + expand3x3_1)
        if stride:
            self.fire2 = Fire(expand1x1_1 + expand3x3_1, squeeze_2,
                              expand1x1_2, expand3x3_2, stride=2)
        else:
            self.fire2 = Fire(expand1x1_1 + expand3x3_1, squeeze_2,
                              expand1x1_2, expand3x3_2)
        self.se2 = SELayer(expand1x1_2 + expand3x3_2)

    def forward(self, x):
        out = self.fire1.forward(x)
        out = self.se1.forward(out)
        out = self.fire2.forward(out)
        out = self.se2.forward(out)
        return out


class JointDetectionModule(nn.Module):

    def __init__(self):
        # 7 × 7 convolution layer with 64 filters and stride 2
        # input = 288x288
        super(JointDetectionModule, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # first (bounding box) branch
        self.module1_1 = nn.Sequential(FireSE(64, 16, 64, 64, 16, 64, 64), nn.MaxPool2d(kernel_size=2))  # 36x36x64
        self.module1_2 = nn.Sequential(FireSE(128, 16, 64, 64, 16, 64, 64), nn.MaxPool2d(kernel_size=2))  # 18x18x128
        self.module1_3 = nn.Sequential(FireSE(128, 16, 64, 64, 16, 64, 64), nn.MaxPool2d(kernel_size=2))  # 9x9x128
        self.fc1_1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True))
        self.fc1_2 = nn.Sequential(
            nn.Linear(128,
                      5 * 5))  # output of this branch = (9 × 9 × 5 × (4+1))  bounding box coordinates (4), confidence(1), anchor boxes(5)

        self.act_coordinates = nn.ReLU(inplace=True)
        self.act_confidence = nn.Sigmoid()

        # fourth (expression) branch used for landmarks
        self.fire4_1 = FireSE(64, 16, 64, 64, 16, 64, 64, stride=2)
        self.fire4_2 = FireSE(256, 16, 64, 64, 16, 64, 64, stride=2)
        self.fire4_3 = FireSE(256, 16, 64, 64, 16, 64, 64, stride=2)
        self.fc4_1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True))
        self.fc4_2 = nn.Sequential(
            nn.Linear(128, 5 * 68 * 2),  # output of this branch = (9 × 9 × 5 × (68*2)) for 2d landmarks
            nn.ReLU(inplace=True))

    def forward(self, x):
        # 1st branch - bbox
        # 4th - Landmarks

        x = self.conv(x)  # 1x64x144x144
        x = self.pooling(x)  # 1x64x72x72

        # first "column" from the architecture figure
        out1_1 = self.module1_1.forward(x)
        out4_1 = self.fire4_1.forward(x)

        # second column
        out1_2 = self.module1_2.forward(out1_1)
        out4_2 = self.fire4_2.forward(
            torch.cat([out1_1, out4_1], 1))

        # third column
        out1_3 = self.module1_3.forward(out1_2)
        out4_3 = self.fire4_3.forward(torch.cat([out1_2, out4_2], 1))

        # fc 1
        out1 = out1_3.view((-1, 9, 9, 128))
        out4 = out4_3.view((-1, 9, 9, 128))
        out1 = self.fc1_1(out1)
        out4 = self.fc4_1(out4)

        # fc 2
        bbox = self.fc1_2(out1)
        bbox = bbox.view(-1, 9, 9, 5, 5)
        # split bbox data into coordinates and confidence data to apply different activation functions
        bbox_coordinates = bbox[:, :, :, :, :4]
        bbox_confidence = bbox[:, :, :, :, 4:]
        bbox_coordinates = self.act_coordinates(bbox_coordinates)  # relu for coordinates
        bbox_confidence = self.act_confidence(bbox_confidence)  # sigmoid for confidence
        landmarks = self.fc4_2(out4)

        landmarks = torch.exp(landmarks)

        bbox = torch.cat([bbox_coordinates, bbox_confidence], dim=4)
        landmarks = landmarks.view(-1, 9, 9, 5, 68 * 2)
        return bbox, landmarks
