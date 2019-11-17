import torch.nn as nn
import torch
from typing import Tuple
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
import math
from collections import namedtuple


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        nn.ReLU6(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def conv_bn(inp, oup, stride, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.ReLU6(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])


class CustomSSD(nn.Module):
    def __init__(self, num_classes=1, num_points=68, width_mult=1.0, use_batch_norm=True):
        super(CustomSSD, self).__init__()
        self.base = MobileNetV2().features
        self.source_layer_indexes = [GraphPath(14, 'conv', 3), 19, ]

        self.extras = ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            Conv2d(256, 64, kernel_size=2, stride=1, padding=0)
        ])

        self.regression_headers = ModuleList([
            SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                            kernel_size=3, padding=1),
            SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])

        self.classification_headers = ModuleList([
            SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3,
                            padding=1),
            SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])

        self.landmarks_headers = ModuleList([
            SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_points * 2, kernel_size=3,
                            padding=1),
            SeperableConv2d(in_channels=1280, out_channels=6 * num_points * 2, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * num_points * 2, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_points * 2, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_points * 2, kernel_size=3, padding=1),
            Conv2d(in_channels=64, out_channels=6 * num_points * 2, kernel_size=1),
        ])

        self.source_layer_add_ons = nn.ModuleList([t[1] for t in self.source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])

        self.num_classes = num_classes
        self.num_points = num_points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        landmarks = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location, landmark = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landmarks.append(landmark)

        for layer in self.base[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location, landmark = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landmarks.append(landmark)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        landmarks = torch.cat(landmarks, 1)
        return confidences, locations, landmarks

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        landmarks = self.landmarks_headers[i](x)
        landmarks = landmarks.permute(0, 2, 3, 1).contiguous()
        landmarks = landmarks.view(location.size(0), -1, self.num_points * 2)

        return confidence, location, landmarks

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
