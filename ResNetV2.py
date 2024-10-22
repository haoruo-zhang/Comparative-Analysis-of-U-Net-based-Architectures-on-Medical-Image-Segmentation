import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = StdConv2d(cin, cmid, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = StdConv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = StdConv2d(cmid, cout, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = StdConv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    def __init__(self, n_in, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            StdConv2d(n_in, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            self._make_block(width, width * 4, block_units[0]),
            self._make_block(width * 4, width * 8, block_units[1], stride=2),
            self._make_block(width * 8, width * 16, block_units[2], stride=2),
        )

    def _make_block(self, cin, cout, blocks, stride=1):
        layers = [PreActBottleneck(cin, cout, cmid=cout // 4, stride=stride)]
        for _ in range(1, blocks):
            layers.append(PreActBottleneck(cout, cout, cmid=cout // 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))

            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 0 < pad < 3, f"x {x.size()} should {right_size}"
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x
            else:
                feat = x

            features.append(feat)

        x = self.body[-1](x)
        return x, features[::-1]


if __name__ == "__main__":
    tensor = torch.randn(1, 1, 224, 224)
    model = ResNetV2(1, [3, 4, 9], 1)
    output, features = model(tensor)
    print(output.shape)
    print(len(features))
    print("finish")
