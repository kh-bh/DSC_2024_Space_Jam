import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(squeeze_planes)
        self.LeakyReLU = nn.LeakyReLU(inplace=False)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(expand1x1_planes)
        #self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(expand3x3_planes)
        #self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        out1x1 = self.expand1x1(x)
        out1x1 = self.bn2(out1x1)
        out1x1 = self.LeakyReLU(out1x1)
        out3x3 = self.expand3x3(x)
        out3x3 = self.bn3(out3x3)
        out3x3 = self.LeakyReLU(out3x3)
        out = torch.cat([out1x1, out3x3], dim=1)
        return self.LeakyReLU(out)


class ResidualFire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(ResidualFire, self).__init__()
        self.fire = Fire(inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes)
        self.residual = nn.Conv1d(inplanes, expand1x1_planes + expand3x3_planes, kernel_size=1) if inplanes != (expand1x1_planes + expand3x3_planes) else None
        self.LeakyReLU = nn.LeakyReLU(inplace=False)  # Changed to out-of-place

    def forward(self, x):
        identity = x
        out = self.fire(x)
        if self.residual is not None:
            identity = self.residual(identity)
        return self.LeakyReLU(out + identity)  # Changed order of operations


class SqueezeNet1D(nn.Module):
    def __init__(self, num_classes=75):
        super(SqueezeNet1D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ResidualFire(64, 16, 64, 64),
            ResidualFire(128, 16, 64, 64),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ResidualFire(128, 32, 128, 128),
            ResidualFire(256, 32, 128, 128),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ResidualFire(256, 48, 192, 192),
            ResidualFire(384, 48, 192, 192),
            ResidualFire(384, 64, 256, 256),
            ResidualFire(512, 64, 256, 256),
            nn.Dropout(p=0.1),
        )

        # Adjust the classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(512, 75, kernel_size=1, padding=0),
            # nn.BatchNorm1d(75),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(75, 75, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Assuming input shape is (batch_size, 12, 500)
    model = SqueezeNet1D()
    input_tensor = torch.randn(32, 12, 500)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([32, 75, 500])
