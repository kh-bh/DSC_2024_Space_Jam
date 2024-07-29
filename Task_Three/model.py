import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv 1D: filters = 64, kernel = 3, stride =2 ,padding = 1
# Maxpool 1D: kernel = 3, stride = 2, padding = 1
# Fire module: sq =16, ex =64
# Fire module: sq=16,ex=64
# Maxpool 1D: kernel = 3, stride = 2, padding = 1
# Fire: sq=32, ex = 128
# Fire: sq=32, ex = 128
# Maxpool 1D: kernel = 3, stride = 2, padding = 1
# Fire: sq=48, ex = 192
# Fire: sq=48, ex = 192
# Fire: sq=64, ex = 256
# Fire: sq=64, ex = 256
# Dropout: p = 0.1
# Conv 1D: filters = 75, kernel = 3, stride =2 ,padding = 0
# Average pool 1D
# Conv 1D: filters = 75, kernel = 3, stride =2 ,padding = 0

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu(x)
        out1x1 = self.expand1x1(x)
        out3x3 = self.expand3x3(x)
        out = torch.cat([out1x1, out3x3], dim=1)
        return self.relu(out)


class SqueezeNet1D(nn.Module):
    def __init__(self, num_classes=75):
        super(SqueezeNet1D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
            nn.Dropout(p=0.1),
        )

        # Adjust the classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(512, 75, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(3),
            nn.Conv1d(75, 75, kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #x = x.reshape(32, 1, 75)
        #x = torch.flatten(x, 1)
        return x


# Example usage
if __name__ == "__main__":
    # Assuming input shape is (batch_size, 12, 500)
    model = SqueezeNet1D()
    input_tensor = torch.randn(32, 12, 500)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([32, 1, 75])
