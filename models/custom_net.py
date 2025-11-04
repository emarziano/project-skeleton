import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self._to_linear = self._get_conv_output()
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
        )

    def _get_conv_output(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
