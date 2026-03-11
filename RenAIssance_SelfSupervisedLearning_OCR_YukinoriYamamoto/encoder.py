import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from ResNet import ResNet18, ResNet34, ResNet50

BACKBONE_REGISTRY = {
    "resnet18": (ResNet18, 512),
    "resnet34": (ResNet34, 512),
    "resnet50": (ResNet50, 512),
}


class Encoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, backbone="resnet50"):
        super(Encoder, self).__init__()

        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        resnet_cls, lstm_input_size = BACKBONE_REGISTRY[backbone]
        self.resnet = resnet_cls()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # [batch size, channel(3), height(32), width(100)]
        resnet_output = self.resnet(x)
        # [batch size, feature length(512), 1, sequence length(22)]
        resnet_output = torch.squeeze(resnet_output, dim=2)
        resnet_output = torch.permute(resnet_output, (0, 2, 1))
        # [batch size, sequence length, feature length]
        rnn_output, hidden = self.lstm(resnet_output)
        # [batch size, sequence length, forward output length + backward output length(256)]
        return rnn_output, hidden
