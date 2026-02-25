import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from ResNet import ResNet18, ResNet34, ResNet50


class Encoder(nn.Module, PyTorchModelHubMixin):
    """
    Encoder that uses a ResNet backbone to extract a 1D feature sequence
    along the width dimension, then feeds it into a BiLSTM.

    Expected ResNet output shape: (B, C, H=1, T),
    where T is the sequence length (width axis).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        # Select backbone
        backbone = backbone.lower()
        if backbone == "resnet18":
            self.resnet = ResNet18()
            lstm_input_size = 512  # adjust to actual output channels of your ResNet18
        elif backbone == "resnet34":
            self.resnet = ResNet34()
            lstm_input_size = 512  # adjust to actual output channels of your ResNet34
        elif backbone == "resnet50":
            self.resnet = ResNet50()
            lstm_input_size = 512  # ResNet50 last_conv outputs 512 channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # BiLSTM over the width dimension
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        """
        x: (B, C, H, W)  e.g. (B, 3, 64, 384)

        Returns:
            rnn_output: (B, T, 2 * hidden_size)
            hidden: (h_n, c_n) where each is (num_layers * 2, B, hidden_size)
        """
        # ResNet should produce (B, C_out, H=1, T)
        resnet_output = self.resnet(x)
        # Squeeze height dimension and treat width as sequence length
        # (B, C_out, 1, T) -> (B, C_out, T)
        resnet_output = torch.squeeze(resnet_output, dim=2)
        # (B, C_out, T) -> (B, T, C_out)
        resnet_output = torch.permute(resnet_output, (0, 2, 1))
        # BiLSTM over sequence dimension T
        rnn_output, hidden = self.lstm(resnet_output)

        return rnn_output, hidden
