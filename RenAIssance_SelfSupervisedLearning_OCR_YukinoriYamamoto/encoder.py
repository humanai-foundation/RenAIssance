import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from ResNet import ResNet18, ResNet34, ResNet50


class Encoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.resnet = ResNet18()
        self.resnet = ResNet50()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

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
