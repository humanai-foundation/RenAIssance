import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
# Assuming these are custom implementations given the specific sequence length comments
from ResNet import ResNet18, ResNet34, ResNet50

class Encoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_type='resnet50'):
        super(Encoder, self).__init__()
        
        # Select Backbone and determine Feature Size
        if model_type == 'resnet50':
            self.resnet = ResNet50()
            enc_channels = 2048 # ResNet50 standard output
        else:
            self.resnet = ResNet18()
            enc_channels = 512  # ResNet18/34 standard output

        # LSTM Input Size must match Encoder Output
        self.lstm = nn.LSTM(
            input_size=enc_channels, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, x):
        features = self.resnet(x)
        # Instead of strict squeeze, we pool vertical features if H' > 1, 
        features = features.mean(dim=2) 
        # Prepare for LSTM (Batch, Seq_Len, Features)
        # Current: [Batch, Channels, Width] -> Permute to [Batch, Width, Channels]
        features = features.permute(0, 2, 1)  
        # Input: [Batch, Width (Seq Len), Channels (2048)]
        self.lstm.flatten_parameters() # Good practice for memory/speed on GPU
        rnn_output, hidden = self.lstm(features)
        
        # Output: [Batch, Seq Len, 512 (256*2)]
        return rnn_output, hidden
