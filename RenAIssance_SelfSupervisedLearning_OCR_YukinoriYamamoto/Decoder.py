import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # hidden: (4, batch_size, hidden_size)
        # modify the shape of hidden to (batch_size, seq_length, hidden_size)
        hidden = hidden.mean(dim=0).unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        # now, hidden: (batch_size, seq_length, hidden_size)
        encoder_outputs = encoder_outputs.view(-1, encoder_outputs.size(1), 2, self.hidden_size).sum(2)
        # encoder_outputs: (batch_size, seq_length, hidden_size)

        attn_weights = F.softmax(torch.sum(hidden * encoder_outputs, dim=2), dim=1).unsqueeze(1)
        # attn_weights: (batch_size, 1, seq_length)
        weighted_sum = torch.bmm(attn_weights, encoder_outputs)
        # weighted_sum: (batch_size, 1, hidden_size)
        return weighted_sum

class LSTMAttnDecoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(LSTMAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        # embedding: (output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)
        # out: (4*hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step: (batch_size, 1)
        embedded = self.embedding(input_step)
        # embedded: (batch_size, 1, hidden_size)
        embedded = self.dropout(embedded)
        attn_weighted = self.attention(last_hidden[0], encoder_outputs)
        # attn_weighted: (batch_size, 1, hidden_size)
        rnn_input = torch.cat((embedded, attn_weighted), 2)
        # rnn_input: (batch_size, 1, 2*hidden_size)
        output, hidden = self.lstm(rnn_input, last_hidden)
        # output: (batch_size, seq_length, 2*hidden_size) because of bidirectional=True
        # hidden: a tuple of two tensors both are (num_layers*2, batch_size, hidden_size)
        context = attn_weighted.squeeze(1)
        # context: (batch_size, hidden_size)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # output: (batch_size, seq_length, hidden_size)
        output = output.squeeze(1)
        # output: (batch_size, hidden_size) if seq_length=1
        output = self.out(torch.cat((output, context), 1))
        # output: (batch_size, output_size)
        return output, hidden