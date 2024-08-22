import warnings
warnings.filterwarnings("ignore")

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import path
import sys
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from models.pytorch_tests.train_utils import train_multiclassification_model

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y  + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        x = self.attention(x, mask=None)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
#         super().__init__()
#         self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
#         self.flatten = nn.Flatten()
#         self.output_layer = nn.Linear(max_sequence_length * d_model, num_classes)
#         self.softmax = nn.Softmax()
#         # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     def forward(self, x):
#         x = self.layers(x)
#         x = self.flatten(x)
#         logits = self.output_layer(x)
#         probabilities = self.softmax(logits)
#         return probabilities

class Encoder(nn.Module):
    def __init__(self, num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_hidden, dropout=drop_prob)
        self.layers = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(max_sequence_length * d_model, num_classes)
        self.softmax = nn.Softmax()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        probabilities = self.softmax(logits)
        return probabilities

# ======================

def main():
    num_classes = 4
    num_layers = 1
    max_sequence_length = 10
    d_model = 50
    num_heads = 5
    drop_prob = 0.3
    ffn_hidden = 100

    model = Encoder(num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    print(model)
    # quit()

    input_data = np.random.rand(1000, max_sequence_length, d_model)
    target_data = np.random.randint(0, 4, 1000)
    print(f"Shapes: {input_data.shape}, {target_data.shape}")

    train_multiclassification_model(input_data, target_data, model)

    
if __name__ == "__main__":
    main()
