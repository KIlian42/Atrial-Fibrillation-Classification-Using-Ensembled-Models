import torch
from torch import nn
from torch.nn.functional import softmax

import numpy as np
from math import sqrt

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

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, hidden)
        self.linear2 = nn.Linear(hidden, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_emb_dim = emb_dim // num_heads
        
        self.w_q = nn.ModuleList([nn.Linear(self.head_emb_dim, self.head_emb_dim) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(self.head_emb_dim, self.head_emb_dim) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(self.head_emb_dim, self.head_emb_dim) for _ in range(num_heads)])
        self.w_o = nn.Linear(emb_dim, emb_dim)

    def _scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attention_values = torch.matmul(q, k.transpose(-1, -2))
        scaled_attention_values = attention_values / sqrt(d_k)
        softmaxed_attention_values = softmax(scaled_attention_values, dim=-1)
        return torch.matmul(softmaxed_attention_values, v)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()
        q = torch.cat([layer(x[:, :, i*self.head_emb_dim:(i+1)*self.head_emb_dim]) for i, layer in enumerate(self.w_q)], dim=-1)
        k = torch.cat([layer(x[:, :, i*self.head_emb_dim:(i+1)*self.head_emb_dim]) for i, layer in enumerate(self.w_k)], dim=-1)
        v = torch.cat([layer(x[:, :, i*self.head_emb_dim:(i+1)*self.head_emb_dim]) for i, layer in enumerate(self.w_v)], dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
        values = self._scaled_dot_product(q, k, v)
        values = values.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        out = self.w_o(values)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNormalization(parameters_shape=[emb_dim])
        self.ffn = FeedForward(emb_dim=emb_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[emb_dim])

    def forward(self, x):
        residual_x = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, sequence_length, num_classes, d_model, ffn_hidden, num_heads, drop_prob, num_layers, positional_encoding=False):
        super().__init__()
        if positional_encoding:
            self.positional_encoding = self._get_positional_encoding(sequence_length, d_model)
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(sequence_length * d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_positional_encoding(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :]
        return torch.tensor(pe, dtype=torch.float32)
    
    def forward(self, x):
        if self.positional_encoding is not None:
            x = x + self.positional_encoding
        x = self.layers(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        probabilities = self.sigmoid(logits)
        return probabilities
    
# ======================
from torch.utils.data import DataLoader, TensorDataset

def main():
    num_classes = 26
    num_layers = 8
    sequence_length = 10
    d_model = 200
    num_heads = 10
    drop_prob = 0.1
    ffn_hidden = 24
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    model = Encoder(sequence_length, num_classes, d_model, ffn_hidden, num_heads, drop_prob, num_layers, True)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    input_data = torch.rand(1000, sequence_length, d_model)
    target_data = torch.randint(0, 2, (1000, num_classes))

    dataset = TensorDataset(input_data, target_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')
    
if __name__ == "__main__":
    main()
    
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_dim, num_heads):
#         super().__init__()
#         assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
#         self.num_heads = num_heads
#         self.head_emb_dim = emb_dim // num_heads
#         self.w_q = nn.Linear(emb_dim, emb_dim)
#         self.w_k = nn.Linear(emb_dim, emb_dim)
#         self.w_v = nn.Linear(emb_dim, emb_dim)
#         self.w_o = nn.Linear(emb_dim, emb_dim)

#     def _scaled_dot_product(self, q, k, v):
#         d_k = q.size()[-1]
#         attention_values = torch.matmul(q, k.transpose(-1, -2))
#         scaled_attention_values = attention_values / sqrt(d_k)
#         softmaxed_attention_values = softmax(scaled_attention_values, dim=-1)
#         return torch.matmul(softmaxed_attention_values, v)

#     def forward(self, x):
#         batch_size, seq_len, emb_dim = x.size()
#         q = self.w_q(x)
#         k = self.w_k(x)
#         v = self.w_v(x)
#         q = q.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
#         k = k.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
#         v = v.view(batch_size, seq_len, self.num_heads, self.head_emb_dim).transpose(1, 2)
#         values = self._scaled_dot_product(q, k, v)
#         values = values.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
#         out = self.w_o(values)
#         return out