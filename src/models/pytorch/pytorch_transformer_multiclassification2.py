import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_hidden, dropout=drop_prob)
        self.layers = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(max_sequence_length * d_model, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        probabilities = self.sigmoid(logits)
        return probabilities

def main():
    num_classes = 4
    num_layers = 1
    max_sequence_length = 30
    d_model = 300
    num_heads = 5
    drop_prob = 0.3
    ffn_hidden = 100

    # model = nn.TransformerEncoder(encoder_layer, num_layers=6)
    model = Encoder(num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    inputs = torch.rand(100, 30, 300)
    outputs = torch.randint(0, 4, 100)
    out = model(inputs)
    print(out.shape)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

if __name__ == "__main__":
    main()