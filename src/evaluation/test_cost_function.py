import torch

# Single instance
weights = torch.tensor([[1.0, 0.5, 0.5],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.5, 1.0]])
L = torch.tensor([[1, 1, 1]])
R = torch.tensor([[1.0, 0.1, 0.1]])
x = torch.ones(3)
N = ((L + R - L * R) * x.t()) * x + 1e-6 # add 1e-6 to avoid division by 0
A = L.t() * (R / N)
custom_loss = (weights * A).sum()
print(custom_loss.numpy())

# Batch
b = 2
weights = torch.tensor([[1.0, 0.5, 0.5],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.5, 1.0]])
L = torch.tensor([[1, 1, 1]] * b).float() # Shape: (b, 3)
R = torch.tensor([[1.0, 0.1, 0.1]] * b).float()
x = torch.ones(3)
N = ((L + R - L * R) * x.t()) * x + 1e-6 # add 1e-6 to avoid division by 0
A = (R / N) * L
custom_loss = (weights * A.unsqueeze(1)).sum(dim=(1, 2)).sum()
print(custom_loss.numpy())
