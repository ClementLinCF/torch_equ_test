import torch
import torch.nn as nn

# Assuming you can modify hidden_states to have 7 columns
hidden_states = torch.randn(3, 7)  # Adjust the shape to (batch_size, embedding_size)

embedding = torch.randn(5, 7)  # (embedding_size, hidden_size)

# Ensure compatible dimensions for matrix multiplication (no change needed here)

# Compute logits using torch.matmul
matmul_result = torch.matmul(hidden_states, embedding.t())

# Replace torch.matmul with torch.nn.Linear
linear_layer = nn.Linear(embedding.size(1), embedding.size(0), bias=True)  # Use input and output dimensions

# Set weight directly to embedding
linear_layer.weight.data = embedding

# Set bias to zero if not needed (assuming bias is False)
linear_layer.bias.data.zero_()  # This line should work now

linear_result = linear_layer(hidden_states)

# Check if the results are the same
print(torch.allclose(matmul_result, linear_result))

