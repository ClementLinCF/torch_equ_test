import torch
import torch.nn as nn

# Example input tensor and parameters
x = torch.randn(10, 20)  # Input tensor with shape (batch_size, in_features)
weight = torch.randn(30, 20)  # Weight tensor with shape (out_features, in_features)
bias = torch.randn(30)  # Bias tensor with shape (out_features)

# Using F.linear
output_f_linear = torch.nn.functional.linear(x, weight, bias)

# Using torch.nn.Linear
linear_layer = nn.Linear(20, 30)  # Initialize Linear layer with the same dimensions
linear_layer.weight.data = weight  # Set the weight of the Linear layer
linear_layer.bias.data = bias  # Set the bias of the Linear layer
output_torch_nn_linear = linear_layer(x)

# Check if the outputs are the same
print(torch.allclose(output_f_linear, output_torch_nn_linear))  # Should print True

